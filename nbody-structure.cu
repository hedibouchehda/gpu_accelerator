#include <stdio.h>
#include <math.h>
#include <stdlib.h> // drand48
#include <time.h>

//#define DUMP


__global__ void MoveParticles(const int nParticles,float* x, float* y, float* z, float* vx ,
        float* vy, float* vz, const float dt) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  // Loop over particles that experience force
  float Fx = 0, Fy = 0, Fz = 0;  
    // Components of the gravity force on particle i
    // Loop over positions that exert force 
    for (int j = 0; j < nParticles; j++) { 
      // No self interaction
      if (i != j) {
          // Avoid singularity and interaction with self
          const float softening = 1e-20;

          // Newton's law of universal gravity
          const float dx = x[j] - x[i];
          const float dy = y[j] - y[i];
          const float dz = z[j] - z[i];
          const float drSquared  = dx*dx + dy*dy + dz*dz + softening;
          const float drPower32  = powf(drSquared, 3.0/2.0);
            
          // Calculate the net force
          Fx += dx / drPower32;  
          Fy += dy / drPower32;  
          Fz += dz / drPower32;
      }
    }


    // Accelerate particles in response to the gravitational force
    vx[i] += dt*Fx; 
    vy[i] += dt*Fy; 
    vz[i] += dt*Fz;
  

  // Move particles according to their velocities
  // O(N) work, so using a serial loop
  //#pragma acc parallel loop
    x[i]  += vx[i]*dt;
    y[i]  += vy[i]*dt;
    z[i]  += vz[i]*dt;
}

void dump(int iter, int nParticles, float* x, float* y ,float* z)
{
    char filename[64];
    snprintf(filename, 64, "output_cuda_%d.txt", iter);

    FILE *f;
    f = fopen(filename, "w+");

    int i;
    for (i = 0; i < nParticles; i++)
    {
        fprintf(f, "%e %e %e\n",
        x[i], y[i], z[i]);
    }

    fclose(f);
}

int main(const int argc, const char** argv)
{

  // Problem size and other parameters
  const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  // Duration of test
  const int nSteps = (argc > 2)?atoi(argv[2]):10;
  // Particle propagation time step
  const float dt = 0.0005f;

  float* x = (float*)malloc(nParticles*sizeof(float));
  float* y = (float*)malloc(nParticles*sizeof(float));
  float* z = (float*)malloc(nParticles*sizeof(float));
  float* vx = (float*)malloc(nParticles*sizeof(float));
  float* vy = (float*)malloc(nParticles*sizeof(float));
  float* vz = (float*)malloc(nParticles*sizeof(float));

  // Initialize random number generator and particles
  srand48(0x2020);

  int i;
  for (i = 0; i < nParticles; i++)
  {
    x[i] =  2.0*drand48() - 1.0;
    y[i] =  2.0*drand48() - 1.0;
    z[i] =  2.0*drand48() - 1.0;
    vx[i]    = 2.0*drand48() - 1.0;
    vy[i]    = 2.0*drand48() - 1.0;
    vz[i]    = 2.0*drand48() - 1.0;
  }
  
  // Perform benchmark
  printf("\nPropagating %d particles using 1 thread...\n\n", 
	 nParticles
	 );
  float rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration (warm-up)
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  for (int step = 1; step <= nSteps; step++) {

    float *d_x,*d_y,*d_z,*d_vx,*d_vy,*d_vz ; 
    
    size_t size = nParticles*sizeof(float);
    cudaMalloc(&d_x, size);cudaMalloc(&d_y, size); cudaMalloc(&d_z, size);
    cudaMalloc(&d_vx, size);cudaMalloc(&d_vy, size); cudaMalloc(&d_vz, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, vx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz   , vz, size, cudaMemcpyHostToDevice);

    int threadPerBlocs = 256;
    /* Ceil */
    int blocksPerGrid   = (nParticles + threadPerBlocs - 1) / threadPerBlocs;

    clock_t tStart = clock(); // Start timing
    MoveParticles<<< blocksPerGrid, threadPerBlocs >>>(nParticles,d_x,d_y,d_z,d_vx,d_vy,d_vz, dt);
    clock_t tEnd = clock(); // End timing 
    float time_spent = (tStart - tEnd)/ CLOCKS_PER_SEC;

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(z, d_z, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(vx, d_vx, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(vy, d_vy, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(vz  , d_vz, size, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y);cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy);cudaFree(d_vz);

    const float HztoInts   = ((float)nParticles)*((float)(nParticles-1)) ;
    const float HztoGFLOPs = 20.0*1e-9*((float)(nParticles))*((float)(nParticles-1));

    if (step > skipSteps) { // Collect statistics
      rate  += HztoGFLOPs/(time_spent); 
      dRate += HztoGFLOPs*HztoGFLOPs/((time_spent)*(time_spent)); 
    }

    printf("%5d %10.3e %10.3e %8.1f %s\n", 
	   step, (time_spent), HztoInts/(time_spent), HztoGFLOPs/(time_spent), (step<=skipSteps?"*":""));
    fflush(stdout);

#ifdef DUMP
    dump(step, nParticles, x,y,z);
#endif
  }
  rate/=(float)(nSteps-skipSteps); 
  dRate=sqrt(dRate/(float)(nSteps-skipSteps)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");
  free(x);free(y);free(z);
  free(vx);free(vz);free(vz); 
  return 0;
}


