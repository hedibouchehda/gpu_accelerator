le fichier nbody.cu contient notre version du code optimisé par CUDA en utilisant le type 
ParticleType pour le compiler : 
nvcc nbody.cu -o exe_aos 

le fichier nbody.cu contient notre version du code optimisé par CUDA sans le type 
ParticleType pour le compiler : 
nvcc nbody.cu -o exe_soa 

le ficheier nbody.c contient notre version du code optimisé par des directives openACC 
pour le compiler : 
pgcc -acc -Minfo=accel -o exe 

