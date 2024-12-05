MKLINCLUDE:=-m64  -I"${MKLROOT}/include"
MKLLIB:= -m64  -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
CUDA_LIB:=-lcublas

all: gemm cuda_gemm climb

gemm: gemm.cu utils.h
	nvcc -o $@ $< $(MKLINCLUDE) $(MKLLIB) $(CUDA_LIB)
cuda_gemm: cuda_gemm.cu utils.h
	nvcc -o $@ $< $(MKLINCLUDE) $(MKLLIB) $(CUDA_LIB)

clean:
	rm -f gemm cuda_gemm *.o

climb:
	echo Hello, Lets Climb Toubkal...
	ls -la
	module load CUDA
	module load GCC/12.3.0
	module load imkl
	make clean
	nvcc deviceQuery.cu -o deviceQuery
	sbatch exec.slurm
	make gemm && ./gemm
	sbatch exec.slurm
	cat slurm.out
	echo You Reached the Peak of Toubkal, Congratulations! 
