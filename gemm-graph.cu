#include <iostream>
#include <iomanip>
#include <fstream>
#include "mkl.h"
#include "cublas_v2.h"
#include "utils.h"

#define SINGLE_PRECISION //Comment out to use double precision arithmetic
#define DOUBLE_PRECISION

#ifdef SINGLE_PRECISION
	#define elem_t float
	#define blasGemm cblas_sgemm 
	#define cublasGemm cublasSgemm
	#define cublasGemmBatched cublasSgemmBatched
#elif defined(DOUBLE_PRECISION)
	#define elem_t double
	#define blasGemm cblas_dgemm 
	#define cublasGemm cublasDgemm
	#define cublasGemmBatched cublasDgemmBatched
#endif

#ifndef GEMM_M
#define GEMM_M 256
#endif
#ifndef GEMM_N
#define GEMM_N 256
#endif
#ifndef GEMM_K
#define GEMM_K 256
#endif

#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif

#ifndef NB_STREAMS
#define NB_STREAMS 16
#endif

#ifndef WARMUPS
#define WARMUPS 1
#endif
#ifndef ITERS
#define ITERS 10
#endif




void tileGemm(cublasHandle_t handle, int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC, int tileM, int tileN)
{
	for (int i=0; i<M/tileM; i++)
	{
		for (int j=0; j<N/tileN; j++)
		{
			elem_t *tileA = A + i*tileM;
			elem_t *tileB = B + j*tileN*K;
			elem_t *tileC = C + i*tileM + j*tileN*M;
			cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tileM, tileN, K, &alpha, tileA, M, tileB, K, &beta, tileC, M);
		}
	}
}

void tileGemmStreams(cublasHandle_t handle, int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC, int tileM, int tileN, int nb_streams, cudaStream_t *streams)
{
	for (int i=0; i<M/tileM; i++)
	{
		for (int j=0; j<N/tileN; j++)
		{
			elem_t *tileA = A + i*tileM;
			elem_t *tileB = B + j*tileN*K;
			elem_t *tileC = C + i*tileM + j*tileN*M;
			int tileId = i*(N/tileN)+j;
			cublasSetStream(handle, streams[tileId%nb_streams]);
			cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tileM, tileN, K, &alpha, tileA, M, tileB, K, &beta, tileC, M);
		}
	}
	cublasSetStream(handle, NULL);
}



void tileGemmBatch(cublasHandle_t handle, int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC, int tileM, int tileN)
{
	int batch_count = (M/tileM) * (N/tileN);
	elem_t **h_ptrs, **d_ptrs;
	h_ptrs = new elem_t*[batch_count*3];
	cudaMalloc( (void***)&d_ptrs, sizeof(elem_t**)*batch_count*3);
	elem_t **h_A_ptrs = h_ptrs;
	elem_t **h_B_ptrs = h_ptrs+batch_count;
	elem_t **h_C_ptrs = h_ptrs+2*batch_count;
	elem_t **d_A_ptrs = d_ptrs;
	elem_t **d_B_ptrs = d_ptrs+batch_count;
	elem_t **d_C_ptrs = d_ptrs+2*batch_count;
	for (int i=0; i<M/tileM; i++)
	{
		for (int j=0; j<N/tileN; j++)
		{
			int tileId = i*(N/tileN) + j;
			h_A_ptrs[tileId] = A + i*tileM;
			h_B_ptrs[tileId] = B + j*tileN*K;
			h_C_ptrs[tileId] = C + i*tileM + j*tileN*M;
		}
	}
	cudaMemcpy( d_ptrs, h_ptrs, sizeof(elem_t**)*batch_count*3, cudaMemcpyHostToDevice );
	cublasGemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, tileM, tileN, K, &alpha, d_A_ptrs, M, d_B_ptrs, K, &beta, d_C_ptrs, M, batch_count);
	cudaFree((void**)d_ptrs);
}



int main(int argc, char **argv)
{

	
	for (int var = 1000; var <= 3000; var += 500) {
		std::cout << "Current value of var: " << var << std::endl;

		
		cublasHandle_t handle;
		cublasCreate(&handle);

		cudaStream_t *streams;
		createStreams(NB_STREAMS, &streams);

		float *times = new float[2*ITERS];
		float *timesCPU = times;
		float *timesGPU = times + ITERS;

		elem_t *A, *B, *C, *Cgpu;
		elem_t *d_A, *d_B, *d_C;
		int M = var;
		int N = var;
		int K = var;

		//TASK 1 (Allocate and init A,B,C)
		allocateMatrixCPU(M,K,&A);
		allocateMatrixCPU(K,N,&B);
		allocateMatrixCPU(M,N,&C);

		initMatrixRandomCPU<elem_t>(M,K,A);
		initMatrixRandomCPU<elem_t>(K,N,B);
		initMatrixCPU<elem_t>(M,N,C,0.0);

		//TASK 2.1 (Allocate and init d_A, d_B, d_C)
		allocateMatrixGPU(M,K,&d_A);
		allocateMatrixGPU(K,N,&d_B);
		allocateMatrixGPU(M,N,&d_C);

		cudaMemcpy(d_A, A, sizeof(elem_t)*M*K, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, sizeof(elem_t)*N*K, cudaMemcpyHostToDevice);
		cudaMemcpy(d_C, C, sizeof(elem_t)*M*N, cudaMemcpyHostToDevice);

		elem_t alpha = 1.0;
		elem_t beta = 0.0;

		//CPU
		struct timespec cpu_start, cpu_end;
		for (int i=0; i<ITERS; i++)
		{
			clock_gettime(CLOCK_MONOTONIC, &cpu_start);
			blasGemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);
			clock_gettime(CLOCK_MONOTONIC, &cpu_end);
			timesCPU[i] = computeCPUTime(&cpu_start, &cpu_end);
		}

			

		//GPU
		for (int i=0; i<WARMUPS; i++)
		{
			cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M,N,K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
			//tileGemm(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M, TILE_M, TILE_N);
			//tileGemmStreams(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M, TILE_M, TILE_N, NB_STREAMS, streams);
			//tileGemmBatch(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M, TILE_M, TILE_N);
			cudaDeviceSynchronize();
		}

		cudaEvent_t gpu_start, gpu_end;
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_end);
		for (int i=0; i<ITERS; i++)
		{
			cudaEventRecord(gpu_start);
			cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M,N,K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
			//tileGemm(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M, TILE_M, TILE_N);
			//tileGemmStreams(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M, TILE_M, TILE_N, NB_STREAMS, streams);
			//tileGemmBatch(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M, TILE_M, TILE_N);
			cudaEventRecord(gpu_end);
			cudaDeviceSynchronize();
			cudaEventElapsedTime(&(timesGPU[i]), gpu_start, gpu_end);
		}
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_end);

		//TASK 1 (Compute and print average execution time/performance on CPU)
		float flops = 2*(float)M*(float)N*(float)K;

		float avg_cpu=0.0;

		float cpu_power = 50.0f; // Example average power in Watts for CPU

		for (int i=0; i<ITERS; i++)
			avg_cpu += timesCPU[i];
		avg_cpu = avg_cpu / (float)ITERS;
		float cpu_energy = cpu_power * (avg_cpu / 1.0e3); 

		std::cout << "==== CPU ====\n";
		std::cout << "Execution time: " << avg_cpu << " ms.\n";
		std::cout << "Performance: " << (flops/1.0e9)/(avg_cpu/1.0e3) << " GFLOP/s.\n";
		std::cout << "Energy: " << cpu_energy << " J.\n";
		std::cout << "Power Efficiency: " << (flops/1.0e9)/(cpu_energy) << " GFLOP/J.\n";
		

		//TASK 2.2 (Compute and print average execution time/performance on GPU)
		float avg_gpu=0.0;

		float gpu_power = 150.0f; // Example average power in Watts for GPU

		for (int i=0; i<ITERS; i++)
			avg_gpu += timesGPU[i];
		avg_gpu = avg_gpu / (float)ITERS;
		float gpu_energy = gpu_power * (avg_gpu / 1.0e3);

		std::cout << "==== GPU ====\n";
		std::cout << "Execution time: " << avg_gpu << " ms.\n";
		std::cout << "Performance: " << (flops/1.0e9)/(avg_gpu/1.0e3) << " GFLOP/s.\n";
		std::cout << "Energy: " << gpu_energy << " J.\n";
		std::cout << "Power Efficiency: " << (flops/1.0e9)/(gpu_energy) << " GFLOP/J.\n";

		//TASK 2.2 (Compute and print speedup)
		std::cout << "==== SPEEDUP ====\n";
		std::cout << "Speedup: " << avg_cpu/avg_gpu << "x.\n";


		// Open files in write mode to overwrite existing data if they exist, or create new files if they do not exist.
		std::ofstream cpu_outfile("cpu_output.csv", std::ios::out | std::ios::trunc);
		std::ofstream gpu_outfile("gpu_output.csv", std::ios::out | std::ios::trunc);

		// Write the header.
		cpu_outfile << "Dim,CPU Execution Time (ms),CPU Performance (GFLOP/s),CPU Energy (J),Power Efficiency (GFLOP/J)\n";
		gpu_outfile << "Dim,GPU Execution Time (ms),GPU Performance (GFLOP/s),GPU Energy (J),Power Efficiency (GFLOP/J),Speedup (x)\n";

		// Write the results to the files.
		cpu_outfile << var << "," << avg_cpu << "," << (flops/1.0e9)/(avg_cpu/1.0e3) << "," << cpu_energy << "," << (flops/1.0e9)/(cpu_energy) << "\n";
		gpu_outfile << var << "," << avg_gpu << "," << (flops/1.0e9)/(avg_gpu/1.0e3) << "," << gpu_energy << "," << (flops/1.0e9)/(gpu_energy) << "," << avg_cpu/avg_gpu << "\n";

		// Close the files.
		cpu_outfile.close();
		gpu_outfile.close();





		//TASK 2.2 (Compare CPU and GPU output)
		allocateMatrixCPU(M,N,&Cgpu);
		cudaMemcpy(Cgpu, d_C, sizeof(elem_t)*M*N, cudaMemcpyDeviceToHost);
		std::cout << std::setprecision(10);
		compareMatrices(M,N,C,Cgpu);
		freeMatrixCPU(M,N,Cgpu);

		//TASK 2.1 (Free d_A, d_B, d_C)
		freeMatrixGPU(M,K,d_A);
		freeMatrixGPU(K,N,d_B);
		freeMatrixGPU(M,N,d_C);

		//TASK 1 (Free A,B,C)
		freeMatrixCPU(M,K,A);
		freeMatrixCPU(K,N,B);
		freeMatrixCPU(M,N,C);

		destroyStreams(NB_STREAMS, streams);
		cublasDestroy(handle);

		delete[] times;

	}

	
	std::cout << "CSV files for CPU and GPU performance metrics have been created successfully." << std::endl;
	

	

}







