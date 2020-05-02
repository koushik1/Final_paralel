/**
* Group Info:
* rwsnyde2 Richard W Snyder
* kshanka2 Koushik Shankar
*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cooperative_groups.h>


#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)


extern int tpdt(double *t, double dt, double end_time);
#define TSCALE 1.0
#define VSQR 0.1
namespace cg = cooperative_groups;


/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

__device__ void evolve13(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, double end_time)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx / n;
  int j = idx % n;
  cg::thread_block block = cg::this_thread_block();


  if( i <= 1 || i >= n-2 || j <= 1 || j >= n - 2 )
  {
    un[idx] = 0.;
  }
  else
  {
    un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *
            ((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx-n] + // west east north south
            0.25 * (uc[idx - n - 1] + uc[idx - n + 1] + uc[idx + n - 1] + uc[idx + n + 1]) + // northwest northeast southwest southeast
            0.125 * (uc[idx - 2] + uc[idx + 2] + uc[idx - (2*n)] + uc[idx + (2*n)]) - // westwest easteast northnorth southsouth
            5.5 * uc[idx])/(h * h) + (-expf(-TSCALE * t) * pebbles[idx]));
  }
  block.sync();

  if (t + dt < end_time)
  {
      t = t + dt;
      evolve13(uo, un, uc, pebbles, n, h, dt, t,end_time);
  }

}


__global__ void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, double end_time)
{
    evolve13(un, uc, uo, pebbles, n, h, dt, t,end_time);
}


void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;
        
  /* HW2: Define your local variables here */
  double t, dt;
  int num_blocks = (n/nthreads)*(n/nthreads);
  int threads_per_block = nthreads * nthreads;
  double *u_d, *u0_d, *u1_d,*pebbles_d;



        /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

  /* HW2: Add CUDA kernel call preperation code here */
  cudaMalloc((void **) &u_d, sizeof(double) * n * n); 
  cudaMalloc((void **) &u0_d, sizeof(double) * n * n); 
  cudaMalloc((void **) &u1_d, sizeof(double) * n * n);
  cudaMalloc((void **) &pebbles_d, sizeof(double) * n * n); 

  cudaMemcpy(u0_d, u0, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(u1_d, u1, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(pebbles_d, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

  /* HW2: Add main lake simulation loop here */
  t = 0.;
  dt = h / 2.;

    evolve<<< num_blocks,threads_per_block >>>(u_d, u1_d, u0_d, pebbles_d, n, h, dt, t,end_time);

	
        /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

  /* HW2: Add post CUDA kernel call processing and cleanup here */
  cudaMemcpy(u, u_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
  cudaFree(u_d);
  cudaFree(u0_d);
  cudaFree(u1_d);
  cudaFree(pebbles_d);

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}