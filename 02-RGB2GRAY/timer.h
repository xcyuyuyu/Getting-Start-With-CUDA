#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>
#include <sys/time.h>

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};



struct CpuTimer
{
  struct timeval start;
  struct timeval stop;

  CpuTimer()
  {
    gettimeofday(&start,NULL);
    gettimeofday(&stop,NULL);
  }

  ~CpuTimer(){}

  void Start()
  {
    gettimeofday(&start,NULL);
  }

  void Stop()
  {
    gettimeofday(&stop,NULL);
  }

  float Elapsed()
  {
    float elapsed;
    elapsed = (double)(stop.tv_sec - start.tv_sec) * 1000 + (double)(stop.tv_usec - start.tv_usec)/1000.0;
    return elapsed;
  }
};

#endif  /* GPU_TIMER_H__ */
