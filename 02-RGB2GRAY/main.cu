//Udacity HW1 Solution
// hello
#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include "reference_calc.h"
#include "compare.h"
#include "rgb2gray.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  uchar4 rgba = rgbaImage[x * numCols + y];
  float channelSum = (299 * rgba.x + 587 * rgba.y + 114 * rgba.z)/1000.f;
  greyImage[x * numCols + y] = channelSum;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, 
                            uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage,
                            size_t numRows, size_t numCols)
{
  printf("numRows: %d, numCols: %d\n", numRows, numCols);
  const dim3 blockSize(32, 32, 1);  //TODO
  const dim3 gridSize((numRows+32)/32, (numCols+32)/32, 1);  //TODO
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

int main(int argc, char **argv) {
  uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage, *h_greyImage_ref;

  std::string input_file;
  std::string output_file;
  std::string reference_file;
  double perPixelError = 0.0;
  double globalError   = 0.0;
  bool useEpsCheck = false;
  switch (argc)
  {
	case 2:
	  input_file = std::string(argv[1]);
	  output_file = "output.png";
	  reference_file = "reference.png";
	  break;
	case 3:
	  input_file  = std::string(argv[1]);
      output_file = std::string(argv[2]);
	  reference_file = "reference.png";
	  break;
	case 4:
	  input_file  = std::string(argv[1]);
      output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  break;
	case 6:
	  useEpsCheck=true;
	  input_file  = std::string(argv[1]);
	  output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  perPixelError = atof(argv[4]);
      globalError   = atof(argv[5]);
	  break;
	default:
      std::cerr << "Usage: ./HW1 input_file [output_filename] [reference_filename] [perPixelError] [globalError]" << std::endl;
      exit(1);
  }
  //load the image and give us our input and output pointers
  size_t numRows;
  size_t numCols;
  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file, &numRows, &numCols);

  GpuTimer gpu_timer;
  gpu_timer.Start();
  //call the students' code
  your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows, numCols);
  gpu_timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("Your code ran in: %f ms on GPU Device.\n", gpu_timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }
  
  size_t numPixels = numRows*numCols;
  checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

  //check results and output the grey image
  postProcess(output_file, h_greyImage, numRows, numCols);
  
  cv::Mat imageGreyRef(numRows, numCols, CV_8UC1);
  h_greyImage_ref = imageGreyRef.ptr<unsigned char>(0);
  CpuTimer cpu_timer;
  cpu_timer.Start();
  referenceCalculation(h_rgbaImage, h_greyImage_ref, numRows, numCols);
  cpu_timer.Stop();
  printf("Your code ran in: %f ms on CPU Device.\n", cpu_timer.Elapsed());

  postProcess(reference_file, h_greyImage_ref, numRows, numCols);

  compareImages(output_file, reference_file, useEpsCheck, perPixelError, globalError);

  cleanup(d_rgbaImage, d_greyImage);

  return 0;
}
