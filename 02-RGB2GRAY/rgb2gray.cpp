#include "rgb2gray.h"
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/imgproc/types_c.h>

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename, size_t *numRows, size_t *numCols) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }
  cv::Mat imageRGBA;
  cv::Mat imageGrey;

  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

  //allocate memory for the output
  imageGrey.create(image.rows, image.cols, CV_8UC1);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  *numRows = imageRGBA.rows;
  *numCols = imageRGBA.cols;
  const size_t numPixels = (*numRows) * (*numCols);
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); //make sure no memory is left laying around

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

//  printf("Hi2\n");
//  for(int i=0; i<1; i++){
//    unsigned char r = (*inputImage)[1005].x;
//    unsigned char g = (*inputImage)[1005].y;
//    unsigned char b = (*inputImage)[1005].z;
//    float tmp = (.299f * (float)r + .587f * (float)g + .114f * (float)b);
//    printf("%u %u %u\t", r, g, b);
//    printf("%u \n", (unsigned char)tmp);
//  }

}

void postProcess(const std::string& output_file, unsigned char* data_ptr, size_t numRows, size_t numCols) {
  cv::Mat output(numRows, numCols, CV_8UC1, (void*)data_ptr);
  //output the image
  // const std::string filename = "../output.jpg";
//  cv::imwrite(filename, output);
  // cv::imwrite("../output.jpg", output);
 cv::imwrite(output_file, output);
}

void cleanup(uchar4 *d_rgbaImage__, unsigned char *d_greyImage__)
{
  //cleanup
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}

void generateReferenceImage(std::string input_filename, std::string output_filename)
{
  cv::Mat reference = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);

  cv::imwrite(output_filename, reference);

}
