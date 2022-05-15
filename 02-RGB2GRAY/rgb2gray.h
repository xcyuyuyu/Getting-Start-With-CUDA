#ifndef RGB2GRAY_H
#define RGB2GRAY_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename, size_t *numRows, size_t *numCols);

void postProcess(const std::string& output_file, unsigned char* data_ptr, size_t numRows, size_t numCols);

void cleanup(uchar4 *d_rgbaImage__, unsigned char *d_greyImage__);

void generateReferenceImage(std::string input_filename, std::string output_filename);

#endif