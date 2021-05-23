#ifndef PREPROCESS_H
#define PREPROCESS_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

#include <chrono>
#include <iostream>

const cv::Size GAUSSIAN_SMOOTH_FILTER_SIZE = cv::Size(5, 5);
const int ADAPTIVE_THRESH_BLOCK_SIZE = 19;
const int ADAPTIVE_THRESH_WEIGHT = 9;


void preprocess(std::vector<cv::Mat>& imgOriginal, std::vector<cv::Mat>& imgGrayscale, std::vector<cv::Mat>& imgThresh);

std::shared_ptr<std::vector<cv::cuda::GpuMat>> extractValue(std::shared_ptr<std::vector< cv::cuda::HostMem >> srcMemArray,
                                                            std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuSrcArray,
                                                            std::shared_ptr<std::vector< cv::cuda::Stream >> streamsArray);

std::shared_ptr<std::vector<cv::cuda::GpuMat>>  maximizeContrast(std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuSrcArray,
                                                                 std::shared_ptr<std::vector< cv::cuda::Stream >> streamsArray);

#endif
