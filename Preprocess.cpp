#include "Preprocess.h"
void preprocess(cv::Mat& imgOriginal, cv::Mat& imgGrayscale, cv::Mat& imgThresh) {
    cv::Mat h_imgThresh, h_imgGrayscale, h_imgOriginal;

    cv::cuda::GpuMat d_imgThresh, d_imgGrayscale, d_imgOriginal;
    d_imgOriginal.upload(imgOriginal);


    cv::cuda::GpuMat src, imgMaxContrastGrayscale, imgBlurred;
    d_imgGrayscale = extractValue(d_imgOriginal);

    imgMaxContrastGrayscale = maximizeContrast(d_imgGrayscale);

    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(imgMaxContrastGrayscale.type(), imgBlurred.type(), GAUSSIAN_SMOOTH_FILTER_SIZE, 0);
    filter->apply(imgMaxContrastGrayscale, imgBlurred);

    cv::cuda::threshold(imgBlurred, d_imgThresh, 100, 255.0, cv::THRESH_BINARY_INV);

    d_imgGrayscale.download(imgGrayscale);
    d_imgThresh.download(imgThresh);

    cv::waitKey();
}

cv::cuda::GpuMat extractValue(cv::cuda::GpuMat& imgOriginal) {
    cv::cuda::GpuMat imgHSV;
    std::vector<cv::cuda::GpuMat> vectorOfHSVImages;
    cv::cuda::GpuMat imgValue;

    cv::cuda::cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV);

    cv::cuda::split(imgHSV, vectorOfHSVImages);

    imgValue = vectorOfHSVImages[2];

    return(imgValue);
}

cv::cuda::GpuMat maximizeContrast(cv::cuda::GpuMat& imgGrayscale) {
    cv::cuda::GpuMat imgTopHat, imgBlackHat, imgGrayscalePlusTopHat, imgGrayscalePlusTopHatMinusBlackHat;
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Ptr<cv::cuda::Filter>morph = cv::cuda::createMorphologyFilter(cv::MORPH_TOPHAT, imgTopHat.type(), structuringElement);
    morph->apply(imgGrayscale, imgTopHat);

    morph = cv::cuda::createMorphologyFilter(cv::MORPH_BLACKHAT, imgBlackHat.type(), structuringElement);
    morph->apply(imgGrayscale, imgBlackHat);

    cv::cuda::add(imgGrayscale, imgTopHat, imgGrayscalePlusTopHat);
    cv::cuda::subtract(imgGrayscalePlusTopHat, imgBlackHat, imgGrayscalePlusTopHatMinusBlackHat);

    return(imgGrayscalePlusTopHatMinusBlackHat);
}