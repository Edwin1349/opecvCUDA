//#include <opencv2/highgui.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudafilters.hpp>
//#include <opencv2/cudaarithm.hpp>
//
//const cv::Size GAUSSIAN_SMOOTH_FILTER_SIZE = cv::Size(5, 5);
//cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
//
//cv::cuda::GpuMat extractValue(cv::cuda::GpuMat& imgOriginal) {
//	cv::cuda::GpuMat imgHSV;
//	std::vector<cv::cuda::GpuMat> vectorOfHSVImages;
//	cv::cuda::GpuMat imgValue;
//
//	cv::cuda::cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV);
//
//	cv::cuda::split(imgHSV, vectorOfHSVImages);
//
//	imgValue = vectorOfHSVImages[2];
//
//	return(imgValue);
//}
//
//cv::cuda::GpuMat maximizeContrast(cv::cuda::GpuMat& imgGrayscale) {
//	cv::cuda::GpuMat imgTopHat, imgBlackHat, imgGrayscalePlusTopHat, imgGrayscalePlusTopHatMinusBlackHat;
//
//	cv::Ptr<cv::cuda::Filter>morph = cv::cuda::createMorphologyFilter(cv::MORPH_TOPHAT, imgTopHat.type(), structuringElement);
//	morph->apply(imgGrayscale, imgTopHat);
//
//	morph = cv::cuda::createMorphologyFilter(cv::MORPH_BLACKHAT, imgBlackHat.type(), structuringElement);
//	morph->apply(imgGrayscale, imgBlackHat);
//
//	cv::cuda::add(imgGrayscale, imgTopHat, imgGrayscalePlusTopHat);
//	cv::cuda::subtract(imgGrayscalePlusTopHat, imgBlackHat, imgGrayscalePlusTopHatMinusBlackHat);
//
//	return(imgGrayscalePlusTopHatMinusBlackHat);
//}
//
//void preprocess(cv::Mat& imgOriginal, cv::Mat& imgGrayscale, cv::Mat& imgThresh) {
//	cv::Mat res;
//
//	cv::cuda::GpuMat d_imgThresh, d_imgGrayscale, d_imgOriginal;
//	d_imgThresh.upload(imgOriginal);
//	d_imgGrayscale.upload(imgOriginal);
//	d_imgOriginal.upload(imgThresh);
//
//	cv::cuda::GpuMat src, imgMaxContrastGrayscale, imgBlurred;
//	d_imgGrayscale = extractValue(d_imgOriginal);
//
//	imgMaxContrastGrayscale = maximizeContrast(d_imgGrayscale);
//
//	cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(imgMaxContrastGrayscale.type(), imgBlurred.type(), GAUSSIAN_SMOOTH_FILTER_SIZE, 0);
//	filter->apply(imgMaxContrastGrayscale, imgBlurred);
//
//	cv::cuda::threshold(imgBlurred, imgThresh, 100, 255.0, cv::THRESH_BINARY_INV);
//
//	d_imgThresh.download(res);
//	cv::imshow("result3", res);
//	cv::waitKey();
//}
//
//int main() {
//	cv::Mat imgThresh, imgGrayscale, imgOriginal;
//
//	cv::Mat imgOriginal = cv::imread("Cars1.png");
//	
//
//	preprocess(imgOriginal, imgGrayscale, imgThresh);
//	return 0;
//}