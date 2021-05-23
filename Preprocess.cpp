#include "Preprocess.h"

double max_time;

void preprocess(std::vector<cv::Mat>& imgOriginal, std::vector<cv::Mat>& imgGrayscale, std::vector<cv::Mat>& imgThresh) {
    std::shared_ptr<std::vector<cv::cuda::Stream>> streamsArray = std::make_shared<std::vector<cv::cuda::Stream>>();
    for (int i = 0; i < imgOriginal.size(); i++) {
        cv::cuda::Stream stream;
        streamsArray->push_back(stream);
    }

    std::vector<cv::cuda::GpuMat> d_imgOriginal(imgOriginal.size());
    
    std::shared_ptr<std::vector<cv::cuda::HostMem >> srcMemArray = std::make_shared<std::vector<cv::cuda::HostMem >>();
    std::shared_ptr<std::vector<cv::cuda::HostMem >> dstMemArrayGr = std::make_shared<std::vector<cv::cuda::HostMem >>();
    std::shared_ptr<std::vector<cv::cuda::HostMem >> dstMemArrayTr = std::make_shared<std::vector<cv::cuda::HostMem >>();
    std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuSrcArray = std::make_shared<std::vector<cv::cuda::GpuMat>>();

    std::shared_ptr<std::vector< cv::Mat >> OutimgGrayscale = std::make_shared<std::vector<cv::Mat>>();
    std::shared_ptr<std::vector< cv::Mat >> OutimgThresh = std::make_shared<std::vector<cv::Mat>>();

    for (int i = 0; i < imgOriginal.size(); i++) {
        cv::cuda::GpuMat srcMat;
        cv::cuda::GpuMat dstMat;

        cv::Mat outMat;

        cv::cuda::HostMem srcHostMem = cv::cuda::HostMem(imgOriginal[i], cv::cuda::HostMem::PAGE_LOCKED);
        cv::cuda::HostMem srcDstMem = cv::cuda::HostMem(outMat, cv::cuda::HostMem::PAGE_LOCKED);

        srcMemArray->push_back(srcHostMem);
        dstMemArrayGr->push_back(srcDstMem);
        dstMemArrayTr->push_back(srcDstMem);

        gpuSrcArray->push_back(srcMat);

        OutimgGrayscale->push_back(outMat);
        OutimgThresh->push_back(outMat);
    }

    std::shared_ptr<std::vector<cv::cuda::GpuMat>> d_imgGrayscale = std::make_shared<std::vector<cv::cuda::GpuMat>>();
    std::shared_ptr<std::vector<cv::cuda::GpuMat>> imgMaxContrastGrayscale = std::make_shared<std::vector<cv::cuda::GpuMat>>();

    std::shared_ptr<std::vector<cv::cuda::GpuMat>> imgBlurred = std::make_shared<std::vector<cv::cuda::GpuMat>>();
    std::shared_ptr<std::vector<cv::cuda::GpuMat>> d_imgThresh = std::make_shared<std::vector<cv::cuda::GpuMat>>();

    for (int i = 0; i < gpuSrcArray->size(); i++) {
        cv::cuda::GpuMat dstImgBlurred, dstD_imgThresh;
        imgBlurred->push_back(dstImgBlurred);
        d_imgThresh->push_back(dstD_imgThresh);
    }

    d_imgGrayscale = extractValue(srcMemArray, gpuSrcArray, streamsArray);

    imgMaxContrastGrayscale = maximizeContrast(d_imgGrayscale, streamsArray);

    for (int i = 0; i < imgMaxContrastGrayscale->size(); i++) {
        cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter((*imgMaxContrastGrayscale)[i].type(), (*imgBlurred)[i].type(), GAUSSIAN_SMOOTH_FILTER_SIZE, 0);
        filter->apply((*imgMaxContrastGrayscale)[i], (*imgBlurred)[i], (*streamsArray)[i % streamsArray->size()]);
        cv::cuda::threshold((*imgBlurred)[i], (*d_imgThresh)[i], 100, 255.0, cv::THRESH_BINARY_INV, (*streamsArray)[i % streamsArray->size()]);
    }

    for (int i = 0; i < d_imgGrayscale->size(); i++) {
        (*imgBlurred)[i].download((*dstMemArrayGr)[i], (*streamsArray)[i % streamsArray->size()]);
        (*OutimgGrayscale)[i] = (*dstMemArrayGr)[i].createMatHeader();

        (*d_imgThresh)[i].download((*dstMemArrayTr)[i], (*streamsArray)[i % streamsArray->size()]);
        (*OutimgThresh)[i] = (*dstMemArrayTr)[i].createMatHeader();
    }

    for (int i = 0; i < (*streamsArray).size(); i++) {
        (*streamsArray)[i].waitForCompletion();
    }

    for (int i = 0; i < OutimgGrayscale->size(); i++) {
        cv::Mat a, b;
        imgGrayscale.push_back(a);
        ((*OutimgGrayscale)[i].copyTo(imgGrayscale[i]));
        imgThresh.push_back(b);
        ((*OutimgThresh)[i]).copyTo(imgThresh[i]);
    }
}

std::shared_ptr<std::vector<cv::cuda::GpuMat>> extractValue(std::shared_ptr<std::vector< cv::cuda::HostMem >> srcMemArray,
                                                            std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuSrcArray,
                                                            std::shared_ptr<std::vector< cv::cuda::Stream >> streamsArray) {

    std::shared_ptr<std::vector<cv::cuda::GpuMat>> imgHSV = std::make_shared<std::vector<cv::cuda::GpuMat>>();
    std::shared_ptr<std::vector<cv::cuda::GpuMat>> imgValue = std::make_shared<std::vector<cv::cuda::GpuMat>>();
    std::shared_ptr<std::vector<std::vector<cv::cuda::GpuMat>>> vectorOfHSVImages = std::make_shared<std::vector<std::vector<cv::cuda::GpuMat>>>();

    for (int i = 0; i < srcMemArray->size(); i++) {
        cv::cuda::GpuMat dstImgHSV, dstImgValue;
        std::vector<cv::cuda::GpuMat> dstVectorOfHSVImages;

        imgHSV->push_back(dstImgHSV);
        imgValue->push_back(dstImgValue);
        vectorOfHSVImages->push_back(dstVectorOfHSVImages);
    }

    for (int i = 0; i < srcMemArray->size(); i++) {
        (*gpuSrcArray)[i].upload((*srcMemArray)[i], (*streamsArray)[i % streamsArray->size()]);

        cv::cuda::cvtColor((*gpuSrcArray)[i], (*imgHSV)[i], cv::COLOR_BGR2HSV, 0, (*streamsArray)[i % streamsArray->size()]);   
        cv::cuda::split((*imgHSV)[i], (*vectorOfHSVImages)[i], (*streamsArray)[i % streamsArray->size()]);
        (*imgValue)[i] = (*vectorOfHSVImages)[i][2];
    }

    for (int i = 0; i < (*streamsArray).size(); i++) {
        (*streamsArray)[i].waitForCompletion();
    }
    return(imgValue);
}

std::shared_ptr<std::vector<cv::cuda::GpuMat>> maximizeContrast(std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuSrcArray,
                                                                std::shared_ptr<std::vector< cv::cuda::Stream >> streamsArray){

    std::shared_ptr<std::vector<cv::cuda::GpuMat>> imgTopHat = std::make_shared<std::vector<cv::cuda::GpuMat>>();
    std::shared_ptr<std::vector<cv::cuda::GpuMat>> imgBlackHat = std::make_shared<std::vector<cv::cuda::GpuMat>>();
    std::shared_ptr<std::vector<cv::cuda::GpuMat>> imgGrayscalePlusTopHat = std::make_shared<std::vector<cv::cuda::GpuMat>>();
    std::shared_ptr<std::vector<cv::cuda::GpuMat>> imgGrayscalePlusTopHatMinusBlackHat = std::make_shared<std::vector<cv::cuda::GpuMat>>();

    for (int i = 0; i < gpuSrcArray->size(); i++) {
        cv::cuda::GpuMat dstMat;

        imgTopHat->push_back(dstMat);
        imgBlackHat->push_back(dstMat);
        imgGrayscalePlusTopHat->push_back(dstMat);
        imgGrayscalePlusTopHatMinusBlackHat->push_back(dstMat);
    }

    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Ptr<cv::cuda::Filter>morph = cv::cuda::createMorphologyFilter(cv::MORPH_TOPHAT, (*imgTopHat)[0].type(), structuringElement);
    cv::Ptr < cv::cuda::Filter>morph2 = cv::cuda::createMorphologyFilter(cv::MORPH_BLACKHAT, (*imgBlackHat)[0].type(), structuringElement);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < gpuSrcArray->size(); i++) {

        morph->apply((*gpuSrcArray)[i], (*imgTopHat)[i], (*streamsArray)[i % streamsArray->size()]);
        morph2->apply((*gpuSrcArray)[i], (*imgBlackHat)[i], (*streamsArray)[i % streamsArray->size()]);

        cv::cuda::add((*gpuSrcArray)[i], (*imgTopHat)[i], (*imgGrayscalePlusTopHat)[i], cv::noArray(), -1, (*streamsArray)[i % streamsArray->size()]);
        cv::cuda::subtract((*imgGrayscalePlusTopHat)[i], (*imgBlackHat)[i], (*imgGrayscalePlusTopHatMinusBlackHat)[i], cv::noArray(), -1, (*streamsArray)[i % streamsArray->size()]);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    max_time += std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
    std::cout << "Max time: " << max_time << std::endl;
    //std::cout << cv::getBuildInformation();
    for (int i = 0; i < (*streamsArray).size(); i++) {
        (*streamsArray)[i].waitForCompletion();
    }

    return(imgGrayscalePlusTopHatMinusBlackHat);
}