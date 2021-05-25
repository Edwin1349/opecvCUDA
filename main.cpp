#include "main.h"
#include <string>

void extract_frames(const std::string& videoFilePath, std::vector<cv::Mat>& frames) {
    try {
        //open the video file
        cv::VideoCapture cap(videoFilePath); // open the video file

        //cap.get(CV_CAP_PROP_FRAME_COUNT) contains the number of frames in the video;
        for (int frameNum = 0; frameNum < cap.get(cv::CAP_PROP_FRAME_COUNT); frameNum++)
        {
            cv::Mat frame;
            cap >> frame; // get the next frame from video
            frames.push_back(frame);
        }
    }
    catch (cv::Exception& e) {
        std::cerr << e.msg << std::endl;
        exit(1);
    }

}

int main(void) {
    cv::cuda::printShortCudaDeviceInfo(0);
    cv::cuda::setDevice(0);
    std::cout<<cv::cuda::getDevice();
    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();
    std::vector<cv::Mat> imgsOriginalScene;

    extract_frames("Cars\\car1.avi", imgsOriginalScene);

    //imgsOriginalScene.clear();
    //imgsOriginalScene.push_back(cv::imread("Cars\\Cars1.png"));

    std::vector<std::vector<PossiblePlate>> vectorOfPossiblePlates = detectPlatesInScene(imgsOriginalScene);
    vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);

    for (int img = 0; img < imgsOriginalScene.size(); img++) {

        if (vectorOfPossiblePlates[img].empty()) {
            //std::cout << std::endl << "no license plates were detected" << std::endl;
            continue;
        }

        else {
            std::sort(vectorOfPossiblePlates[img].begin(), vectorOfPossiblePlates[img].end(), PossiblePlate::sortDescendingByNumberOfChars);

            PossiblePlate a = vectorOfPossiblePlates[img].front();

            cv::imshow("imgPlate", a.imgPlate);
            cv::imshow("imgThresh", a.imgThresh);
            cv::imshow("imgOriginalScene", imgsOriginalScene[img]);
            cv::waitKey(0);
            if (a.strChars.length() == 0) {
                //std::cout << std::endl << "no characters were detected" << std::endl << std::endl;
                continue;
            }

            std::cout << std::endl << "license plate read from image = " << a.strChars << std::endl;
            std::cout << std::endl << "-----------------------------------------" << std::endl;
        }
    }
    cv::waitKey(0);
    return(0);
}
