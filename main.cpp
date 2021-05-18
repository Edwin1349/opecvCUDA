#include "main.h"

int main(void) {

    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();

    cv::Mat imgOriginalScene;
    imgOriginalScene = cv::imread("Cars1.png");
    std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);

    vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);

    cv::imshow("imgOriginalScene", imgOriginalScene);

    if (vectorOfPossiblePlates.empty()) {
        std::cout << std::endl << "no license plates were detected" << std::endl;
    }
    else {
        std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);
        PossiblePlate licPlate = vectorOfPossiblePlates.front();

        cv::imshow("imgPlate", licPlate.imgPlate);
        cv::imshow("imgThresh", licPlate.imgThresh);

        if (licPlate.strChars.length() == 0) {
            std::cout << std::endl << "no characters were detected" << std::endl << std::endl;
            return(0);
        }

        std::cout << std::endl << "license plate read from image = " << licPlate.strChars << std::endl;
        std::cout << std::endl << "-----------------------------------------" << std::endl;

        cv::imshow("imgOriginalScene", imgOriginalScene);
    }

    cv::waitKey(0);
    return(0);
}
