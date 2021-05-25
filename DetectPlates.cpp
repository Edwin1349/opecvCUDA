#include "DetectPlates.h"
std::vector<std::vector<PossiblePlate>> detectPlatesInScene(std::vector<cv::Mat>& imgOriginalScene) {
    std::vector<std::vector<PossiblePlate>> vectorOfPossiblePlates(imgOriginalScene.size());			// this will be the return value

    std::vector<cv::Mat> imgGrayscaleScene;
    std::vector<cv::Mat> imgThreshScene;
    //cv::Mat imgContours(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);

    cv::RNG rng;

    cv::destroyAllWindows();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    preprocess(imgOriginalScene, imgGrayscaleScene, imgThreshScene);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Plates preprocess time: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;

    for (int i = 0; i < imgOriginalScene.size(); i++) {
        std::vector<PossibleChar> vectorOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene[i]);

        std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInScene = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInScene);

        for (auto& vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) {                     // for each group of matching chars
            PossiblePlate possiblePlate = extractPlate(imgOriginalScene[i], vectorOfMatchingChars);        // attempt to extract plate
            if (possiblePlate.imgPlate.empty() == false) {                                              // if plate was found
                vectorOfPossiblePlates[i].push_back(possiblePlate);                                        // add to vector of possible plates
            }
        }
    }

    return vectorOfPossiblePlates;
}

std::vector<PossibleChar> findPossibleCharsInScene(cv::Mat& imgThresh) {
    std::vector<PossibleChar> vectorOfPossibleChars;            // this will be the return value

    cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);
    int intCountOfPossibleChars = 0;

    cv::Mat imgThreshCopy;
    imgThresh.copyTo(imgThreshCopy);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(imgThreshCopy, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);        // find all contours

    for (unsigned int i = 0; i < contours.size(); i++) {                // for each contour
        PossibleChar possibleChar(contours[i]);
        if (checkIfPossibleChar(possibleChar)) {                // if contour is a possible char, note this does not compare to other chars (yet) . . .
            intCountOfPossibleChars++;                          // increment count of possible chars
            vectorOfPossibleChars.push_back(possibleChar);      // and add to vector of possible chars
        }
    }

    return(vectorOfPossibleChars);
}

PossiblePlate extractPlate(cv::Mat& imgOriginal, std::vector<PossibleChar>& vectorOfMatchingChars) {
    PossiblePlate possiblePlate;            // this will be the return value

                                            // sort chars from left to right based on x position
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    // calculate the center point of the plate
    double dblPlateCenterX = (double)(vectorOfMatchingChars[0].intCenterX + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterX) / 2.0;
    double dblPlateCenterY = (double)(vectorOfMatchingChars[0].intCenterY + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY) / 2.0;
    cv::Point2d p2dPlateCenter(dblPlateCenterX, dblPlateCenterY);

    // calculate plate width and height
    int intPlateWidth = (int)((vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.x + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.width - vectorOfMatchingChars[0].boundingRect.x) * PLATE_WIDTH_PADDING_FACTOR);

    double intTotalOfCharHeights = 0;

    for (auto& matchingChar : vectorOfMatchingChars) {
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.boundingRect.height;
    }

    double dblAverageCharHeight = (double)intTotalOfCharHeights / vectorOfMatchingChars.size();

    int intPlateHeight = (int)(dblAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR);

    // calculate correction angle of plate region
    double dblOpposite = vectorOfMatchingChars[int(vectorOfMatchingChars.size()) - 1].intCenterY - vectorOfMatchingChars[0].intCenterY;
    double dblHypotenuse = distanceBetweenChars(vectorOfMatchingChars[0], vectorOfMatchingChars[vectorOfMatchingChars.size() - 1]);
    double dblCorrectionAngleInRad = asin(dblOpposite / dblHypotenuse);
    double dblCorrectionAngleInDeg = dblCorrectionAngleInRad * (180.0 / CV_PI);

    // assign rotated rect member variable of possible plate
    possiblePlate.rrLocationOfPlateInScene = cv::RotatedRect(p2dPlateCenter, cv::Size2f((float)intPlateWidth, (float)intPlateHeight), (float)dblCorrectionAngleInDeg);

    cv::Mat rotationMatrix;             // final steps are to perform the actual rotation
    cv::Mat imgRotated;
    cv::Mat imgCropped;

    rotationMatrix = cv::getRotationMatrix2D(p2dPlateCenter, dblCorrectionAngleInDeg, 1.0);         // get the rotation matrix for our calculated correction angle

    cv::warpAffine(imgOriginal, imgRotated, rotationMatrix, imgOriginal.size());            // rotate the entire image

                                                                                            // crop out the actual plate portion of the rotated image
    cv::getRectSubPix(imgRotated, possiblePlate.rrLocationOfPlateInScene.size, possiblePlate.rrLocationOfPlateInScene.center, imgCropped);
    possiblePlate.imgPlate = imgCropped;            // copy the cropped plate image into the applicable member variable of the possible plate
    return(possiblePlate);
}