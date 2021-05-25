#include "DetectChars.h"

cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();

bool loadKNNDataAndTrainKNN(void) {
    cv::Mat matClassificationInts;              // we will read the classification numbers into this variable as though it is a vector

    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        // open the classifications file

    if (fsClassifications.isOpened() == false) {                                                        // if the file was not opened successfully
        std::cout << "error, unable to open training classifications file, exiting program\n\n";        // show error message
        return(false);                                                                                  // and exit program
    }

    fsClassifications["classifications"] >> matClassificationInts;          // read classifications section into Mat classifications variable
    fsClassifications.release();                                            // close the classifications file

                                                                            // read in training images ////////////////////////////////////////////////////////////

    cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);              // open the training images file

    if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
        std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
        return(false);                                                                          // and exit program
    }

    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
    fsTrainingImages.release();                                                 // close the traning images file

                                                                                // train //////////////////////////////////////////////////////////////////////////////

                                                                                // finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
                                                                                // even though in reality they are multiple images / numbers
    kNearest->setDefaultK(1);

    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    return true;
}

std::vector<std::vector<PossiblePlate>> detectCharsInPlates(std::vector<std::vector<PossiblePlate>>& vectorOfPossiblePlates) {
    cv::Mat imgContours;
    std::vector<std::vector<cv::Point> > contours;
    cv::RNG rng;
    std::vector<std::vector<cv::Mat>> imgPlate(vectorOfPossiblePlates.size()), imgGrayscale(vectorOfPossiblePlates.size()), imgThresh(vectorOfPossiblePlates.size());

    for (int i = 0; i < vectorOfPossiblePlates.size(); i++) {
        if (vectorOfPossiblePlates[i].empty()) {               // if vector of possible plates is empty
            //return(vectorOfPossiblePlates);                 // return
            continue;
        }
        else {
            for (int j = 0; j < vectorOfPossiblePlates[i].size(); j++) {
                imgPlate[i].push_back(vectorOfPossiblePlates[i][j].imgPlate);
                imgGrayscale[i].push_back(vectorOfPossiblePlates[i][j].imgGrayscale);
                imgThresh[i].push_back(vectorOfPossiblePlates[i][j].imgThresh);
            }
        }
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < vectorOfPossiblePlates.size(); i++) {
    //for (auto& possiblePlate : vectorOfPossiblePlates) {            // for each possible plate, this is a big for loop that takes up most of the function
        if (vectorOfPossiblePlates[i].size() == 0) {
            continue;
        }
        else {
            preprocess(imgPlate[i], imgGrayscale[i], imgThresh[i]);        // preprocess to get grayscale and threshold images

                                                                            // upscale size by 60% for better viewing and character recognition

            for (int j = 0; j < vectorOfPossiblePlates[i].size(); j++) {
                cv::resize(imgThresh[i][j], imgThresh[i][j], cv::Size(), 1.6, 1.6);

                // threshold again to eliminate any gray areas
                cv::threshold(imgThresh[i][j], imgThresh[i][j], 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);

                // find all possible chars in the plate,
                // this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
                std::vector<PossibleChar> vectorOfPossibleCharsInPlate = findPossibleCharsInPlate(imgGrayscale[i][j], imgThresh[i][j]);

                // given a vector of all possible chars, find groups of matching chars within the plate
                std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInPlate = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInPlate);

                if (vectorOfVectorsOfMatchingCharsInPlate.size() == 0) {                // if no groups of matching chars were found in the plate
                    vectorOfPossiblePlates[i][j].strChars = "";            // set plate string member variable to empty string
                    continue;                               // go back to top of for loop
                }

                for (auto& vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {                                         // for each vector of matching chars in the current plate
                    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);      // sort the chars left to right
                    vectorOfMatchingChars = removeInnerOverlappingChars(vectorOfMatchingChars);                                     // and eliminate any overlapping chars
                }

                // within each possible plate, suppose the longest vector of potential matching chars is the actual vector of chars
                unsigned int intLenOfLongestVectorOfChars = 0;
                unsigned int intIndexOfLongestVectorOfChars = 0;
                // loop through all the vectors of matching chars, get the index of the one with the most chars
                for (unsigned int i = 0; i < vectorOfVectorsOfMatchingCharsInPlate.size(); i++) {
                    if (vectorOfVectorsOfMatchingCharsInPlate[i].size() > intLenOfLongestVectorOfChars) {
                        intLenOfLongestVectorOfChars = vectorOfVectorsOfMatchingCharsInPlate[i].size();
                        intIndexOfLongestVectorOfChars = i;
                    }
                }
                // suppose that the longest vector of matching chars within the plate is the actual vector of chars
                std::vector<PossibleChar> longestVectorOfMatchingCharsInPlate = vectorOfVectorsOfMatchingCharsInPlate[intIndexOfLongestVectorOfChars];

                // perform char recognition on the longest vector of matching chars in the plate
                vectorOfPossiblePlates[i][j].strChars = recognizeCharsInPlate(imgThresh[i][j], longestVectorOfMatchingCharsInPlate);
                vectorOfPossiblePlates[i][j].imgGrayscale = imgGrayscale[i][j];
                vectorOfPossiblePlates[i][j].imgThresh = imgThresh[i][j];
            }
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Chars preprocess time: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << std::endl;
    return(vectorOfPossiblePlates);
}

std::vector<PossibleChar> findPossibleCharsInPlate(cv::Mat& imgGrayscale, cv::Mat& imgThresh) {
    std::vector<PossibleChar> vectorOfPossibleChars;                            // this will be the return value

    cv::Mat imgThreshCopy;

    std::vector<std::vector<cv::Point> > contours;

    imgThreshCopy = imgThresh.clone();				// make a copy of the thresh image, this in necessary b/c findContours modifies the image

    cv::findContours(imgThreshCopy, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);        // find all contours in plate

    for (auto& contour : contours) {                            // for each contour
        PossibleChar possibleChar(contour);

        if (checkIfPossibleChar(possibleChar)) {                // if contour is a possible char, note this does not compare to other chars (yet) . . .
            vectorOfPossibleChars.push_back(possibleChar);      // add to vector of possible chars
        }
    }

    return(vectorOfPossibleChars);
}

bool checkIfPossibleChar(PossibleChar& possibleChar) {
    // this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    // note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.boundingRect.area() > MIN_PIXEL_AREA &&
        possibleChar.boundingRect.width > MIN_PIXEL_WIDTH && possibleChar.boundingRect.height > MIN_PIXEL_HEIGHT &&
        MIN_ASPECT_RATIO < possibleChar.dblAspectRatio && possibleChar.dblAspectRatio < MAX_ASPECT_RATIO) {
        return(true);
    }
    else {
        return(false);
    }
}

std::vector<std::vector<PossibleChar> > findVectorOfVectorsOfMatchingChars(const std::vector<PossibleChar>& vectorOfPossibleChars) {
    // with this function, we start off with all the possible chars in one big vector
    // the purpose of this function is to re-arrange the one big vector of chars into a vector of vectors of matching chars,
    // note that chars that are not found to be in a group of matches do not need to be considered further
    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingChars;             // this will be the return value

    for (auto& possibleChar : vectorOfPossibleChars) {                  // for each possible char in the one big vector of chars

                                                                        // find all chars in the big vector that match the current char
        std::vector<PossibleChar> vectorOfMatchingChars = findVectorOfMatchingChars(possibleChar, vectorOfPossibleChars);

        vectorOfMatchingChars.push_back(possibleChar);          // also add the current char to current possible vector of matching chars

                                                                // if current possible vector of matching chars is not long enough to constitute a possible plate
        if (vectorOfMatchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS) {
            continue;                       // jump back to the top of the for loop and try again with next char, note that it's not necessary
                                            // to save the vector in any way since it did not have enough chars to be a possible plate
        }
        // if we get here, the current vector passed test as a "group" or "cluster" of matching chars
        vectorOfVectorsOfMatchingChars.push_back(vectorOfMatchingChars);            // so add to our vector of vectors of matching chars

                                                                                    // remove the current vector of matching chars from the big vector so we don't use those same chars twice,
                                                                                    // make sure to make a new big vector for this since we don't want to change the original big vector
        std::vector<PossibleChar> vectorOfPossibleCharsWithCurrentMatchesRemoved;

        for (auto& possChar : vectorOfPossibleChars) {
            if (std::find(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), possChar) == vectorOfMatchingChars.end()) {
                vectorOfPossibleCharsWithCurrentMatchesRemoved.push_back(possChar);
            }
        }
        // declare new vector of vectors of chars to get result from recursive call
        std::vector<std::vector<PossibleChar> > recursiveVectorOfVectorsOfMatchingChars;

        // recursive call
        recursiveVectorOfVectorsOfMatchingChars = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsWithCurrentMatchesRemoved);	// recursive call !!

        for (auto& recursiveVectorOfMatchingChars : recursiveVectorOfVectorsOfMatchingChars) {      // for each vector of matching chars found by recursive call
            vectorOfVectorsOfMatchingChars.push_back(recursiveVectorOfMatchingChars);               // add to our original vector of vectors of matching chars
        }

        break;		// exit for loop
    }

    return(vectorOfVectorsOfMatchingChars);
}

std::vector<PossibleChar> findVectorOfMatchingChars(const PossibleChar& possibleChar, const std::vector<PossibleChar>& vectorOfChars) {
    // the purpose of this function is, given a possible char and a big vector of possible chars,
    // find all chars in the big vector that are a match for the single possible char, and return those matching chars as a vector
    std::vector<PossibleChar> vectorOfMatchingChars;                // this will be the return value

    for (auto& possibleMatchingChar : vectorOfChars) {              // for each char in big vector

                                                                    // if the char we attempting to find matches for is the exact same char as the char in the big vector we are currently checking
        if (possibleMatchingChar == possibleChar) {
            // then we should not include it in the vector of matches b/c that would end up double including the current char
            continue;           // so do not add to vector of matches and jump back to top of for loop
        }
        // compute stuff to see if chars are a match
        double dblDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar);
        double dblAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar);
        double dblChangeInArea = (double)abs(possibleMatchingChar.boundingRect.area() - possibleChar.boundingRect.area()) / (double)possibleChar.boundingRect.area();
        double dblChangeInWidth = (double)abs(possibleMatchingChar.boundingRect.width - possibleChar.boundingRect.width) / (double)possibleChar.boundingRect.width;
        double dblChangeInHeight = (double)abs(possibleMatchingChar.boundingRect.height - possibleChar.boundingRect.height) / (double)possibleChar.boundingRect.height;

        // check if chars match
        if (dblDistanceBetweenChars < (possibleChar.dblDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) &&
            dblAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS &&
            dblChangeInArea < MAX_CHANGE_IN_AREA &&
            dblChangeInWidth < MAX_CHANGE_IN_WIDTH &&
            dblChangeInHeight < MAX_CHANGE_IN_HEIGHT) {
            vectorOfMatchingChars.push_back(possibleMatchingChar);      // if the chars are a match, add the current char to vector of matching chars
        }
    }

    return(vectorOfMatchingChars);          // return result
}

// use Pythagorean theorem to calculate distance between two chars
double distanceBetweenChars(const PossibleChar& firstChar, const PossibleChar& secondChar) {
    int intX = abs(firstChar.intCenterX - secondChar.intCenterX);
    int intY = abs(firstChar.intCenterY - secondChar.intCenterY);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

// use basic trigonometry(SOH CAH TOA) to calculate angle between chars
double angleBetweenChars(const PossibleChar& firstChar, const PossibleChar& secondChar) {
    double dblAdj = abs(firstChar.intCenterX - secondChar.intCenterX);
    double dblOpp = abs(firstChar.intCenterY - secondChar.intCenterY);

    double dblAngleInRad = atan(dblOpp / dblAdj);

    double dblAngleInDeg = dblAngleInRad * (180.0 / CV_PI);

    return(dblAngleInDeg);
}

// if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
// this is to prevent including the same char twice if two contours are found for the same char,
// for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
std::vector<PossibleChar> removeInnerOverlappingChars(std::vector<PossibleChar>& vectorOfMatchingChars) {
    std::vector<PossibleChar> vectorOfMatchingCharsWithInnerCharRemoved(vectorOfMatchingChars);

    for (auto& currentChar : vectorOfMatchingChars) {
        for (auto& otherChar : vectorOfMatchingChars) {
            if (currentChar != otherChar) {                         // if current char and other char are not the same char . . .
                                                                    // if current char and other char have center points at almost the same location . . .
                if (distanceBetweenChars(currentChar, otherChar) < (currentChar.dblDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY)) {
                    // if we get in here we have found overlapping chars
                    // next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it

                    // if current char is smaller than other char
                    if (currentChar.boundingRect.area() < otherChar.boundingRect.area()) {
                        // look for char in vector with an iterator
                        std::vector<PossibleChar>::iterator currentCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), currentChar);
                        // if iterator did not get to end, then the char was found in the vector
                        if (currentCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(currentCharIterator);       // so remove the char
                        }
                    }
                    else {        // else if other char is smaller than current char
                                  // look for char in vector with an iterator
                        std::vector<PossibleChar>::iterator otherCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), otherChar);
                        // if iterator did not get to end, then the char was found in the vector
                        if (otherCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(otherCharIterator);         // so remove the char
                        }
                    }
                }
            }
        }
    }

    return(vectorOfMatchingCharsWithInnerCharRemoved);
}

// this is where we apply the actual char recognition
std::string recognizeCharsInPlate(cv::Mat& imgThresh, std::vector<PossibleChar>& vectorOfMatchingChars) {
    std::string strChars;               // this will be the return value, the chars in the lic plate

    cv::Mat imgThreshColor;

    // sort chars from left to right
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    cv::cvtColor(imgThresh, imgThreshColor, cv::COLOR_GRAY2BGR);       // make color version of threshold image so we can draw contours in color on it

    for (auto& currentChar : vectorOfMatchingChars) {           // for each char in plate
        cv::rectangle(imgThreshColor, currentChar.boundingRect, SCALAR_GREEN, 2);       // draw green box around the char

        cv::Mat imgROItoBeCloned = imgThresh(currentChar.boundingRect);                 // get ROI image of bounding rect

        cv::Mat imgROI = imgROItoBeCloned.clone();      // clone ROI image so we don't change original when we resize

        cv::Mat imgROIResized;
        // resize image, this is necessary for char recognition
        cv::resize(imgROI, imgROIResized, cv::Size(RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT));

        cv::Mat matROIFloat;

        imgROIResized.convertTo(matROIFloat, CV_32FC1);         // convert Mat to float, necessary for call to findNearest

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);       // flatten Matrix into one row

        cv::Mat matCurrentChar(0, 0, CV_32F);                   // declare Mat to read current char into, this is necessary b/c findNearest requires a Mat

        kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);       // convert current char from Mat to float

        strChars = strChars + char(int(fltCurrentChar));        // append current char to full string
    }

    return(strChars);               // return result
}