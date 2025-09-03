#include "ClassicFaceFinder.hpp"


void ClassicFaceFinder::read(const std::string &path){
    this->_classifier.load(path);
}


void ClassicFaceFinder::find(const cv::Mat &frame, float confThreshold)
{
    cv::Mat image;
    cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> rects;
    std::vector<int> rejectLevels;
    std::vector<double> levelWeights;

    constexpr double scaleFactor = 1.1;
    constexpr int minNeighbors = 3;
    const cv::Size minSize(30, 30);
    this->_classifier.detectMultiScale(image, rects, rejectLevels, levelWeights, scaleFactor, minNeighbors,
        /*flags=*/ 0, minSize, /*maxSize=*/ cv::Size(), true);

    this->confidences.clear();

    for (const double& levelWeight : levelWeights) {
        this->confidences.push_back(static_cast<float>(levelWeight));
    }

    this->faces = rects;
}
