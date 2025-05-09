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

    double scaleFactor = 1.1;
    int minNeighbors = 3;
    cv::Size minSize(30, 30);
    this->_classifier.detectMultiScale(image, rects, rejectLevels, levelWeights, scaleFactor, minNeighbors, 0, minSize, cv::Size(), true);

    this->confidences.clear();

    for (size_t i = 0; i < levelWeights.size(); i++) {
        this->confidences.push_back(levelWeights[i]);
    }

    this->faces = rects;
}
