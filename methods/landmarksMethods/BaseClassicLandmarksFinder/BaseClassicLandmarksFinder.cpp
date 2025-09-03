#include "BaseClassicLandmarksFinder.hpp"

void BaseClassicLandmarksFinder::read(const std::string &path){
    this->_facemark->loadModel(path);
}


void BaseClassicLandmarksFinder::find(const cv::Mat &frame, const cv::Rect &face)
{
    this->landmarks.clear();

    const std::vector<cv::Rect> faces{this->getExtendedRect(frame, face)};
    std::vector<std::vector<cv::Point2f>> landmarks;
    std::vector<cv::Point2f> tempMarks;

    this->_facemark->fit(frame, faces, landmarks);    

    this->landmarks = landmarks[0];
}