#pragma once

#include "../BaseLandmarksFinder/BaseLandmarksFinder.hpp"


/* Facial landmark detector backed by OpenCV DNN. */
class DNNLandmarksFinder : public BaseLandmarksFinder
{
public:

    /* Load the landmark DNN model. */
    void read(const std::string& path) override;

    /* Detect facial landmarks on the given face. */
    void find(const cv::Mat& frame, const cv::Rect& face) override;

private:

    cv::dnn::Net _net;
};