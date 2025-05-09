#pragma once
#include <opencv2/opencv.hpp>


class BaseFaceFinder
{
public:

    virtual void read(const std::string& path) = 0;
    virtual void find(const cv::Mat& frame, float confThreshold = 0.5f) = 0;

    virtual ~BaseFaceFinder(){};

    std::vector<cv::Rect> faces;
    std::vector<float> confidences;
};