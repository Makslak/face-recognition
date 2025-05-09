#pragma once

#include <opencv2/opencv.hpp>

class BaseLandmarksFinder
{
public:

    virtual void read(const std::string& path) = 0;
    virtual void find(const cv::Mat& frame, const cv::Rect& face) = 0;

    virtual ~BaseLandmarksFinder(){};

    std::vector<cv::Point2f> landmarks;

protected:
    cv::Rect getExtendedRect(const cv::Mat& frame, const cv::Rect& rect);
};