#pragma once

#include <opencv2/opencv.hpp>

/* Base interface for facial landmark detectors. */
class BaseLandmarksFinder
{
public:

    /* Load the landmark detector */
    virtual void read(const std::string& path) = 0;

    /* Detect facial landmarks inside a face */
    virtual void find(const cv::Mat& frame, const cv::Rect& face) = 0;

    virtual ~BaseLandmarksFinder() = default;

public:
    std::vector<cv::Point2f> landmarks;

protected:
    cv::Rect getExtendedRect(const cv::Mat& frame, const cv::Rect& rect);
};