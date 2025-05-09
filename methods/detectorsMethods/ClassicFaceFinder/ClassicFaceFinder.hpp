#pragma once

#include <opencv2/opencv.hpp>
#include "../BaseFaceFinder/BaseFaceFinder.hpp"


class ClassicFaceFinder : public BaseFaceFinder
{
public:

    void read(const std::string& path) override;
    void find(const cv::Mat& frame, float confThreshold = 0.5f) override;


private:

    cv::CascadeClassifier _classifier;
};