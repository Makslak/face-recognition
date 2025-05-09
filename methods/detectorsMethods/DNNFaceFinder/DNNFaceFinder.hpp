#pragma once

#include <opencv2/opencv.hpp>
#include "../BaseFaceFinder/BaseFaceFinder.hpp"


class DNNFaceFinder : public BaseFaceFinder
{
public:

    void read(const std::string& path) override;
    void read(const std::string& path, bool opencl);
    void find(const cv::Mat& frame, float confThreshold = 0.5f) override;


private:

    cv::dnn::Net _Net;
};