#pragma once

#include "../BaseLandmarksFinder/BaseLandmarksFinder.hpp"


class DNNLandmarksFinder : public BaseLandmarksFinder
{
public:

    void read(const std::string& path) override;
    void find(const cv::Mat& frame, const cv::Rect& face) override;

private:

    cv::dnn::Net _net;
};