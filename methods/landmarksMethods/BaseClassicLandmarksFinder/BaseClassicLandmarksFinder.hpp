#pragma once

#include "../BaseLandmarksFinder/BaseLandmarksFinder.hpp"
#include <opencv2/face.hpp>

class BaseClassicLandmarksFinder : public BaseLandmarksFinder
{
public:

    void read(const std::string& path) override;
    void find(const cv::Mat& frame, const cv::Rect& face) override;

    virtual ~BaseClassicLandmarksFinder(){};
protected:
    cv::Ptr<cv::face::Facemark> _facemark;
};