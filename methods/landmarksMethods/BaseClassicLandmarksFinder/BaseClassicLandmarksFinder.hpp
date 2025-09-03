#pragma once

#include "../BaseLandmarksFinder/BaseLandmarksFinder.hpp"
#include <opencv2/face.hpp>


/* Base class for OpenCV classic keypoints finders.*/
class BaseClassicLandmarksFinder : public BaseLandmarksFinder
{
public:

    /* Load the facemark parameters. */
    void read(const std::string& path) override;

    /* Detect landmarks on the given face. */
    void find(const cv::Mat& frame, const cv::Rect& face) override;

    virtual ~BaseClassicLandmarksFinder(){};
protected:
    cv::Ptr<cv::face::Facemark> _facemark;
};