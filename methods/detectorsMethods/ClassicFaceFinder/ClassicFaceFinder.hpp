#pragma once

#include <opencv2/opencv.hpp>
#include "../BaseFaceFinder/BaseFaceFinder.hpp"


/**
 * Face detector based on OpenCV's classic cascades (Haar/LBP).
 */
class ClassicFaceFinder : public BaseFaceFinder
{
public:

    /** Load the cascade model from file. */
    void read(const std::string& path) override;

    /**
     * Detect faces in the given frame.
     * @param frame Input image.
     * @param confThreshold Minimum confidence to keep a detection.
     */
    void find(const cv::Mat& frame, float confThreshold = 0.5f) override;

private:

    cv::CascadeClassifier _classifier;
};