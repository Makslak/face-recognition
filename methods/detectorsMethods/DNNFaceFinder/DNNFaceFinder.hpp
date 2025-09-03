#pragma once

#include <opencv2/opencv.hpp>
#include "../BaseFaceFinder/BaseFaceFinder.hpp"


/* Face detector built on OpenCV DNN. */
class DNNFaceFinder : public BaseFaceFinder
{
public:

    /** Load the network from file. */
    void read(const std::string& path) override;

    /**
     * @brief Load the network and select the compute target.
     * @param path    Path to model file.
     * @param opencl  If true, prefer an OpenCL target.
     */
    void read(const std::string& path, bool opencl);

    /**
     * Run face detection on an image.
     * @param frame          Input image.
     * @param confThreshold  Minimum confidence to keep a detection.
     */
    void find(const cv::Mat& frame, float confThreshold = 0.5f) override;


private:

    cv::dnn::Net _Net;
};