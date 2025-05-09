#pragma once

#include "../BaseFaceRecognizer/BaseFaceRecognizer.hpp"
#include <opencv2/face.hpp>

class BaseClassicFaceRecognizer : public BaseFaceRecognizer
{
public:

    void train(const std::vector<cv::Mat> &faces, const std::vector<int> &facesLabels, const std::string& savePath) override;
    void read(const std::string& path) override;
    std::pair<int, float> predict(const cv::Mat& frame) override;

    virtual ~BaseClassicFaceRecognizer(){};

protected:

    BaseClassicFaceRecognizer(){};

    cv::Ptr<cv::face::FaceRecognizer> _model;
};