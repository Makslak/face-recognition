#pragma once

#include "../BaseFaceRecognizer/BaseFaceRecognizer.hpp"
#include <opencv2/face.hpp>

/* Base class for OpenCV classic face recognizers (Eigen/Fisher/LBPH).*/
class BaseClassicFaceRecognizer : public BaseFaceRecognizer
{
public:

    /* Train a model on the given face images and labels */
    void train(const std::vector<cv::Mat> &faces, const std::vector<int> &facesLabels, const std::string& savePath) override;

    /* Load a pretrained model from file. */
    void read(const std::string& path) override;

    /**
     * Predict face from image of face.
     *
     * @param frame  Input face image.
     * @return Pair {labelId, confidence}.
     */
    std::pair<int, float> predict(const cv::Mat& frame) override;

protected:

    BaseClassicFaceRecognizer(){};

    cv::Ptr<cv::face::FaceRecognizer> _model;
};