#pragma once

#include <opencv2/opencv.hpp>

/* Base interface for face recognizers. */
class BaseFaceRecognizer
{
public:

    /* Train a model on the given face images and labels */
    virtual void train(const std::vector<cv::Mat> &faces, const std::vector<int> &facesLabels, const std::string& savePath) = 0;

    /* Load a pretrained model from file. */
    virtual void read(const std::string& path) = 0;

    /**
     * Predict face from image of face.
     *
     * @param frame  Input face image.
     * @return Pair {labelId, confidence}.
     */
    virtual std::pair<int, float> predict(const cv::Mat& frame) = 0;

    virtual ~BaseFaceRecognizer() = default;
};