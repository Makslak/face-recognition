#include "DNNFaceFinder.hpp"


void DNNFaceFinder::read(const std::string &path){
    this->_Net = cv::dnn::readNetFromONNX(path);
}

void DNNFaceFinder::read(const std::string &path, bool opencl)
{
    this->read(path);
    if(!opencl) return;

    this->_Net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->_Net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
}

void DNNFaceFinder::find(const cv::Mat &frame, float confThreshold)
{
    const cv::Size size = {640, 640};
    const cv::Mat blob = cv::dnn::blobFromImage(frame, /*scale=*/ 1.0/255.0, size, /*mean= */ {0, 0, 0}, /*swapRB= */true);
    this->_Net.setInput(blob);
    cv::Mat output = this->_Net.forward();

    this->faces.clear();
    this->confidences.clear();

    constexpr int outputSize = 8400;

    const float* data = output.ptr<float>();

    for (size_t i = 0; i < outputSize; ++i)
    {
        float confidence = data[4 * outputSize + i];
        if(confidence < confThreshold) continue;

        const float cx = data[i];
        const float cy = data[outputSize + i];
        const float w = data[2 * outputSize + i];
        const float h = data[3 * outputSize + i];

        int width = static_cast<int>((w * frame.cols) / size.width);
        int height = static_cast<int>((h * frame.rows) / size.height);
        const int x = static_cast<int>((cx * frame.cols) / size.width) - width / 2;
        const int y = static_cast<int>((cy * frame.rows) / size.height) - height / 2;

        if((x + width) > frame.cols) {width = frame.cols - x - 1;}
        if((y + height) > frame.rows) {height = frame.rows - y - 1;}

        this->faces.emplace_back(x, y, width, height);
        this->confidences.push_back(confidence);
    }

    constexpr float nmsThreshold = 0.4;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(this->faces, this->confidences, confThreshold, nmsThreshold, indices);

    std::vector<cv::Rect> finalFaces;
    std::vector<float> finalConfidences;
    for (const int idx : indices)
    {
        finalFaces.push_back(this->faces[idx]);
        finalConfidences.push_back(this->confidences[idx]);
    }

    this->faces = finalFaces;
    this->confidences = finalConfidences;
}
