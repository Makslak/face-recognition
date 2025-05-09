#include "FisherFaceRecognizer.hpp"

FisherFaceRecognizer::FisherFaceRecognizer(){
    this->_model = cv::face::FisherFaceRecognizer::create();
}