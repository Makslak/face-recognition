#include "EigenFaceRecognizer.hpp"

EigenFaceRecognizer::EigenFaceRecognizer(){
    this->_model = cv::face::EigenFaceRecognizer::create();
}