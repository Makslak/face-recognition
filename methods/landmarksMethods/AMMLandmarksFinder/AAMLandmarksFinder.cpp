#include "AAMLandmarksFinder.hpp"

AAMLandmarksFinder::AAMLandmarksFinder(){
    this->_facemark = cv::face::FacemarkAAM::create();
}