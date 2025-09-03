#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <vector>
#include <chrono>
#include <numeric>
#include <fstream>

#include "methods/detectorsMethods/BaseFaceFinder/BaseFaceFinder.hpp"
#include "methods/detectorsMethods/ClassicFaceFinder/ClassicFaceFinder.hpp"
#include "methods/detectorsMethods/DNNFaceFinder/DNNFaceFinder.hpp"

#include "methods/landmarksMethods/BaseClassicLandmarksFinder/BaseClassicLandmarksFinder.hpp"
#include "methods/landmarksMethods/BaseLandmarksFinder/BaseLandmarksFinder.hpp"
#include "methods/landmarksMethods/KazemiLandmarksFinder/KazemiLandmarksFinder.hpp"
#include "methods/landmarksMethods/LBFLandmarksFinder/LBFLandmarksFinder.hpp"
#include "methods/landmarksMethods/DNNLandmarksFinder/DNNLandmarksFinder.hpp"

#include "methods/recognizerMethods/BaseClassicFaceRecognizer/BaseClassicFaceRecognizer.hpp"
#include "methods/recognizerMethods/BaseFaceRecognizer/BaseFaceRecognizer.hpp"
#include "methods/recognizerMethods/DNNRecognizer/DNNRecognizer.hpp"
#include "methods/recognizerMethods/EigenFaceRecognizer/EigenFaceRecognizer.hpp"
#include "methods/recognizerMethods/FisherFaceRecognizer/FisherFaceRecognizer.hpp"
#include "methods/recognizerMethods/LBPHFaceRecognizer/LBPHFaceRecognizer.hpp"