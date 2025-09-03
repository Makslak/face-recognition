#pragma once

#include "../BaseClassicFaceRecognizer/BaseClassicFaceRecognizer.hpp"

/* PCA-based Eigenfaces recognizer */
class EigenFaceRecognizer : public BaseClassicFaceRecognizer
{
public:

    /* Construct and initialize the EigenFace model */
    EigenFaceRecognizer();
};