#pragma once

#include "../BaseClassicFaceRecognizer/BaseClassicFaceRecognizer.hpp"


/* LDA-based Fisherfaces recognizer */
class FisherFaceRecognizer : public BaseClassicFaceRecognizer
{
public:

    /* Construct and initialize the FisherFace model */
    FisherFaceRecognizer();
};