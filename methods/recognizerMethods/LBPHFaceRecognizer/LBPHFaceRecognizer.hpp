#pragma once

#include "../BaseClassicFaceRecognizer/BaseClassicFaceRecognizer.hpp"

/* Local Binary Patterns Histograms (LBPH) recognizer */
class LBPHFaceRecognizer : public BaseClassicFaceRecognizer
{
public:

    /** Construct and initialize the LBPH model. */
    LBPHFaceRecognizer();
};