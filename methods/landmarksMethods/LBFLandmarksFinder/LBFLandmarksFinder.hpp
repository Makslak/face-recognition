#pragma once

#include "../BaseClassicLandmarksFinder/BaseClassicLandmarksFinder.hpp"

/* LBF (Local Binary Features) facial landmark finder. */
class LBFLandmarksFinder : public BaseClassicLandmarksFinder
{
public:

    /* Construct and initialize the LBF instance. */
    LBFLandmarksFinder();
};