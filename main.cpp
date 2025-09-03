#include "main.hpp"

enum LandmarksKeypointsID
{
    LeftEyeBegin = 36,
    LeftEyeEnd = 41,
    RightEyeBegin = 42,
    RightEyeEnd = 47,
    NoseTip = 30,
    LeftMouth = 48,
    RightMouth = 54,

    Size = 68
};

/**
 * Draws a rectangle over `face` and a white filled label box above it
 * with black text `label` on that box.
 * @param frame Image to draw on.
 * @param face  Rectangle to highlight.
 * @param label Text to display.
 */
void drawRectWithLable(cv::Mat& frame, const cv::Rect& face, const std::string& label)
{
    cv::rectangle(frame, face, /*color=*/ {0, 255, 0});
    int baseLine = 0;
    const cv::Size labelSize = cv::getTextSize(label, /*font=*/ cv::FONT_HERSHEY_SIMPLEX, /*fontScale=*/ 1, /*thickness=*/ 1, &baseLine);
    cv::rectangle(frame, cv::Point(face.x, face.y - labelSize.height),
                   cv::Point(face.x + labelSize.width, face.y + baseLine), /*color=*/ {255, 255, 255}, /*thickness=*/ cv::FILLED);
    cv::putText(frame, label, cv::Point(face.x, face.y),
                 /*font=*/ cv::FONT_HERSHEY_SIMPLEX, /*fontScale=*/ 1, /*color=*/ {0, 0, 0});
}


/**
 * Tries to add ~50 px padding to the left/top and ~100 px to width/height.
 * If the expansion go outside the image, it clips to `frame` size.
 * @param frame Source image.
 * @param face  Original rectangle to expand.
 * @param xExtension x Extension in pixels.
 * @param yExtension y Extension in pixels.
 * @return Expanded rectangle that fits inside the image.
 */
cv::Rect getExtendedRect(const cv::Mat& frame, const cv::Rect& face, int xExtension = 50, int yExtension = 50)
{
    cv::Rect res = face;

    (res.x > xExtension) ? res.x -= xExtension : res.x = 0;
    (res.y > yExtension) ? res.y -= yExtension : res.y = 0;
    (res.x + res.width > frame.cols) ? res.width = frame.cols - res.x : res.width += (2 * xExtension);
    (res.y + res.height > frame.rows) ? res.height = frame.rows - res.y : res.height += (2 * yExtension);

    // std::cout << frame.size << "\n";
    // std::cout << "[" << res.x << "," <<  res.y << "," << res.x + res.width << "," << res.y + res.height << "]" << "\n";
    return res;
}


/**
 * Clamp a rectangle to the image bounds.
 * @param frame Image used for bounds (not modified).
 * @param face  Input rectangle (in pixels).
 * @return Rectangle adjusted to fit inside the image.
 */
cv::Rect getClampedRect(const cv::Mat& frame, const cv::Rect& face)
{
    cv::Rect res = face;

    if (res.x < 0) { res.x = 0; }
    if (res.y < 0) { res.y = 0; }
    if (res.x + res.width > frame.cols) { res.width = frame.cols - res.x; }
    if (res.y + res.height > frame.rows) { res.height = frame.rows - res.y; }

    // std::cout << frame.size << "\n";
    // std::cout << "[" << res.x << "," <<  res.y << "," << res.x + res.width << "," << res.y + res.height << "]" << "\n";
    return res;
}


/**
 * Align a face to a fixed 200×200 crop using eye landmarks.
 *
 * @param frame      Input image.
 * @param landmarks  2D facial landmarks in image coordinates.
 * @return Aligned face image of size 200×200.
 */
cv::Mat alignFace2D(const cv::Mat &frame, const std::vector<cv::Point2f> &landmarks)
{
    if (landmarks.size() != LandmarksKeypointsID::Size)
    {
        std::cerr << cv::format("expected landmark size: %d. Got: %zu", LandmarksKeypointsID::Size, landmarks.size()) << std::endl;
        return {};
    }

    cv::Point2f leftEyeCenter(0, 0), rightEyeCenter(0, 0);
    for (int i = LandmarksKeypointsID::LeftEyeBegin; i <= LandmarksKeypointsID::LeftEyeEnd; ++i){
        leftEyeCenter += landmarks[i];
    }
    leftEyeCenter *= (1.0 / 6.0);

    for (int i = LandmarksKeypointsID::RightEyeBegin; i <= LandmarksKeypointsID::RightEyeEnd; ++i){
        rightEyeCenter += landmarks[i];
    }
    rightEyeCenter *= (1.0 / 6.0);

    const double dy = rightEyeCenter.y - leftEyeCenter.y;
    const double dx = rightEyeCenter.x - leftEyeCenter.x;
    const double angle = atan2(dy, dx) * 180.0 / CV_PI;


    constexpr double requiredLeftEyeX = 0.35;
    constexpr double requiredLeftEyeY = 0.35;
    constexpr int requiredFaceWidth = 200;
    constexpr int requiredFaceHeight = 200;

    const cv::Point2f eyesCenter = (leftEyeCenter + rightEyeCenter) * 0.5f;

    const double dist = sqrt(dx * dx + dy * dy);
    constexpr double requiredDist = (1.0 - 2 * requiredLeftEyeX) * requiredFaceWidth;
    const double scale = requiredDist / dist;

    cv::Mat rotMat = cv::getRotationMatrix2D(eyesCenter, angle, scale);

    const cv::Point2d requiredEyesCenter(requiredFaceWidth * 0.5, requiredFaceHeight * requiredLeftEyeY);

    rotMat.at<double>(0, 2) += requiredEyesCenter.x - eyesCenter.x;
    rotMat.at<double>(1, 2) += requiredEyesCenter.y - eyesCenter.y;

    cv::Mat alignedFace;

    warpAffine(frame, alignedFace, rotMat, {requiredFaceWidth, requiredFaceHeight});

    return alignedFace;
}


/**
 * Align a face to a canonical 256×256 view using keypoints.
 *
 * @param face      Input face crop.
 * @param location  Face rectangle in the original image.
 * @param landmarks Full landmark set.
 * @return Aligned 256×256 face, or empty Mat if landmark count is invalid.
 */
cv::Mat alignFace3D(const cv::Mat &face, const cv::Rect location, const std::vector<cv::Point2f> &landmarks)
{
    if (landmarks.size() != LandmarksKeypointsID::Size)
    {
        std::cerr << cv::format("expected landmark size: %d. Got: %zu", LandmarksKeypointsID::Size, landmarks.size()) << std::endl;
        return {};
    }

    const cv::Point2f locationPoint(static_cast<float>(location.x), static_cast<float>(location.y));
    cv::Point2f leftEye(0, 0);
    for (int i = LandmarksKeypointsID::LeftEyeBegin; i <= LandmarksKeypointsID::LeftEyeEnd; ++i) {
        leftEye += landmarks[i];
    }
    leftEye *= (1.0f / 6.0f);
    leftEye -= locationPoint;

    cv::Point2f rightEye(0, 0);
    for (int i = LandmarksKeypointsID::RightEyeBegin; i <= LandmarksKeypointsID::RightEyeEnd; ++i) {
        rightEye += landmarks[i];
    }
    rightEye *= (1.0f / 6.0f);
    rightEye -= locationPoint;

    const cv::Point2f noseTip = landmarks[LandmarksKeypointsID::NoseTip] - locationPoint;
    const cv::Point2f leftMouth = landmarks[LandmarksKeypointsID::LeftMouth] - locationPoint;
    const cv::Point2f rightMouth = landmarks[LandmarksKeypointsID::RightMouth] - locationPoint;

    const std::vector<cv::Point2f> srcPoints = { leftEye, rightEye, noseTip, leftMouth, rightMouth };

    constexpr float scale = 256.0 / 112.0;

    const std::vector<cv::Point2f> perfectFace = {
        /*leftEye=*/    cv::Point2f(38.2946f * scale, 51.6963f * scale),
        /*rightEye=*/   cv::Point2f(73.5318f * scale, 51.5014f * scale),
        /*noseTip=*/    cv::Point2f(56.0252f * scale, 71.7366f * scale),
        /*leftMouth=*/  cv::Point2f(41.5493f * scale, 92.3655f * scale),
        /*rightMouth=*/ cv::Point2f(70.7299f * scale, 92.2041f * scale)
    };

    const cv::Mat H = cv::findHomography(srcPoints, perfectFace);

    cv::Mat alignedFace;
    cv::warpPerspective(face, alignedFace, H, cv::Size(256, 256));

    return alignedFace;
}


/**
 * @brief Draw 3D points on an image with depth-coded color.
 *
 * @param frame      Image to draw on.
 * @param points     3D points; x/y are pixel coords, z is used for color.
 * @param radius     Circle radius in pixels (default: 2).
 * @param thickness  Circle thickness in pixels (default: cv::FILLED).
 */
void draw3DPoints(cv::Mat& frame, const std::vector<cv::Point3f>& points, int radius = 2, int thickness = cv::FILLED)
{
    double min = points[0].z, max = points[0].z;
    for (size_t i = 1; i < points.size(); ++i)
    {
        if (points[i].z > max) max = points[i].z;
        if (points[i].z < min) min = points[i].z;
    }

    const double scale = 255.0 / (max - min);

    for (const auto& point : points)
    {
        // std::cout << (points[i].z + min) * scale << "\n";
        cv::circle(frame, /*center=*/ {static_cast<int>(point.x), static_cast<int>(point.y)}, radius,
            /*color=*/ {0, 0, (point.z - min) * scale}, thickness);
    }
}





/**
 * Live cam demo for a face finder.
 * @param finder Face finder instance.
 */
void FaceDetectorTest(const std::unique_ptr<BaseFaceFinder>& finder)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }

    const std::string windowName = "FaceDetectorTest";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    cv::Mat frame;

    while (cv::waitKey(1) != 'q')
    {
        cap.read(frame);
        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            std::string label = cv::format("confidence: %.2f", finder->confidences[i]);
            const cv::Rect face = getClampedRect(frame, finder->faces[i]);
            drawRectWithLable(frame, face, label);
        }

        cv::imshow(windowName, frame);
    }
}


/**
 * Collect face samples and train a recognizer.
 * @param finder          Face finder instance.
 * @param recognizer      Face recognizer to be trained on collected samples.
 * @param recognizerPath  Path where the trained model will be saved.
 */
void FaceRecognitionTrainTest(const std::unique_ptr<BaseFaceFinder>& finder,
    const std::unique_ptr<BaseFaceRecognizer>& recognizer,
    const std::string& recognizerPath)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }

    const std::string windowName = "FaceRecognitionTrainTest";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    cv::Mat frame;
    int faceCounter = 0, faceIndex = 0;
    bool loop = true;

    std::vector<int> labels;
    std::vector<cv::Mat> faces;

    do
    {
        cap.read(frame);

        finder->find(frame);

        for (const auto& face : finder->faces)
        {
            std::string label = cv::format("person: %d, counter: %d", faceIndex, faceCounter);
            drawRectWithLable(frame, face, label);
        }

        cv::imshow(windowName, frame);

        switch (static_cast<char>(cv::waitKey(/*delay=*/ 1)))
        {
        case 'q':
            return;

        case 'a':
            if(finder->faces.size() != 1){
                break;
            }
            faces.push_back(frame(finder->faces[0]).clone());
            labels.push_back(faceIndex);
            ++faceCounter;
            break;

        case 'n':
            ++faceIndex;
            faceCounter = 0;
            break;

        case 's':
            loop = false;
            break;

        default:
            break;
        }
    } while (loop);

    recognizer->train(faces, labels, recognizerPath);
}


/**
 * Сam demo: detect faces and run recognition, showing ID and confidence.
 * @param finder      Face finder instance.
 * @param recognizer  Trained face recognizer used to predict.
 */
void FaceRecognitionPredictTest(const std::unique_ptr<BaseFaceFinder>& finder,
    const std::unique_ptr<BaseFaceRecognizer>& recognizer)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }

    const std::string windowName = "FaceRecognitionPredictTest";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::Mat frame;

    while (cv::waitKey(1) != 'q')
    {
        cap.read(frame);

        finder->find(frame);

        for (const auto& face : finder->faces)
        {
            cv::Mat roi = frame(getClampedRect(frame, face)).clone();
            auto [id, conf] = recognizer->predict(roi);
            std::string label = cv::format("ID: %d, confidence: %.2f", id, conf);
            drawRectWithLable(frame, face, label);
        }

        cv::imshow(windowName, frame);
    }
}


/**
 * Cam demo: detect faces, find landmarks, and draw them.
 * @param finder           Face finder instance.
 * @param landmarksFinder  Landmark finder instance.
 * @param radius           Circle radius for landmark points (default: 2).
 * @param color            Landmark color in BGR (default: {255,255,255}).
 * @param thickness        Circle thickness (default: cv::FILLED).
 */
void LandmarksTest(const std::unique_ptr<BaseFaceFinder>& finder,
    const std::unique_ptr<BaseLandmarksFinder>& landmarksFinder,
    int radius = 2,
    const cv::Scalar& color = {255, 255, 255},
    int thickness = cv::FILLED)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }

    const std::string windowName = "LandmarksTest";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    cv::Mat frame;

    while (cv::waitKey(1) != 'q')
    {
        cap.read(frame);
        finder->find(frame);

        for (const auto& face : finder->faces)
        {
            cv::Rect extended = getExtendedRect(frame, face);
            drawRectWithLable(frame, extended, "");
            landmarksFinder->find(frame, extended);
            for (auto point : landmarksFinder->landmarks){
                cv::circle(frame, point, radius, color, thickness);
            }

        }

        cv::imshow(windowName, frame);
    }
}


/**
 * Cam demo: detect a face, find landmarks and align it (2D).
 * @param finder          Face detector instance.
 * @param landmarksFinder Landmark detector instance.
 */
void Alignment2DTest(const std::unique_ptr<BaseFaceFinder>& finder,
    const std::unique_ptr<BaseLandmarksFinder>& landmarksFinder)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }
    const std::string windowName = "Alignment2DTest";
    const std::string faceWindowName = "Face";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);

    cv::Mat frame, alignedFace;

    while (cv::waitKey(1) != 'q')
    {
        cap.read(frame);
        finder->find(frame);

        for (const auto& face : finder->faces)
        {
            cv::Rect extended = getExtendedRect(frame, face);
            landmarksFinder->find(frame, extended);

            alignedFace = alignFace2D(frame, landmarksFinder->landmarks);
        }

        cv::imshow(windowName, frame);
        if(alignedFace.rows != 0){
            cv::imshow(faceWindowName, alignedFace);
        }
    }
}


/**
 * Cam demo: 3D-align the first detected face and show the cropped result.
 * @param finder          Face detector instance.
 * @param landmarksFinder Landmark detector instance.
 *
 * @note Uses only the first detected face.
 */
void Alignment3DTest(const std::unique_ptr<BaseFaceFinder>& finder,
    const std::unique_ptr<BaseLandmarksFinder>& landmarksFinder)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }
    const std::string faceWindowName = "Face";
    cv::namedWindow(faceWindowName, cv::WINDOW_NORMAL);

    cv::Mat frame, face;

    while (cv::waitKey(1) != 'q')
    {
        cap.read(frame);
        finder->find(frame);

        if(finder->faces.empty())   continue;

        landmarksFinder->find(frame, getExtendedRect(frame, finder->faces[0]));

        cv::Mat alignedFace = alignFace3D(frame(finder->faces[0]), finder->faces[0], landmarksFinder->landmarks);

        finder->find(alignedFace);
        alignedFace = alignedFace(finder->faces[0]);
        cv::imshow(faceWindowName, alignedFace);
    }
}


/**
 * Collect aligned (2D) face samples from the cam and train a recognizer.
 * @param finder          Face detector instance.
 * @param recognizer      Face recognizer to train on collected aligned crops.
 * @param landmarksFinder Landmark detector instance.
 * @param recognizerPath  Path to save the trained model.
 */
void FaceRecognitionWithAlignment2DTrainTest(const std::unique_ptr<BaseFaceFinder>& finder,
    const std::unique_ptr<BaseFaceRecognizer>& recognizer,
    const std::unique_ptr<BaseLandmarksFinder>& landmarksFinder,
    const std::string& recognizerPath)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }
    const std::string windowName = "FaceRecognitionWithAlignment2DTrainTest";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    cv::Mat frame;
    int faceCounter = 0, faceIndex = 0;
    bool loop = true;

    std::vector<int> labels;
    std::vector<cv::Mat> faces;

    do
    {
        cap.read(frame);

        finder->find(frame);

        for (const auto& face : finder->faces)
        {
            std::string label = cv::format("id: %d, counter: %d", faceIndex, faceCounter);
            drawRectWithLable(frame, face, label);
        }

        cv::imshow(windowName, frame);

        switch (static_cast<char>(cv::waitKey(1)))
        {
        case 'q':
            return;

        case 'a':
        {
            if(finder->faces.size() != 1){
                break;
            }
            cv::Rect extended = getExtendedRect(frame, finder->faces[0]);
            landmarksFinder->find(frame, extended);
            cv::Mat alignedFace = alignFace2D(frame, landmarksFinder->landmarks);
            faces.push_back(alignedFace.clone());
            labels.push_back(faceIndex);
            ++faceCounter;
            break;
        }

        case 'n':
            ++faceIndex;
            faceCounter = 0;
            break;

        case 's':
            loop = false;
            break;

        default:
            break;
        }
    } while (loop);

    recognizer->train(faces, labels, recognizerPath);
}


/**
 * Cam demo: detect faces, align (2D) and recognize.
 * @param finder          Face detector instance.
 * @param recognizer      Face recognizer instance.
 * @param landmarksFinder Landmark detector instance.
 */
void FaceRecognitionWithAlignment2DPredictTest(const std::unique_ptr<BaseFaceFinder>& finder,
    const std::unique_ptr<BaseFaceRecognizer>& recognizer,
    const std::unique_ptr<BaseLandmarksFinder>& landmarksFinder)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }

    const std::string windowName = "FaceRecognitionWithAlignment2DPredictTest";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    cv::Mat frame;


    while (cv::waitKey(1) != 'q')
    {
        cap.read(frame);

        finder->find(frame);

        for (const auto& face : finder->faces)
        {
            cv::Rect extended = getExtendedRect(frame, face);
            landmarksFinder->find(frame, extended);
            cv::Mat alignedFace = alignFace2D(frame, landmarksFinder->landmarks);
            const auto [id, conf] = recognizer->predict(alignedFace);
            std::string label = cv::format("Person %d with conf %.2f", id, conf);
            drawRectWithLable(frame, face, label);
        }

        cv::imshow(windowName, frame);
    }
}


/**
 * Collect 3D-aligned face samples from the cam and train a recognizer.
 * @param finder          Face detector instance.
 * @param recognizer      Face recognizer instance.
 * @param landmarksFinder Landmark detector uinstance.
 * @param recognizerPath  Path to save the trained model.
 */
void FaceRecognitionWithAlignment3DTrainTest(const std::unique_ptr<BaseFaceFinder>& finder,
    const std::unique_ptr<BaseFaceRecognizer>& recognizer,
    const std::unique_ptr<BaseLandmarksFinder>& landmarksFinder,
    const std::string& recognizerPath)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }
    const std::string windowName = "FaceRecognitionWithAlignment3DTrainTest";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    cv::Mat frame;
    int faceCounter = 0, faceIndex = 0;
    bool loop = true;

    std::vector<int> labels;
    std::vector<cv::Mat> faces;

    do
    {
        cap.read(frame);

        finder->find(frame);

        for (const auto& face : finder->faces)
        {
            std::string label = cv::format("person: %d, counter: %d", faceIndex, faceCounter);
            drawRectWithLable(frame, face, label);
        }

        cv::imshow(windowName, frame);

        switch (static_cast<char>(cv::waitKey(1)))
        {
        case 'q':
            return;

        case 'a':
        {
            if(finder->faces.size() != 1){
                break;
            }
            cv::Rect extended = getExtendedRect(frame, finder->faces[0]);
            landmarksFinder->find(frame, extended);
            cv::Mat alignedFace = alignFace3D(frame(finder->faces[0]), finder->faces[0], landmarksFinder->landmarks);
            finder->find(alignedFace);
            faces.push_back(alignedFace(finder->faces[0]).clone());
            labels.push_back(faceIndex);
            ++faceCounter;
            break;
        }

        case 'n':
            ++faceIndex;
            faceCounter = 0;
            break;

        case 's':
            loop = false;
            break;

        default:
            break;
        }
    } while (loop);

    recognizer->train(faces, labels, recognizerPath);
}


/**
 * Cam demo: detect faces, 3D-align each face and recognize.
 * @param finder          Face detector instance.
 * @param recognizer      Face recognizer instance.
 * @param landmarksFinder Landmark detector instance.
 */
void FaceRecognitionWithAlignment3DPredictTest(const std::unique_ptr<BaseFaceFinder>& finder,
    const std::unique_ptr<BaseFaceRecognizer>& recognizer,
    const std::unique_ptr<BaseLandmarksFinder>& landmarksFinder)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }

    const std::string windowName = "FaceRecognitionWithAlignment3DPredictTest";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    cv::Mat frame;

    while (cv::waitKey(1) != 'q')
    {
        cap.read(frame);

        finder->find(frame);
        std::vector<cv::Rect> faces = finder->faces;

        for (const auto& face : faces)
        {
            cv::Rect extended = getExtendedRect(frame, face);
            landmarksFinder->find(frame, extended);

            cv::Rect roi = face;

            cv::Mat alignedFace = alignFace3D(frame(roi).clone(), face, landmarksFinder->landmarks);
            finder->find(alignedFace);

            (finder->faces.empty()) ? (roi = cv::Rect(0, 0, alignedFace.cols, alignedFace.rows))
                                    : roi = getClampedRect(alignedFace, finder->faces[0]);

            const auto [id, conf] = recognizer->predict(alignedFace(roi).clone());
            std::string label = cv::format("Person %d with conf %.2f", id, conf);
            drawRectWithLable(frame, face, label);
        }

        cv::imshow(windowName, frame);
    }
}



int main()
{
    const std::unique_ptr<BaseFaceFinder> finder = std::make_unique<DNNFaceFinder>();
    const std::string finderPath = "models/faceDetectors/dnn/yolov11n-face.onnx";
    finder->read(finderPath);

    FaceDetectorTest(finder);

    const std::string dnnWeightsPath = "models/recognizers/dnn/arcface.onnx";
    const std::unique_ptr<BaseFaceRecognizer> recognizer = std::make_unique<DNNRecognizer>(dnnWeightsPath);
    const std::string recognizerPath = "models/recognizers/dnn/DNNFaceRecognizer.xml";
    recognizer->read(recognizerPath);

    // FaceRecognitionTrainTest(finder, recognizer, finderPath, recognizerPath);
    // FaceRecognitionPredictTest(finder, recognizer, finderPath, recognizerPath);


    const std::unique_ptr<BaseLandmarksFinder> landmarksFinder = std::make_unique<DNNLandmarksFinder>();
    const std::string landmarksPath = "models/landmarks/2D/dnn/face_alignment.onnx";
    landmarksFinder->read(landmarksPath);

    // LandmarksTest(finder, landmarksFinder, finderPath, landmarksPath);
    // Alignment2DTest(finder, landmarksFinder, finderPath, landmarksPath);
    // Alignment3DTest(finder, landmarksFinder, finderPath, landmarksPath);


    // FaceRecognitionWithAlignment2DTrainTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);
    // FaceRecognitionWithAlignment2DPredictTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);

    // FaceRecognitionWithAlignment3DTrainTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);
    // FaceRecognitionWithAlignment3DPredictTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);

    return 0;
}
