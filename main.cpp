#include "main.hpp"

class Timer 
{
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()), end_time(std::chrono::high_resolution_clock::now()) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    double getElapsedTime() const 
    {
        std::chrono::duration<double> elapsed = end_time - start_time;
        return elapsed.count();
    }

private:

    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};


void drawRectWithLable(cv::Mat& frame, cv::Rect& face, std::string label)
{
    cv::rectangle(frame, face, {0, 255, 0});
    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
    cv::rectangle(frame, cv::Point(face.x, face.y - labelSize.height),
                   cv::Point(face.x + labelSize.width, face.y + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(face.x, face.y),
                 cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
}


cv::Rect getExtendedRect(const cv::Mat& frame, const cv::Rect& face)
{
    cv::Rect res = face;

    (res.x > 50) ? res.x -= 50 : res.x = 0;
    (res.y > 50) ? res.y -= 50 : res.y = 0;
    (res.x + res.width > frame.cols) ? res.width = frame.cols - res.x : res.width += 100;
    (res.y + res.height > frame.rows) ? res.height = frame.rows - res.y : res.height += 100;

    // std::cout << frame.size << "\n";

    // std::cout << "[" << res.x << "," <<  res.y << "," << res.x + res.width << "," << res.y + res.height << "]" << "\n";

    return res;
}


cv::Rect getNormalRect(const cv::Mat& frame, const cv::Rect& face)
{
    cv::Rect res = face;

    (res.x < 0) ? res.x = 0 : res.x = res.x;
    (res.y < 0) ? res.y = 0 : res.y = res.y;
    (res.x + res.width > frame.cols) ? res.width = frame.cols - res.x : res.width = res.width;
    (res.y + res.height > frame.rows) ? res.height = frame.rows - res.y : res.height = res.height;

    // std::cout << frame.size << "\n";

    // std::cout << "[" << res.x << "," <<  res.y << "," << res.x + res.width << "," << res.y + res.height << "]" << "\n";

    return res;
}


cv::Mat frontalizeFace2D(const cv::Mat &frame, const std::vector<cv::Point2f> &landmarks)
{
    cv::Point2f leftEyeCenter(0, 0), rightEyeCenter(0, 0);
    for (int i = 36; i <= 41; ++i){
        leftEyeCenter += landmarks[i];
    }
    leftEyeCenter *= (1.0 / 6.0);
    for (int i = 42; i <= 47; ++i){
        rightEyeCenter += landmarks[i];
    }
    rightEyeCenter *= (1.0 / 6.0);

    double dy = rightEyeCenter.y - leftEyeCenter.y;
    double dx = rightEyeCenter.x - leftEyeCenter.x;
    double angle = atan2(dy, dx) * 180.0 / CV_PI;

    const double desiredLeftEyeX = 0.35;
    const double desiredLeftEyeY = 0.35;
    const int desiredFaceWidth = 200;
    const int desiredFaceHeight = 200;

    cv::Point2f eyesCenter = (leftEyeCenter + rightEyeCenter) * 0.5f;

    double dist = sqrt(dx * dx + dy * dy);
    double desiredDist = (1.0 - 2 * desiredLeftEyeX) * desiredFaceWidth;
    double scale = desiredDist / dist;

    cv::Mat rotMat = getRotationMatrix2D(eyesCenter, angle, scale);

    cv::Point2f desiredEyesCenter(desiredFaceWidth * 0.5f, desiredFaceHeight * desiredLeftEyeY);

    rotMat.at<double>(0, 2) += desiredEyesCenter.x - eyesCenter.x;
    rotMat.at<double>(1, 2) += desiredEyesCenter.y - eyesCenter.y;

    cv::Mat alignedFace;

    warpAffine(frame, alignedFace, rotMat, {desiredFaceWidth, desiredFaceHeight});

    return alignedFace;
}


cv::Mat frontalizeFace3D(const cv::Mat &face, cv::Rect location, const std::vector<cv::Point2f> &landmarks)
{
    if (landmarks.size() != 68) {
        std::cerr << "Ошибка: ожидается 68 ключевых точек, получено " << landmarks.size() << std::endl;
        return face.clone();
    }

    cv::Point2f locationPoint(location.x, location.y);
    cv::Point2f leftEye(0, 0);
    for (int i = 36; i <= 41; ++i) {
        leftEye += landmarks[i];
    }
    leftEye *= (1.0f / 6.0f);
    leftEye -= locationPoint;

    cv::Point2f rightEye(0, 0);
    for (int i = 42; i <= 47; ++i) {
        rightEye += landmarks[i];
    }
    rightEye *= (1.0f / 6.0f);
    rightEye -= locationPoint;

    cv::Point2f noseTip = landmarks[30] - locationPoint;

    cv::Point2f leftMouth = landmarks[48] - locationPoint;

    cv::Point2f rightMouth = landmarks[54] - locationPoint;

    std::vector<cv::Point2f> srcPoints = { leftEye, rightEye, noseTip, leftMouth, rightMouth };

    double scale = 256.0 / 112.0;
    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(38.2946f * scale, 51.6963f * scale),
        cv::Point2f(73.5318f * scale, 51.5014f * scale),
        cv::Point2f(56.0252f * scale, 71.7366f * scale),
        cv::Point2f(41.5493f * scale, 92.3655f * scale),
        cv::Point2f(70.7299f * scale, 92.2041f * scale)
    };

    cv::Mat H = cv::findHomography(srcPoints, dstPoints);

    cv::Mat alignedFace;
    cv::warpPerspective(face, alignedFace, H, cv::Size(256, 256));

    return alignedFace;
}


void draw3DPoints(cv::Mat& frame, std::vector<cv::Point3f>& points)
{
    double min = points[0].z, max = points[0].z;
    for (size_t i = 1; i < points.size(); ++i)
    {
        if (points[i].z > max) max = points[i].z;
        if (points[i].z < min) min = points[i].z;
    }

    double scale = 255.0 / (max - min);

    for (size_t i = 0; i < points.size(); ++i)
    {
        std::cout << (points[i].z + min) * scale << "\n";
        cv::circle(frame, {static_cast<int>(points[i].x), static_cast<int>(points[i].y)}, 2, {0, 0, (points[i].z - min) * scale}, 2);
    }
}




void FaceDetectorTest(cv::Ptr<BaseFaceFinder> finder, std::string& path)
{
    finder->read(path);

    cv::VideoCapture cap(0);
    cv::Mat frame;

    cv::Rect face;

    while (true)
    {
        cap.read(frame);
        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            std::string label = cv::format("conf: %.2f", finder->confidences[i]);
            face = getNormalRect(frame, finder->faces[i]);
            drawRectWithLable(frame, face, label);
        }

        cv::imshow("FaceDetectorTest", frame);
        if(cv::waitKey(1) == 'q'){
            break;
        }
    }
}


void FaceRecognitionTrainTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, std::string& finderPath, std::string& recognizerPath)
{
    finder->read(finderPath);

    cv::VideoCapture cap(0);
    cv::Mat frame;
    int counter = 0, index = 0;
    std::string label;
    char key;
    bool loop = true;

    std::vector<int> labels;
    std::vector<cv::Mat> faces;

    do
    {
        cap.read(frame);

        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            std::string label = cv::format("person: %d, counter: %d", index, counter);
            drawRectWithLable(frame, finder->faces[i], label);
        }

        cv::imshow("FaceRecognitionsTrainTest", frame);
        key = cv::waitKey(1);

        switch (key)
        {
        case 'q':
            return;

        case 'a':
            if(finder->faces.size() != 1){
                break;
            }
            faces.push_back(frame(finder->faces[0]).clone());
            labels.push_back(index);
            ++counter;
            break;

        case 'n':
            ++index;
            counter = 0;
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


void FaceRecognitionTrainOnPhotosTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, std::string& finderPath, std::string& recognizerPath)
{
    finder->read(finderPath);

    cv::Mat frame;
    int i = 0, index = 0;

    std::vector<int> labels;
    std::vector<cv::Mat> faces;

    while (true)
    {
        frame = cv::imread(cv::format("dataframes/regognition/ideal/img_%d.png", i));
        if(frame.empty())   break;

        finder->find(frame);

        faces.push_back(frame(finder->faces[0]).clone());
        labels.push_back(index);
        if (i % 10 == 9){
            ++index;
        }
        ++i;
    }

    recognizer->train(faces, labels, recognizerPath);
}


void FaceRecognitionPredictTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, std::string& finderPath, std::string& recognizerPath)
{
    finder->read(finderPath);
    recognizer->read(recognizerPath);
    cv::Mat frame;
    cv::VideoCapture cap(0);

    std::string label;
    std::pair<int, float> predictRes;

    while (true)
    {
        cap.read(frame);

        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            cv::Mat roi = frame(getNormalRect(frame, finder->faces[i])).clone();
            predictRes = recognizer->predict(roi);
            label = cv::format("ID: %d, distance: %.2f", predictRes.first, predictRes.second);
            drawRectWithLable(frame, finder->faces[i], label);
        }

        cv::imshow("FaceRecognitionPredictTest", frame);
        if (cv::waitKey(1) == 'q'){
            break;
        }
    }
}


void LandmarksTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& finderPath, std::string& landmarksPath)
{
    finder->read(finderPath);
    landmarksFinder->read(landmarksPath);

    cv::VideoCapture cap(0);
    cv::Mat frame;

    while (true)
    {
        cap.read(frame);
        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            cv::Rect extended = getExtendedRect(frame, finder->faces[i]);
            drawRectWithLable(frame, extended, "");
            landmarksFinder->find(frame, extended);
            for (size_t j = 0; j < landmarksFinder->landmarks.size(); ++j){
                cv::circle(frame, landmarksFinder->landmarks[j], 2, {255, 255, 255}, 2);
            }

        }

        cv::imshow("LandmarksTest", frame);
        if(cv::waitKey(1) == 'q'){
            break;
        }
    }
}


void Allignment2DTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& finderPath, std::string& landmarksPath)
{
    finder->read(finderPath);
    landmarksFinder->read(landmarksPath);

    cv::VideoCapture cap(0);
    cv::Mat frame, alignedFace;

    while (true)
    {
        cap.read(frame);
        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            cv::Rect extended = getExtendedRect(frame, finder->faces[i]);
            landmarksFinder->find(frame, extended);

            alignedFace = frontalizeFace2D(frame, landmarksFinder->landmarks);
        }

        cv::imshow("Allignment2DTest", frame);
        if(alignedFace.rows != 0){
            cv::imshow("Rotated", alignedFace);
        }
        if(cv::waitKey(1) == 'q'){
            break;
        }
    }
}


void Allignment3DTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& finderPath, std::string& landmarksPath)
{
    finder->read(finderPath);
    landmarksFinder->read(landmarksPath);

    cv::VideoCapture cap(0);
    cv::Mat frame, alignedFace, face;

    while (true)
    {
        cap.read(frame);
        finder->find(frame);

        if(finder->faces.size() == 0)   continue;

        landmarksFinder->find(frame, getExtendedRect(frame, finder->faces[0]));

        alignedFace = frontalizeFace3D(frame(finder->faces[0]), finder->faces[0], landmarksFinder->landmarks);

        finder->find(alignedFace);
        alignedFace = alignedFace(finder->faces[0]);
        cv::imshow("Rotated", alignedFace);
        if(cv::waitKey(1) == 'q'){
            break;
        }
    }
}


void FaceRecognitionWithAllignment2DTrainTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& faceFinderPath, std::string& recognizerPath, std::string& landmarksPath)
{
    finder->read(faceFinderPath);
    recognizer->read(recognizerPath);
    landmarksFinder->read(landmarksPath);

    cv::VideoCapture cap(0);
    cv::Mat frame, allignedFace;
    cv::Rect extended;
    int counter = 0, index = 0;
    std::string label;
    char key;
    bool loop = true;

    std::vector<int> labels;
    std::vector<cv::Mat> faces;

    do
    {
        cap.read(frame);

        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            std::string label = cv::format("person: %d, counter: %d", index, counter);
            drawRectWithLable(frame, finder->faces[i], label);
        }

        cv::imshow("FaceRecognitionsTrainTest", frame);
        key = cv::waitKey(1);

        switch (key)
        {
        case 'q':
            return;

        case 'a':
            if(finder->faces.size() != 1){
                break;
            }
            extended = getExtendedRect(frame, finder->faces[0]);
            landmarksFinder->find(frame, extended);
            allignedFace = frontalizeFace2D(frame, landmarksFinder->landmarks);
            faces.push_back(allignedFace.clone());
            labels.push_back(index);
            ++counter;
            break;

        case 'n':
            ++index;
            counter = 0;
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


void FaceRecognitionWithAllignment2DPredictTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& faceFinderPath, std::string& recognizerPath, std::string& landmarksPath)
{
    finder->read(faceFinderPath);
    recognizer->read(recognizerPath);
    landmarksFinder->read(landmarksPath);
    cv::Mat frame, allidnedFace;
    cv::VideoCapture cap(0);

    std::string label;
    std::pair<int, float> predictRes;


    while (true)
    {
        cap.read(frame);

        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            cv::Rect extended = getExtendedRect(frame, finder->faces[i]);
            landmarksFinder->find(frame, extended);
            allidnedFace = frontalizeFace2D(frame, landmarksFinder->landmarks);
            predictRes = recognizer->predict(allidnedFace);
            label = cv::format("Person %d with %.2f", predictRes.first, predictRes.second);
            drawRectWithLable(frame, finder->faces[i], label);
        }

        cv::imshow("FaceRecognitionPredictTest", frame);
        if (cv::waitKey(1) == 'q'){
            break;
        }
    }
}


void FaceRecognitionWithAllignment3DTrainTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& faceFinderPath, std::string& recognizerPath, std::string& landmarksPath)
{
    finder->read(faceFinderPath);
    recognizer->read(recognizerPath);
    landmarksFinder->read(landmarksPath);

    cv::VideoCapture cap(0);
    cv::Mat frame, allignedFace;
    cv::Rect extended;
    int counter = 0, index = 0;
    std::string label;
    char key;
    bool loop = true;

    std::vector<int> labels;
    std::vector<cv::Mat> faces;

    do
    {
        cap.read(frame);

        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            std::string label = cv::format("person: %d, counter: %d", index, counter);
            drawRectWithLable(frame, finder->faces[i], label);
        }

        cv::imshow("FaceRecognitionsTrainTest", frame);
        key = cv::waitKey(1);

        switch (key)
        {
        case 'q':
            return;

        case 'a':
            if(finder->faces.size() != 1){
                break;
            }
            extended = getExtendedRect(frame, finder->faces[0]);
            landmarksFinder->find(frame, extended);
            allignedFace = frontalizeFace3D(frame(finder->faces[0]), finder->faces[0], landmarksFinder->landmarks);
            finder->find(allignedFace);
            faces.push_back(allignedFace(finder->faces[0]).clone());
            labels.push_back(index);
            ++counter;
            break;

        case 'n':
            ++index;
            counter = 0;
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


void FaceRecognitionWithAllignment3DPredictTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& faceFinderPath, std::string& recognizerPath, std::string& landmarksPath)
{
    finder->read(faceFinderPath);
    recognizer->read(recognizerPath);
    landmarksFinder->read(landmarksPath);
    cv::Mat frame, allidnedFace;
    cv::Rect extended, roi;
    cv::VideoCapture cap(0);

    std::vector<cv::Rect> faces;

    std::string label;
    std::pair<int, float> predictRes;

    while (true)
    {
        cap.read(frame);

        finder->find(frame);
        faces = finder->faces;

        for (size_t i = 0; i < faces.size(); ++i)
        {
            extended = getExtendedRect(frame, faces[i]);
            landmarksFinder->find(frame, extended);

            roi = faces[i];

            allidnedFace = frontalizeFace3D(frame(roi).clone(), faces[i], landmarksFinder->landmarks);
            finder->find(allidnedFace);

            (finder->faces.size() == 0) ? (roi = cv::Rect(0, 0, allidnedFace.cols, allidnedFace.rows)) : roi = getNormalRect(allidnedFace, finder->faces[0]);

            predictRes = recognizer->predict(allidnedFace(roi).clone());
            label = cv::format("Person %d with %.2f", predictRes.first, predictRes.second);
            drawRectWithLable(frame, faces[i], label);
        }

        cv::imshow("FaceRecognitionPredictTest", frame);
        if (cv::waitKey(1) == 'q'){
            break;
        }
    }
}


void FaceRecognitionWithAllignment3DTrainOnPhotosTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& faceFinderPath, std::string& recognizerPath, std::string& landmarksPath)
{
    finder->read(faceFinderPath);
    landmarksFinder->read(landmarksPath);

    int i = 0, index = 0;

    std::vector<int> labels;
    std::vector<cv::Mat> faces;

    while (true)
    {
        cv::Mat frame = cv::imread(cv::format("dataframes/regognition/ideal/img_%d.png", i));
        if(frame.empty())   break;

        finder->find(frame);

        cv::Rect extended = getExtendedRect(frame, finder->faces[0]);
        landmarksFinder->find(frame, extended);
        cv::Mat allignedFace = frontalizeFace3D(frame(finder->faces[0]), finder->faces[0], landmarksFinder->landmarks);
        finder->find(allignedFace);
        faces.push_back(allignedFace(finder->faces[0]).clone());
        labels.push_back(index);

        if (i % 10 == 9){
            ++index;
        }
        ++i;
    }

    recognizer->train(faces, labels, recognizerPath);
}


void FaceRecognitionWithAllignment2DTrainOnPhotosTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& faceFinderPath, std::string& recognizerPath, std::string& landmarksPath)
{
    finder->read(faceFinderPath);
    landmarksFinder->read(landmarksPath);

    int i = 0, index = 0;

    std::vector<int> labels;
    std::vector<cv::Mat> faces;

    while (true)
    {
        cv::Mat frame = cv::imread(cv::format("dataframes/regognition/ideal/img_%d.png", i));
        if(frame.empty())   break;

        finder->find(frame);

        cv::Rect extended = getExtendedRect(frame, finder->faces[0]);
        landmarksFinder->find(frame, extended);
        cv::Mat allignedFace = frontalizeFace2D(frame, landmarksFinder->landmarks);

        finder->find(allignedFace);

        faces.push_back(allignedFace(getNormalRect(allignedFace, finder->faces[0])).clone());
        labels.push_back(index);

        if (i % 10 == 9){
            ++index;
        }
        ++i;
    }

    recognizer->train(faces, labels, recognizerPath);
}


void FaceReconstractionTest(cv::Ptr<BaseFaceFinder> finder, FaceReconstraction reconstraction, std::string& finderPath, std::string& landmarks3DPath, std::string& depthPath)
{
    finder->read(finderPath);
    reconstraction.read(landmarks3DPath, depthPath);

    cv::VideoCapture cap(0);
    cv::Mat frame;

    while (true)
    {
        cap.read(frame);
        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            reconstraction.getPoints(frame, finder->faces[i]);
            draw3DPoints(frame, reconstraction.points);
        }

        cv::imshow("FaceReconstractionTest", frame);
        if(cv::waitKey(1) == 'q'){
            break;
        }
    }
}





double FaceDetectorTimeTest(cv::Ptr<BaseFaceFinder> finder, std::string& path)
{
    finder->read(path);

    cv::VideoCapture cap(0);
    cv::Mat frame;

    cv::Rect face;

    Timer timer;
    double res;
    std::vector<double> times;

    while (true)
    {
        cap.read(frame);


        timer.start();
        finder->find(frame);
        timer.stop();

        if(finder->faces.size() != 0){
            times.push_back(timer.getElapsedTime());
        }

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            std::string label = cv::format("conf: %.2f", finder->confidences[i]);
            face = getNormalRect(frame, finder->faces[i]);
            drawRectWithLable(frame, face, label);
        }

        cv::imshow("FaceDetectorTest", frame);
        if(cv::waitKey(1) == 'q'){
            break;
        }
    }

    res = std::accumulate(times.begin(), times.end(), 0.0);
    res /= times.size();
    return res;
}


double FaceRecognitionTimeTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, std::string& finderPath, std::string& recognizerPath)
{
    finder->read(finderPath);
    recognizer->read(recognizerPath);
    cv::Mat frame;
    cv::VideoCapture cap(0);

    std::string label;
    std::pair<int, float> predictRes;

    Timer timer;
    double res;
    std::vector<double> times;

    while (true)
    {
        cap.read(frame);

        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            cv::Mat roi = frame(finder->faces[i]).clone();

            timer.start();
            predictRes = recognizer->predict(roi);
            timer.stop();

            times.push_back(timer.getElapsedTime());

            label = cv::format("Person %d with %.2f", predictRes.first, predictRes.second);
            drawRectWithLable(frame, finder->faces[i], label);
        }

        cv::imshow("FaceRecognitionPredictTest", frame);
        if (cv::waitKey(1) == 'q'){
            break;
        }
    }

    res = std::accumulate(times.begin(), times.end(), 0.0);
    res /= times.size();
    return res;
}


double LandmarksTimeTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& finderPath, std::string& landmarksPath)
{
    finder->read(finderPath);
    landmarksFinder->read(landmarksPath);

    cv::VideoCapture cap(0);
    cv::Mat frame;

    Timer timer;
    double res;
    std::vector<double> times;

    while (true)
    {
        cap.read(frame);
        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            cv::Rect extended = getExtendedRect(frame, finder->faces[i]);
            drawRectWithLable(frame, extended, "");

            timer.start();
            landmarksFinder->find(frame, extended);
            timer.stop();

            times.push_back(timer.getElapsedTime());

            for (size_t j = 0; j < landmarksFinder->landmarks.size(); ++j){
                cv::circle(frame, landmarksFinder->landmarks[j], 2, {255, 255, 255}, 2);
            }
        }

        cv::imshow("LandmarksTest", frame);
        if(cv::waitKey(1) == 'q'){
            break;
        }
    }

    res = std::accumulate(times.begin(), times.end(), 0.0);
    res /= times.size();
    return res;
}




void addFrames(int nextI)
{
    cv::VideoCapture cap(0);
    cv::Mat frame;

    char key;
    bool loop = true;

    while (loop)
    {
        cap.read(frame);


        cv::imshow("FaceDetectorTest", frame);
        key = cv::waitKey(1);

        switch (key)
        {
        case 'q':
            loop = false;
            break;
        case 'a':
            cv::imwrite(cv::format("dataframes/textData/img_%d.png", nextI), frame);
            ++nextI;
            break;
        }
    }
}


std::vector<int> FaceDetectorAccuracyTest(cv::Ptr<BaseFaceFinder> finder, std::string& path)
{
    finder->read(path);

    cv::Mat frame;

    cv::Rect face;
    std::vector<int> data(3, 0);
    int i = 0;
    char key;

    while (true)
    {
        frame = cv::imread(cv::format("dataframes/dark/img_%d.png", i));
        if(frame.empty())   break;

        finder->find(frame);

        for (size_t k = 0; k < finder->faces.size(); ++k)
        {
            std::string label = cv::format("conf: %.2f", finder->confidences[k]);
            face = getNormalRect(frame, finder->faces[k]);
            drawRectWithLable(frame, face, label);
        }

        cv::imshow("FaceDetectorTest", frame);

        for (size_t j = 0; j < data.size(); ++j)
        {
            key = cv::waitKey();
            data[j] += (key - '0');
        }

        ++i;

        if(key == 'q'){
            break;
        }
    }

    std::cout << cv::format("cor: %d\nlos: %d\ninc: %d", data[0], data[1], data[2]);
    return data;
}


void FaceRecognitionDataCollector(cv::Ptr<BaseFaceFinder> finder, std::string& path)
{
    finder->read(path);

    cv::Mat frame;
    int i = 0;

    char key;

    cv::namedWindow("FaceRecognitionDataCollector");

    //std::string filename = "dataframes/regognition/recodnitiondata.txt";
    //std::fstream file(filename, std::fstream::in | std::fstream::out | std::fstream::app);

    while (true)
    {
        frame = cv::imread(cv::format("dataframes/regognition/data/img_%d.png", i));
        if(frame.empty())   break;
        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            cv::Rect extended = getExtendedRect(frame, finder->faces[i]);
            drawRectWithLable(frame, extended, "");
        }

        cv::imshow("LandmarksDataCollector", frame);

        key = cv::waitKey();

        if(key == 'q'){
            break;
        }

        //file << cv::format("%d\n", key - '0');
        ++i;
    }
}


std::vector<std::vector<int>> FaceRecognitionAccuracyTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, std::string& finderPath, std::string& recognizerPath, double threshold)
{
    finder->read(finderPath);
    recognizer->read(recognizerPath);

    cv::Mat frame;

    std::string label;
    std::pair<int, float> predictRes;

    std::string filename = "dataframes/regognition/recodnitiondata.txt";
    std::ifstream file(filename);
    std::string line;

    std::vector<std::vector<int>> data(4, std::vector<int>(4, 0));
    int i = 0;
    bool loop = true;

    while (loop)
    {
        if (i == 250)   break;
        frame = cv::imread(cv::format("dataframes/regognition/data/img_%d.png", i));
        if(frame.empty())   break;

        finder->find(frame);

        cv::Mat roi = frame(getNormalRect(frame, finder->faces[0])).clone();
        predictRes = recognizer->predict(roi);
        (predictRes.second > threshold) ? predictRes.first = 3 : 0;

        std::getline(file, line);

        data[predictRes.first][std::stoi(line)] += 1;

        ++i;
    }

    for (size_t j = 0; j < data.size(); ++j)
    {
        for (size_t k = 0; k < data[j].size(); ++k){
            std::cout << data[j][k] << '\t';
        }
        std::cout << '\n';
    }

    return data;
}


std::vector<std::vector<int>> FaceRecognitionAccuracyAlligment3DTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& faceFinderPath, std::string& recognizerPath, std::string& landmarksPath, double threshold)
{
    finder->read(faceFinderPath);
    recognizer->read(recognizerPath);
    landmarksFinder->read(landmarksPath);

    cv::Mat frame;

    std::string label;
    std::pair<int, float> predictRes;

    std::string filename = "dataframes/regognition/recodnitiondata.txt";
    std::ifstream file(filename);
    std::string line;

    std::vector<std::vector<int>> data(4, std::vector<int>(4, 0));
    int i = 0;
    bool loop = true;

    while (loop)
    {
        if (i == 333)   break;
        frame = cv::imread(cv::format("dataframes/regognition/data/img_%d.png", i));
        if(frame.empty())   break;

        finder->find(frame);
        cv::Rect roi = finder->faces[0];
        landmarksFinder->find(frame, getExtendedRect(frame, roi));
        cv::Mat allidnedFace = frontalizeFace3D(frame(getNormalRect(frame, roi)).clone(), roi, landmarksFinder->landmarks);
        finder->find(allidnedFace);
        (finder->faces.size() == 0) ? (roi = cv::Rect(0, 0, allidnedFace.cols, allidnedFace.rows)) : roi = getNormalRect(allidnedFace, finder->faces[0]);
        predictRes = recognizer->predict(allidnedFace(roi).clone());
        (predictRes.second < threshold) ? predictRes.first = 3 : 0;
        std::getline(file, line);
        data[predictRes.first][std::stoi(line)] += 1;

        ++i;
    }

    for (size_t j = 0; j < data.size(); ++j)
    {
        for (size_t k = 0; k < data[j].size(); ++k){
            std::cout << data[j][k] << '\t';
        }
        std::cout << '\n';
    }

    return data;
}


std::vector<std::vector<int>> FaceRecognitionAccuracyAlligment2DTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& faceFinderPath, std::string& recognizerPath, std::string& landmarksPath, double threshold)
{
    finder->read(faceFinderPath);
    recognizer->read(recognizerPath);
    landmarksFinder->read(landmarksPath);

    cv::Mat frame;

    std::string label;
    std::pair<int, float> predictRes;

    std::string filename = "dataframes/regognition/recodnitiondata.txt";
    std::ifstream file(filename);
    std::string line;

    std::vector<std::vector<int>> data(4, std::vector<int>(4, 0));
    int i = 0;
    bool loop = true;

    while (loop)
    {
        if (i == 250)   break;
        frame = cv::imread(cv::format("dataframes/regognition/data/img_%d.png", i));
        if(frame.empty())   break;

        finder->find(frame);

        cv::Rect roi = finder->faces[0];
        landmarksFinder->find(frame, getExtendedRect(frame, roi));
        cv::Mat allidnedFace = frontalizeFace2D(frame, landmarksFinder->landmarks);
        finder->find(allidnedFace);
        allidnedFace = (finder->faces.empty()) ? allidnedFace : allidnedFace(getNormalRect(allidnedFace, finder->faces[0])).clone();
        predictRes = recognizer->predict(allidnedFace);
        (predictRes.second > threshold) ? predictRes.first = 3 : 0;

        std::getline(file, line);

        data[predictRes.first][std::stoi(line)] += 1;

        ++i;
    }

    for (size_t j = 0; j < data.size(); ++j)
    {
        for (size_t k = 0; k < data[j].size(); ++k){
            std::cout << data[j][k] << '\t';
        }
        std::cout << '\n';
    }

    return data;
}


void getDistances(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, std::string& finderPath, std::string& recognizerPath)
{
    finder->read(finderPath);
    recognizer->read(recognizerPath);

    cv::Mat frame;

    std::pair<int, float> predictRes;

    int i = 0;
    bool loop = true;

    double res = 0;

    while (loop)
    {
        frame = cv::imread(cv::format("dataframes/regognition/ideal/img_%d.png", i));
        if(frame.empty())   break;

        finder->find(frame);

        for (size_t j = 0; j < finder->faces.size(); ++j)
        {
            cv::Mat roi = frame(finder->faces[j]).clone();
            predictRes = recognizer->predict(roi);
            res += predictRes.second;
        }
        ++i;

        if(i % 10 == 0)
        {
            std::cout << res / 10 << '\n';
            res = 0;
        }
    }
}


double getThresholdValue(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, std::string& finderPath, std::string& recognizerPath)
{
    finder->read(finderPath);
    recognizer->read(recognizerPath);

    cv::Mat frame;

    std::string label;
    std::pair<int, float> predictRes;

    int best = 0, temp = 0;

    for (double threshold = 700; threshold <= 1500; threshold += 50)
    {
        std::string filename = "dataframes/regognition/recodnitiondata.txt";
        std::ifstream file(filename);
        std::string line;

        std::vector<std::vector<int>> data(4, std::vector<int>(4, 0));
        int i = 0;
        bool loop = true;

        while (loop)
        {
            if (i == 280) break;
            frame = cv::imread(cv::format("dataframes/regognition/data/img_%d.png", i));

            finder->find(frame);

            cv::Mat roi = frame(getNormalRect(frame, finder->faces[0])).clone();
            predictRes = recognizer->predict(roi);
            (predictRes.second > static_cast<float>(threshold)) ? predictRes.first = 3 : 0;

            std::getline(file, line);

            data[predictRes.first][std::stoi(line)] += 1;

            ++i;
        }

        temp = 0;

        for (int j = 0; j < 3; ++j){
            temp += data[j][j];
        }

        if (temp > best){
            best = temp;
        }

        std::cout << cv::format("%f\t%d\t%d\n", threshold, temp, data[3][3]);
    }

    std::cout << "\n\n\n" << best << "\n\n\n";
    return best;
}


double getThresholdValue3DAlligment(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& faceFinderPath, std::string& recognizerPath, std::string& landmarksPath)
{
    finder->read(faceFinderPath);
    recognizer->read(recognizerPath);
    landmarksFinder->read(landmarksPath);

    cv::Mat frame;

    std::string label;
    std::pair<int, float> predictRes;

    int best = 0, temp = 0;

    for (double threshold = 0.1; threshold <= 0.25; threshold += 0.01)
    {
        std::string filename = "dataframes/regognition/recodnitiondata.txt";
        std::ifstream file(filename);
        std::string line;

        std::vector<std::vector<int>> data(4, std::vector<int>(4, 0));
        int i = 0;
        bool loop = true;

        while (loop)
        {
            if(i == 333)   break;
            frame = cv::imread(cv::format("dataframes/regognition/data/img_%d.png", i));

            finder->find(frame);
            cv::Rect roi = finder->faces[0];
            landmarksFinder->find(frame, getExtendedRect(frame, roi));
            cv::Mat allidnedFace = frontalizeFace3D(frame(getNormalRect(frame, roi)).clone(), roi, landmarksFinder->landmarks);
            finder->find(allidnedFace);
            (finder->faces.size() == 0) ? (roi = cv::Rect(0, 0, allidnedFace.cols, allidnedFace.rows)) : roi = getNormalRect(allidnedFace, finder->faces[0]);
            predictRes = recognizer->predict(allidnedFace(roi).clone());
            (predictRes.second < threshold) ? predictRes.first = 3 : 0;
            std::getline(file, line);
            data[predictRes.first][std::stoi(line)] += 1;

            ++i;
        }

        temp = 0;

        for (int j = 0; j < 3; ++j){
            temp += data[j][j];
        }

        if (temp > best){
            best = temp;
        }

        std::cout << cv::format("%f\t%d\t%d\n", threshold, temp, data[3][3]);
    }

    std::cout << "\n\n\n" << best << "\n\n\n";
    return best;
}


double getThresholdValue2DAlligment(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseFaceRecognizer> recognizer, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& faceFinderPath, std::string& recognizerPath, std::string& landmarksPath)
{
    finder->read(faceFinderPath);
    recognizer->read(recognizerPath);
    landmarksFinder->read(landmarksPath);

    cv::Mat frame;

    std::string label;
    std::pair<int, float> predictRes;

    int best = 0, temp = 0;

    for (double threshold = 2000; threshold <= 4000; threshold += 50)
    {
        std::string filename = "dataframes/regognition/recodnitiondata.txt";
        std::ifstream file(filename);
        std::string line;

        std::vector<std::vector<int>> data(4, std::vector<int>(4, 0));
        int i = 0;
        bool loop = true;

        while (loop)
        {
            if(i == 250)   break;
            frame = cv::imread(cv::format("dataframes/regognition/data/img_%d.png", i));

            finder->find(frame);
            cv::Rect roi = finder->faces[0];
            landmarksFinder->find(frame, getExtendedRect(frame, roi));
            cv::Mat allidnedFace = frontalizeFace2D(frame, landmarksFinder->landmarks);

            finder->find(allidnedFace);
            allidnedFace = (finder->faces.empty()) ? allidnedFace : allidnedFace(getNormalRect(allidnedFace, finder->faces[0])).clone();

            predictRes = recognizer->predict(allidnedFace.clone());
            (predictRes.second > threshold) ? predictRes.first = 3 : 0;
            std::getline(file, line);
            data[predictRes.first][std::stoi(line)] += 1;

            ++i;
        }

        temp = 0;

        for (int j = 0; j < 3; ++j){
            temp += data[j][j];
        }

        if (temp > best){
            best = temp;
        }

        std::cout << cv::format("%f\t%d\t%d\n", threshold, temp, data[3][3]);
    }

    std::cout << "\n\n\n" << best << "\n\n\n";
    return best;
}

struct ClickData
{
    int x = 0;
    int y = 0;
    bool clicked = false;
};

void onMouse(int event, int x, int y, int /*flags*/, void* userdata)
{
    auto data = reinterpret_cast<ClickData*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        data->x = x;
        data->y = y;
        data->clicked = true;
    }
}

void LandmarksDataCollector(cv::Ptr<BaseFaceFinder> finder, std::string& finderPath)
{
    finder->read(finderPath);

    cv::Mat frame;
    int i = 0;

    cv::namedWindow("LandmarksDataCollector");

    std::string filename = "dataframes/landmarksdata/landmarksdata.txt";
    std::fstream file(filename, std::fstream::in | std::fstream::out | std::fstream::app);

    ClickData data;

    cv::setMouseCallback("LandmarksDataCollector", onMouse, &data);

    while (true)
    {
        frame = cv::imread(cv::format("dataframes/landmarksdata/landmarksframes/img_%d.png", i));
        if(frame.empty())   break;
        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            cv::Rect extended = getExtendedRect(frame, finder->faces[i]);
            drawRectWithLable(frame, extended, "");
        }

        cv::imshow("LandmarksDataCollector", frame);

        int counter = 0;

        while (counter < 3)
        {
            if(cv::waitKey(1) == 'q'){
                break;
            }
            if(!data.clicked) continue;

            if(counter == 0)    file << cv::format("img_%d:\n", i);
            file << data.x << '\t' << data.y << '\n';
            data.clicked = false;
            ++counter;
        }

        if(cv::waitKey(1) == 'q'){
            break;
        }
        ++i;
    }
}


double LandmarksAccuracyTest(cv::Ptr<BaseFaceFinder> finder, cv::Ptr<BaseLandmarksFinder> landmarksFinder, std::string& finderPath, std::string& landmarksPath)
{
    finder->read(finderPath);
    landmarksFinder->read(landmarksPath);

    cv::Mat frame;
    int i = 0;

    std::string filename = "dataframes/landmarksdata/landmarksdata.txt";
    std::ifstream file(filename);
    std::string line;

    double result = 0;

    while (true)
    {
        frame = cv::imread(cv::format("dataframes/landmarksdata/landmarksframes/img_%d.png", i));
        if(frame.empty())   break;
        finder->find(frame);

        for (size_t i = 0; i < finder->faces.size(); ++i)
        {
            cv::Rect extended = getExtendedRect(frame, finder->faces[i]);
            landmarksFinder->find(frame, extended);
        }

        cv::Point2i leftEye = {0, 0}, rightEye = {0, 0}, mounse = {0, 0};

        for(int j = 36; j <= 41; ++j)
        {
            leftEye.x += landmarksFinder->landmarks[j].x;
            leftEye.y += landmarksFinder->landmarks[j].y;
        }

        leftEye /= 6;

        //cv::circle(frame, leftEye, 2, {255, 0, 0});

        for(int j = 42; j <= 47; ++j)
        {
            rightEye.x += landmarksFinder->landmarks[j].x;
            rightEye.y += landmarksFinder->landmarks[j].y;
        }

        rightEye /= 6;

        //cv::circle(frame, rightEye, 2, {0, 255, 0});

        for(int j = 48; j <= 59; ++j)
        {
            mounse.x += landmarksFinder->landmarks[j].x;
            mounse.y += landmarksFinder->landmarks[j].y;
            //cv::circle(frame, landmarksFinder->landmarks[j], 2, {255, 0, 0});
        }

        mounse /= 12;

        //cv::circle(frame, mounse, 2, {0, 0, 255});

        for (size_t j = 0; j < 4; j++)
        {
            std::getline(file, line);
            if(j == 0)  continue;
            std::stringstream stream(line);
            cv::Point2i point;
            stream >> point.x >> point.y;

            switch (j)
            {
            case 1:
            point -= leftEye;
                break;
            case 2:
            point -= rightEye;
                break;
            case 3:
            point -= mounse;
                break;
            }
            result += sqrt(point.x * point.x + point.y * point.y);
        }

        if(cv::waitKey(1) == 'q'){
            break;
        }
        ++i;
    }
    return result;
}


void joinFrames()
{
    cv::Ptr<BaseFaceFinder> finder = cv::makePtr<DNNFaceFinder>();
    std::string finderPath = "models/faceDetectors/dnn/yolov11n-face.onnx";
    finder->read(finderPath);

    std::string dnnWeightsPath = "models/recognizers/dnn/arcface.onnx";
    cv::Ptr<BaseFaceRecognizer> recognizer = cv::makePtr<DNNRecognizer>(dnnWeightsPath);
    std::string recognizerPath = "models/recognizers/dnn/DNNFaceRecognizer.xml";
    recognizer->read(recognizerPath);

    cv::Ptr<BaseLandmarksFinder> landmarksFinder = cv::makePtr<DNNLandmarksFinder>();
    std::string landmarksPath = "models/landmarks/2D/dnn/face_alignment.onnx";
    landmarksFinder->read(landmarksPath);

    std::vector<cv::Mat> imgs =
        {
            cv::imread("dataframes/withfon/img_69.png")
        };

    for (size_t i = 0; i < imgs.size(); ++i)
    {
        finder->find(imgs[i]);
        for (size_t j = 0; j < finder->faces.size(); ++j)
        {
            cv::rectangle(imgs[i], finder->faces[j], cv::Scalar(0, 255, 0), 5);
        }
    }

    // finder->find(imgs[0]);
    //
    // cv::Mat resultImg = imgs[0].clone();
    //
    // finder->find(imgs[0]);
    // cv::Rect extended = getExtendedRect(imgs[0], finder->faces[0]);
    // landmarksFinder->find(imgs[0], extended);
    // cv::Mat allidnedFace = frontalizeFace2D(imgs[0], landmarksFinder->landmarks);
    //
    // for (size_t j = 0; j < landmarksFinder->landmarks.size(); ++j){
    //     cv::circle(imgs[0], landmarksFinder->landmarks[j], 2, {0, 255, 0}, 5);
    // }
    //
    // std::pair<int, float> predictRes = recognizer->predict(allidnedFace);
    // std::string label = cv::format("ID: %d, distance: %.2f", predictRes.first, predictRes.second);
    // drawRectWithLable(resultImg, finder->faces[0], label);
    //
    // cv::resize(allidnedFace, allidnedFace, cv::Size(allidnedFace.cols * (static_cast<double>(imgs[0].rows) / allidnedFace.rows), imgs[0].rows));
    // imgs.push_back(allidnedFace);
    // imgs.push_back(resultImg);

    cv::Mat res;
    cv::hconcat(imgs, res);
    cv::imwrite("dataframes/textData/img_1234post.png", res);
}


int main()
{
    cv::Ptr<BaseFaceFinder> finder = cv::makePtr<DNNFaceFinder>();
    std::string finderPath = "models/faceDetectors/dnn/yolov11n-face.onnx";

    //FaceDetectorTest(finder, finderPath);
    //std::cout << '\n' << FaceDetectorTimeTest(finder, finderPath);

    std::string dnnWeightsPath = "models/recognizers/dnn/arcface.onnx";
    cv::Ptr<BaseFaceRecognizer> recognizer = cv::makePtr<DNNRecognizer>(dnnWeightsPath);
    std::string recognizerPath = "models/recognizers/dnn/DNNFaceRecognizer.xml";

    //FaceRecognitionTrainTest(finder, recognizer, finderPath, recognizerPath);
    //FaceRecognitionTrainOnPhotosTest(finder, recognizer, finderPath, recognizerPath);
    //FaceRecognitionPredictTest(finder, recognizer, finderPath, recognizerPath);


    //std::cout << '\n' << FaceRecognitionTimeTest(finder, recognizer, finderPath, recognizerPath);


    cv::Ptr<BaseLandmarksFinder> landmarksFinder = cv::makePtr<LBFLandmarksFinder>();
    std::string landmarksPath = "models/landmarks/2D/classic/lbfmodel_landmark.yaml";

    //LandmarksTest(finder, landmarksFinder, finderPath, landmarksPath);
    //Allignment2DTest(finder, landmarksFinder, finderPath, landmarksPath);
    //Allignment3DTest(finder, landmarksFinder, finderPath, landmarksPath);


    //FaceRecognitionWithAllignment2DTrainTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);
    //FaceRecognitionWithAllignment2DPredictTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);

    //FaceRecognitionWithAllignment3DTrainTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);
    //FaceRecognitionWithAllignment3DPredictTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);


    //FaceRecognitionWithAllignment3DTrainOnPhotosTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);
    //FaceRecognitionWithAllignment2DTrainOnPhotosTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);

    //std::cout << '\n' << LandmarksTimeTest(finder, landmarksFinder, finderPath, landmarksPath);


    // FaceReconstraction reconstraction;
    // std::string landmarks3DPath = "models/landmarks/3D/face-alignment3D.onnx";
    // std::string depthPath = "models/landmarks/3D/depthLandmark.onnx";
    //FaceReconstractionTest(finder, reconstraction, finderPath, landmarks3DPath, depthPath);



    //addFrames(6);
    //FaceDetectorAccuracyTest(finder, finderPath);
    //LandmarksDataCollector(finder, finderPath);
    //std::cout << "\n\n" << LandmarksAccuracyTest(finder, landmarksFinder, finderPath, landmarksPath) << "\n\n";
    //getDistances(finder, recognizer, finderPath, recognizerPath);
    //FaceRecognitionAccuracyTest(finder, recognizer, finderPath, recognizerPath, INFINITY);
    //FaceRecognitionAccuracyAlligment3DTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath, 0.20675585);
    //FaceRecognitionAccuracyAlligment2DTest(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath, 3290.97);
    //FaceRecognitionDataCollector(finder, finderPath);
    //getThresholdValue(finder, recognizer, finderPath, recognizerPath);
    //getThresholdValue3DAlligment(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);
    //getThresholdValue2DAlligment(finder, recognizer, landmarksFinder, finderPath, recognizerPath, landmarksPath);
    joinFrames();

    return 0;
}
