#include "FaceReconstraction.hpp"


void FaceReconstraction::read(std::string landmarksPath, std::string depthPath)
{
    this->_landmarksNet = cv::dnn::readNetFromONNX(landmarksPath);
    this->_depthNet = cv::dnn::readNetFromONNX(depthPath);
}



void FaceReconstraction::getPoints(const cv::Mat &frame, cv::Rect face)
{
    this->points.clear();

    face = this->_getExtendedRect(frame, face);
    // 1. Вырезаем область лица из исходного кадра и изменяем размер до 256x256
    cv::Mat faceROI = frame(face).clone();               // вырезаем прямоугольную область лица
    cv::Mat resized;
    cv::resize(faceROI, resized, cv::Size(256, 256));    // изменяем размер до 256x256

    // 2. Предобработка: создаём blob из изображения
    // Масштабируем пиксели к [0,1] (1/255), не вычитаем среднее (mean={0,0,0}),
    // меняем порядок с BGR на RGB (swapRB=true), без обрезки (crop=false).
    cv::Mat inputBlob = cv::dnn::blobFromImage(
        resized,                                     // входное изображение
        1.0 / 255.0,                                 // масштабирование интенсивности пикселей
        cv::Size(256, 256),                          // целевой размер
        cv::Scalar(0, 0, 0),                         // смещение (mean subtract), здесь без вычитания среднего
        true, /*swapRB*/ false /*crop*/              // меняем BGR->RGB, не обрезаем изображение
    );

    // 3. Передаём подготовленный blob в первую модель (поиск 2D ключевых точек лица)
    this->_landmarksNet.setInput(inputBlob);
    cv::Mat landmarksBlob = this->_landmarksNet.forward();  // получаем выход модели ключевых точек

    // 4. Обработка выхода первой модели.
    // Предположим, что первая модель возвращает N карт признаков/heatmap (например, N=68 каналов для 68 ключевых точек).
    // Комбинируем эти карты с оригинальным изображением для подачи во вторую модель.
    // Разделяем выходной blob на отдельные каналы (heatmap для каждой ключевой точки).
    std::vector<cv::Mat> landmarkChannels;
    int numLandmarks = landmarksBlob.size[1];           // число каналов (ключевых точек)
    int h = landmarksBlob.size[2];                      // высота карты (ожидается 256)
    int w = landmarksBlob.size[3];                      // ширина карты (ожидается 256)
    // Преобразуем выходной blob в набор 2D матриц (одна матрица на канал)
    for (int i = 0; i < numLandmarks; ++i) {
        // Извлекаем i-й канал в матрицу размером h x w
        cv::Mat heatmap(h, w, CV_32F, landmarksBlob.ptr(0, i));
        cv::resize(heatmap, heatmap, {256, 256});
        landmarkChannels.push_back(heatmap.clone());
    }

    // Также разделяем исходный blob изображения на 3 канала (RGB)
    std::vector<cv::Mat> imageChannels;
    cv::Mat faceImageNorm; 
    // Преобразуем исходный blob обратно в нормализованное изображение (1x3x256x256 -> 3 каналa 256x256)
    cv::dnn::imagesFromBlob(inputBlob, imageChannels);
    // Функция imagesFromBlob вернёт вектор изображений; для одного изображения он будет содержать 1 Mat 256x256 с 3 каналами.
    // Нам нужно разделить этот Mat на отдельные каналы.
    if (!imageChannels.empty()) {
        cv::Mat faceImage = imageChannels[0];           // получаем изображение лица (256x256, 3 канала, в диапазоне [0,1])
        cv::split(faceImage, imageChannels);           // разделяем на 3 одноканальных матрицы (R, G, B)
    }

    // Объединяем 3 канала изображения и N каналов heatmap в один входной blob для модели глубины
    std::vector<cv::Mat> allChannels;
    allChannels.insert(allChannels.end(), imageChannels.begin(), imageChannels.end());
    allChannels.insert(allChannels.end(), landmarkChannels.begin(), landmarkChannels.end());

    int size[4] = {1, 71, 256, 256};
    cv::Mat depthInputBlob(4, size, CV_32FC1);
    for (size_t i = 0; i < allChannels.size(); ++i){
        depthInputBlob.at<cv::Mat>(0, i) = allChannels[i].clone();
    }

    // 5. Передаём сформированный blob во вторую модель (оценка глубины)
    this->_depthNet.setInput(depthInputBlob);
    cv::Mat depthBlob = this->_depthNet.forward();        // выход модели глубины (например, карта глубины лица или значения глубины точек)

    // 6. Комбинируем результаты для получения 3D-координат.
    std::vector<cv::Point3f> keypoints3D;
    keypoints3D.reserve(numLandmarks);
    // Получаем 2D координаты каждой ключевой точки по данным первой модели.
    // Предположим, что для каждой heatmap мы находим координаты максимума – это положение ключевой точки.
    for (int i = 0; i < numLandmarks; ++i) 
    {
        // Находим координаты максимального значения на heatmap i-й точки
        double maxVal;
        cv::Point maxLoc;
        cv::minMaxLoc(landmarkChannels[i], nullptr, &maxVal, nullptr, &maxLoc);
        int px = maxLoc.x;
        int py = maxLoc.y;
        // Получаем соответствующее значение глубины.
        float depthValue;

        depthValue = depthBlob.at<float>(0, i);

        // Преобразуем координаты обратно в координаты исходного изображения (если нужно)
        float origX = face.x + px * (face.width  / 256.0f);
        float origY = face.y + py * (face.height / 256.0f);
        // Добавляем 3D-точку: (x, y, z)
        keypoints3D.emplace_back(origX, origY, depthValue);
    }
    this->points = keypoints3D;
}

cv::Rect FaceReconstraction::_getExtendedRect(const cv::Mat &frame, const cv::Rect &rect)
{
    cv::Rect res = rect;

    (res.x > 50) ? res.x -= 50 : res.x = 0;
    (res.y > 50) ? res.y -= 50 : res.y = 0;
    (res.x + res.width + 100 > frame.cols) ? res.width = (frame.cols - (res.x + 1)) : res.width += 100;
    (res.y + res.height + 100 > frame.rows) ? res.height = (frame.rows - (res.y + 1)) : res.height += 100;

    return res;
}
