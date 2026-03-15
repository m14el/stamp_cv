import cv2
import numpy as np

class Calibrator:
    def __init__(self, src_points, target_size=(400, 400)):
        """
        Инициализация модуля калибровки.
        :param src_points: Список из 4-х точек на исходном кадре [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        :param target_size: (width, height) результирующей "плоской" проекции парковки
        """
        self.src_points = np.float32(src_points)
        self.target_w, self.target_h = target_size
        
        # Точки назначения (прямоугольник, представляющий плоскую зону)
        self.dst_points = np.float32([
            [0, 0],
            [self.target_w, 0],
            [self.target_w, self.target_h],
            [0, self.target_h]
        ])
        
        # Расчет матрицы гомографии
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def wrap_image(self, frame):
        """
        Преобразует кадр перспективы (bird's-eye view).
        :param frame: Исходный кадр OpenCV
        :return: Преобразованный кадр
        """
        return cv2.warpPerspective(frame, self.matrix, (self.target_w, self.target_h))

    def transform_point(self, point):
        """
        Преобразует точку с исходного кадра в координаты плоской карты.
        :param point: Кортеж или список (x, y)
        :return: (x', y')
        """
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed_pt = cv2.perspectiveTransform(pt, self.matrix)
        return int(transformed_pt[0][0][0]), int(transformed_pt[0][0][1])

    def transform_bbox_bottom_center(self, bbox):
        """
        Берет нижний центр bbox (это лучшая точка для проекции автомобиля на асфальт).
        :param bbox: [x1, y1, x2, y2]
        :return: пребразованная точка (x, y) на плоскости
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        bottom_y = y2 # Нижний край bbox, где машина стоит на земле
        return self.transform_point((center_x, bottom_y))
