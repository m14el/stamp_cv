from ultralytics import YOLO

class Detector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Инициализация детектора.
        Используется легковесная модель YOLOv8 по умолчанию для скорости.
        """
        self.model = YOLO(model_name)
        # В COCO датасете индексы классов: 2=car, 3=motorcycle, 5=bus, 7=truck. 
        # Будем детектировать их все как "занято".
        self.target_classes = [2, 3, 5, 7] 

    def detect(self, frame):
        """
        Поиск автомобилей на кадре
        :param frame: Изображение OpenCV
        :return: Список словарей с bounding boxes и confidence
        """
        results = self.model(frame, classes=self.target_classes, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": cls
                })
        return detections
