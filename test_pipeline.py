import cv2
import numpy as np
import json
import os
from models.calibrator import Calibrator
from models.analyzer import Analyzer

def create_mock_image(path="dataset/test_image.jpg"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    # Рисуем "парковку"
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), 2)
    cv2.putText(img, "A1", (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    cv2.rectangle(img, (200, 50), (300, 150), (255, 255, 255), 2)
    cv2.putText(img, "A2", (240, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Рисуем "машину" (допустим она стоит в A1)
    cv2.rectangle(img, (60, 60), (140, 140), (0, 0, 255), -1) 
    
    cv2.imwrite(path, img)
    return path

def run_mock_pipeline():
    # 1. Загрузка или создание тестовых данных
    img_path = create_mock_image()
    
    if not os.path.exists("config.json"):
        print("Error: config.json not found for testing.")
        return

    with open("config.json", 'r') as f:
        config = json.load(f)

    # 2. Инициализация
    calibrator = Calibrator(
        src_points=config["calibration_points"], 
        target_size=(config["target_width"], config["target_height"])
    )
    analyzer = Analyzer(config)
    
    # 3. Мок детекции (представим, что YOLO нашла машину в (60,60)-(140,140))
    # Это координаты на "искаженном кадре", но в нашем тесте кадр плоский 
    # для простоты проверки самой геометрии
    mock_detections = [
        {"bbox": [60, 60, 140, 140], "confidence": 0.95, "class": 2}
    ]

    transformed_points = []
    for det in mock_detections:
        point = calibrator.transform_bbox_bottom_center(det["bbox"])
        transformed_points.append({
            "point": point,
            "confidence": det["confidence"]
        })

    print("Transformed points:", transformed_points)
    events = analyzer.analyze(transformed_points)
    print("\nEvents:")
    print(json.dumps(events, indent=2))

if __name__ == "__main__":
    print("Mock Pipeline Test (Geometry and Intersections only):")
    run_mock_pipeline()
    print("\nTo run the full YOLO pipeline, execute:")
    print("python main.py --image dataset/test_image.jpg")
