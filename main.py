import argparse
import json
import cv2
import os
from models.calibrator import Calibrator
from models.detector import Detector
from models.analyzer import Analyzer

def main():
    parser = argparse.ArgumentParser(description="ParkCloud CV Module")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image frame")
    args = parser.parse_args()

    # Загрузка конфигурации
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Инициализация модулей
    calibrator = Calibrator(
        src_points=config["calibration_points"], 
        target_size=(config["target_width"], config["target_height"])
    )
    
    # Ленивая загрузка YOLO чтобы не ждать при тестах если нет изображения
    detector = Detector()
    analyzer = Analyzer(config)

    # Чтение кадра
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Cannot load image {args.image}")
        return

    # Шаг 1: Детекция на оригинальном кадре
    print("Running detection...")
    detections = detector.detect(frame)
    
    # Шаг 2: Проброс координат на плоскую карту парковки
    print("Projecting coordinates to bird's-eye view...")
    transformed_points = []
    for det in detections:
        point = calibrator.transform_bbox_bottom_center(det["bbox"])
        transformed_points.append({
            "point": point,
            "confidence": det["confidence"]
        })
        
    # Шаг 3: Анализ занятости
    print("Analyzing occupancy...")
    events = analyzer.analyze(transformed_points)

    # Шаг 4: Вывод результата (События для облака)
    print("\n--- Event Output ---")
    print(json.dumps(events, indent=2))

if __name__ == "__main__":
    main()
