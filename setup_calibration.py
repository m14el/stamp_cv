import cv2
import json
import numpy as np
import argparse
# Глобальные переменные
calib_points = []
parking_polygons = []
current_polygon = []
mode = "calib" # "calib" или "parking"
warped_img = None
target_w, target_h = 400, 400

def click_event(event, x, y, flags, param):
    global calib_points, current_polygon, mode, img, clone, warped_img, clone_warped

    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == "calib":
            if len(calib_points) < 4:
                calib_points.append([x, y])
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(img, str(len(calib_points)), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.imshow("Calibration", img)
                if len(calib_points) == 4:
                    print("=> 4 точки выбраны! Нажмите 'n' чтобы перейти к разметке мест.")
                    cv2.polylines(img, [np.array(calib_points)], True, (0,255,255), 2)
                    cv2.imshow("Calibration", img)
        elif mode == "parking":
            current_polygon.append([x, y])
            cv2.circle(warped_img, (x, y), 5, (0, 255, 0), -1)
            if len(current_polygon) > 1:
                cv2.line(warped_img, tuple(current_polygon[-2]), tuple(current_polygon[-1]), (0, 255, 0), 2)
            cv2.imshow("Draw Parking Spaces (Bird's Eye)", warped_img)

def main():
    global img, clone, warped_img, clone_warped, mode, calib_points, parking_polygons, current_polygon, target_w, target_h
    parser = argparse.ArgumentParser(description="ParkCloud UI Configurator")
    parser.add_argument("--image", type=str, required=True, help="Path to the dataset image (e.g. dataset/cam1.jpg)")
    parser.add_argument("--config_out", type=str, default="config.json", help="Path to save config (default: config.json)")
    args = parser.parse_args()
    
    img_path = args.image
    config_out = args.config_out
    
    print("===== Инструмент калибровки ParkCloud =====")
    print("ШАГ 1: Калибровка перспективы")
    print(f"- Картина: {img_path}")
    print("- Кликните 4 точки на асфальте, которые образуют прямоугольник в реальной жизни (например, контур 2-х парковочных мест).")
    print("- Порядок кликов: Верхний Левый, Верхний Правый, Нижний Правый, Нижний Левый.")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Ошибка: не могу загрузить {img_path}")
        return
        
    clone = img.copy()
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", click_event)
    
    while True:
        if mode == "calib":
            cv2.imshow("Calibration", img)
        else:
            cv2.imshow("Draw Parking Spaces (Bird's Eye)", warped_img)
            
        key = cv2.waitKey(1) & 0xFF
        
        # Нажатие "n" (next)
        if key == ord("n"):
            if mode == "calib" and len(calib_points) == 4:
                mode = "parking"
                cv2.destroyWindow("Calibration")
                
                # Расчет перспективы
                src_pts = np.float32(calib_points)
                dst_pts = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Трансформация картинки, чтобы рисовать полигоны прямо на плоском кадре
                warped_img = cv2.warpPerspective(clone, matrix, (target_w, target_h))
                clone_warped = warped_img.copy()
                
                cv2.namedWindow("Draw Parking Spaces (Bird's Eye)")
                cv2.setMouseCallback("Draw Parking Spaces (Bird's Eye)", click_event)
                
                print("\nШАГ 2: Разметка машиномест")
                print("- Теперь вы видите парковку 'Сверху' (Bird's Eye View).")
                print("- Кликайте точки, чтобы обвести КАЖДОЕ парковочное место многоугольником.")
                print("- Обвели одно место? Нажмите 'n', чтобы сохранить его и начать рисовать следующее.")
                print(f"- Когда разметите все места, нажмите 's' для сохранения в {config_out}.")
                
            elif mode == "parking" and len(current_polygon) >= 3:
                cv2.line(warped_img, tuple(current_polygon[-1]), tuple(current_polygon[0]), (0, 255, 0), 2)
                parking_polygons.append(current_polygon.copy())
                print(f"=> Сохранено машиноместо A{len(parking_polygons)}")
                current_polygon = []
                clone_warped = warped_img.copy() # Фиксируем рисунок
                
        # Нажатие "r" (reset)
        elif key == ord("r"):
            if mode == "calib":
                img = clone.copy()
                calib_points.clear()
            else:
                warped_img = clone_warped.copy()
                current_polygon.clear()
            print("Сброс точек...")
            
        # Нажатие "s" (save config)
        elif key == ord("s"):
            if mode == "parking":
                if len(current_polygon) >= 3: 
                     parking_polygons.append(current_polygon.copy())
                
                config = {
                    "camera_id": img_path,
                    "calibration_points": calib_points,
                    "target_width": target_w,
                    "target_height": target_h,
                    "parking_places": []
                }
                
                for i, p in enumerate(parking_polygons):
                    config["parking_places"].append({
                        "id": f"A{i+1}",
                        "polygon": p
                    })
                    
                with open(config_out, "w") as f:
                    json.dump(config, f, indent=2)
                
                print(f"\nУСПЕХ! Данные сохранены в {config_out} ({len(parking_polygons)} мест).")
                print(f"Теперь можете снова запустить: python main.py --config {config_out} --image {img_path}")
                break
                
        # Нажатие "q" или ESC (выход)
        elif key == ord("q") or key == 27:
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
