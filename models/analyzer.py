import json
from shapely.geometry import Point, Polygon

class Analyzer:
    def __init__(self, config_data):
        """
        Инициализация анализатора.
        :param config_data: конфигурация с парковочными местами
        """
        self.camera_id = config_data.get("camera_id", "unknown")
        self.parking_places = []
        
        # Преобразуем координаты полигонов в объекты Shapely для удобного расчета пересечений
        for place in config_data.get("parking_places", []):
            poly = Polygon(place["polygon"])
            self.parking_places.append({
                "id": place["id"],
                "polygon": poly,
                "area": poly.area
            })

    def analyze(self, transformed_points):
        """
        Определяет занятость парковок на основе спроецированных точек (центров авто).
        :param transformed_points: Список словарей {"point": (x, y), "confidence": 0...100}
        :return: JSON объект со списком событий занятости
        """
        events = []
        for place in self.parking_places:
            poly = place["polygon"]
            
            # Простейшая метрика: если центр или нижняя точка машины находится внутри полигона, место занято.
            # Для более сложных задач можно использовать IoU спроецированных bbox.
            occupancy = 0
            for pt in transformed_points:
                point_geom = Point(pt["point"])
                if poly.contains(point_geom):
                    occupancy = pt["confidence"]
                    break # Одного авто достаточно для занятости. Можно добавить логику агрегации.
            
            events.append({
                "camera_id": self.camera_id,
                "parking_id": place["id"],
                "occupancy_percent": occupancy * 100,
                "timestamp": "now" # В реальной системе тут timestamp кадра
            })
            
        return events
