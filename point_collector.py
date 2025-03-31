from qgis.core import (
    QgsProject,
    QgsPointXY,
    QgsVectorLayer,
    QgsGeometry,
    QgsFeature,
)
from qgis.gui import QgsMapToolEmitPoint
from PyQt5.QtCore import pyqtSignal

# 1. Сбор точек пользователем
class PointCollector(QgsMapToolEmitPoint):
    collection_complete = pyqtSignal()  # Сигнал завершения сбора точек

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.points = []
        
        # Создаем временный слой для отображения точек
        self.point_layer = QgsVectorLayer("Point?crs=EPSG:3857", "Selected Points", "memory")
        self.point_provider = self.point_layer.dataProvider()
        QgsProject.instance().addMapLayer(self.point_layer)
        
        # Список аннотаций для отображения координат
        self.annotations = []

    def canvasPressEvent(self, event):
        point = self.toMapCoordinates(event.pos())
        self.points.append(point)
        print(f"Точка выбрана: {point}")
        
        # Добавляем точку в слой
        point_feature = QgsFeature()
        point_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point)))
        self.point_provider.addFeatures([point_feature])
        self.point_layer.updateExtents()
        self.point_layer.triggerRepaint()

    def get_points(self):
        return self.points

    def complete_collection(self):
        # Генерируем сигнал, чтобы уведомить об окончании сбора
        self.collection_complete.emit()