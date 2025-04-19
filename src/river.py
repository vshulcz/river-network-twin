import os
import processing
import requests
from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsPointXY,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsGeometry,
    QgsField,
    QgsRaster,
    QgsFeature,
    QgsApplication,
    QgsRectangle,
)
from qgis.analysis import QgsNativeAlgorithms
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QMessageBox,
    QAction,
    QInputDialog
)
from processing_saga_nextgen.saga_nextgen_plugin import SagaNextGenAlgorithmProvider
from qgis.PyQt.QtCore import QVariant, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.gui import QgsVertexMarker, QgsMapToolEmitPoint
from qgis.utils import iface
from collections import deque
from .common import *

class PointSelectionTool(QgsMapToolEmitPoint):
    selection_completed = pyqtSignal(list)

    def __init__(self, canvas):
        super().__init__(canvas)
        self.points = []
        self.canvas = canvas
        self.markers = []

    def canvasPressEvent(self, event):
        point = self.toMapCoordinates(event.pos())
        self.points.append(point)

        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(point)
        marker.setColor(QColor(255, 0, 0))
        marker.setIconSize(10)
        self.markers.append(marker)

        if len(self.points) == 4:
            self.selection_completed.emit(self.points)
            self.cleanup()

    def cleanup(self):
        for marker in self.markers:
            self.canvas.scene().removeItem(marker)
        self.markers = []
        self.canvas.unsetMapTool(self)



def load_saga_algorithms():
    provider = SagaNextGenAlgorithmProvider()
    provider.loadAlgorithms()

    QgsApplication.processingRegistry().addProvider(provider=provider)


def transform_bbox(x_min, x_max, y_min, y_max, from_epsg, to_epsg):
    # Создаем объекты систем координат
    from_crs = QgsCoordinateReferenceSystem(f"EPSG:{from_epsg}")
    to_crs = QgsCoordinateReferenceSystem(f"EPSG:{to_epsg}")
    tr = QgsCoordinateTransform(from_crs, to_crs, QgsProject.instance())

    # Преобразуем все 4 угла bbox
    points = [
        QgsPointXY(x_min, y_min),
        QgsPointXY(x_min, y_max),
        QgsPointXY(x_max, y_max),
        QgsPointXY(x_max, y_min)
    ]

    transformed_points = [tr.transform(point) for point in points]

    # Находим новые границы
    x_coords = [p.x() for p in transformed_points]
    y_coords = [p.y() for p in transformed_points]

    return f"{min(x_coords)}, {max(x_coords)}, {min(y_coords)}, {max(y_coords)}"

def set_project_crs():
    crs = QgsCoordinateReferenceSystem("EPSG:3857")
    QgsProject.instance().setCrs(crs)

def enable_processing_algorithms():
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

def add_opentopo_layer():
    opentopo_url = 'type=xyz&zmin=0&zmax=21&url=https://tile.opentopomap.org/{z}/{x}/{y}.png'
    opentopo_layer = QgsRasterLayer(opentopo_url, 'OpenTopoMap', 'wms')
    QgsProject.instance().addMapLayer(opentopo_layer)

def get_coordinates():
    x, ok_x = QInputDialog.getDouble(None, "Координата X", "Введите координату X:", value=4316873, decimals=6)
    if not ok_x:
        return None, None

    y, ok_y = QInputDialog.getDouble(None, "Координата Y", "Введите координату Y:", value=7711643, decimals=6)
    if not ok_y:
        return None, None

    return x, y


def transform_coordinates(x, y):
    transformer = QgsCoordinateTransform(
        QgsCoordinateReferenceSystem("EPSG:3857"),
        QgsCoordinateReferenceSystem("EPSG:4326"),
        QgsProject.instance()
    )
    point = QgsPointXY(x, y)
    transformed_point = transformer.transform(point)
    return transformed_point.x(), transformed_point.y()


def river(project_folder):
    set_project_crs()
    enable_processing_algorithms()
    add_opentopo_layer()

    method, ok = QInputDialog.getItem(
        None,
        "Выбор метода",
        "Как определить область анализа?",
        ["По радиусу вокруг точки", "По 4 точкам на карте"],
        0, False
    )
    if not ok:
        return
    if method == "По радиусу вокруг точки":
        radius, ok = QInputDialog.getDouble(
            None,
            "Выбор территории",
            "Введите радиус вокруг точки (в градусах):",
            value=0.5, min=0.1, max=5, decimals=1
        )
        if not ok:
            return

        x, y = get_coordinates()
        if x is None or y is None:
            return

        longitude, latitude = transform_coordinates(x, y)
        bbox = [longitude - 0.5, latitude - 0.5, longitude + 0.5, latitude + 0.5]

    else:
        canvas = iface.mapCanvas()
        tool = PointSelectionTool(canvas)
        canvas.setMapTool(tool)

        QMessageBox.information(
            None,
            "Выбор территории",
            "Выберите 4 точки на карте, определяющие область анализа"
        )

        from qgis.PyQt.QtCore import QEventLoop
        loop = QEventLoop()
        tool.selection_completed.connect(loop.quit)
        loop.exec_()

        points = tool.points
        if len(points) != 4:
            return

        x_coords = [p.x() for p in points]
        y_coords = [p.y() for p in points]
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

        transform = QgsCoordinateTransform(
            QgsCoordinateReferenceSystem("EPSG:3857"),
            QgsCoordinateReferenceSystem("EPSG:4326"),
            QgsProject.instance()
        )

        points_4326 = []
        for point in points:
            point_4326 = transform.transform(point)
            points_4326.append(point_4326)

        x_coords = [p.x() for p in points_4326]
        y_coords = [p.y() for p in points_4326]
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    dem_path = download_dem(bbox, project_folder)
    dem_layer = add_dem_layer(dem_path)
    reprojected_relief = reproject_dem(dem_path)

    load_saga_algorithms()
    # Использовать SAGA Fill Sinks для извлечения водосборов
    basins_path = os.path.join(project_folder, "basins.sdat")
    processing.run(
        "sagang:fillsinkswangliu",
        {
            "ELEV": reprojected_relief,
            "FILLED": "TEMPORARY_OUTPUT",
            "FDIR": "TEMPORARY_OUTPUT",
            "WSHED": basins_path,
            "MINSLOPE": 0.01,
        },
    )

    # Сохранить и добавить заполненные области водосбора в проект
    basins = QgsRasterLayer(basins_path, "basins")
    QgsProject.instance().addMapLayer(basins)


    # Использовать QuickOSM для запроса данных о водных путях на заданной территории
    extent =  transform_bbox(bbox[0], bbox[2], bbox[1], bbox[3], 4326, 3857)

    # Загрузить реки
    rivers_layer = load_quickosm_layer("rivers", "waterway", "river", extent)
    # Загрузить ручьи
    streams_layer = load_quickosm_layer("streams", "waterway", "stream", extent)

    # Объединить слои рек и ручьев
    merged_filepath = os.path.join(project_folder, "merge_result.gpkg")
    merge_result = processing.run(
        "qgis:mergevectorlayers",
        {
            "LAYERS": [rivers_layer, streams_layer],
            "CRS": rivers_layer.crs(),
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]
    merge_result = processing.run(
        "native:dissolve",
        {
            "INPUT": merge_result,
            "FIELD": [],
            "SEPARATE_DISJOINT": False,
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]
    processing.run(
        "native:multiparttosingleparts",
        {
            "INPUT": merge_result,
            "OUTPUT": merged_filepath,
        },
    )["OUTPUT"]

    # Добавить объединенный слой в проект
    rivers_merged = QgsVectorLayer(merged_filepath, "rivers_merged", "ogr")
    QgsProject.instance().addMapLayer(rivers_merged)

    # Загрузить полигональные данные о водных объектах
    water_output = os.path.join(project_folder, "water.gpkg")
    load_quickosm_layer(
        "water",
        "natural",
        "water",
        extent,
        output_path=water_output,
        quickosm_layername="multipolygons",
    )

    # Рассчитать координаты начальных и конечных точек линий
    start_x = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": merged_filepath,
            "FIELD_NAME": "start_x",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": "x(start_point($geometry))",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]

    start_y = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": start_x,
            "FIELD_NAME": "start_y",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": "y(start_point($geometry))",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]

    end_x = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": start_y,
            "FIELD_NAME": "end_x",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": "x(end_point($geometry))",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]

    end_y = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": end_x,
            "FIELD_NAME": "end_y",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": "y(end_point($geometry))",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]

    # Добавить новые поля для хранения высотных данных
    layer_provider = end_y.dataProvider()
    layer_provider.addAttributes(
        [QgsField("start_z", QVariant.Double), QgsField("end_z", QVariant.Double)]
    )
    end_y.updateFields()
    idx_start_z = layer_provider.fields().indexOf("start_z")
    idx_end_z = layer_provider.fields().indexOf("end_z")

    # Начать редактирование и заполнение значений высоты
    end_y.startEditing()
    line_provider = end_y.dataProvider()
    changes = {}
    for feature in end_y.getFeatures():
        geom = feature.geometry()
        if geom.isMultipart():
            polyline = geom.asMultiPolyline()[0]
        else:
            polyline = geom.asPolyline()

        start_point = QgsPointXY(polyline[0])
        end_point = QgsPointXY(polyline[-1])

        # Высотные данные начальной точки
        start_z = dem_layer.dataProvider().identify(
            start_point, QgsRaster.IdentifyFormatValue
        )
        # Высотные данные конечной точки
        end_z = dem_layer.dataProvider().identify(
            end_point, QgsRaster.IdentifyFormatValue
        )

        start_z_value = None
        end_z_value = None

        if start_z.isValid():
            start_z_value = start_z.results()[1]
            feature["start_z"] = start_z_value

        if end_z.isValid():
            end_z_value = end_z.results()[1]
            feature["end_z"] = end_z_value

        changes[feature.id()] = {idx_start_z: start_z_value, idx_end_z: end_z_value}

    line_provider.changeAttributeValues(changes)
    end_y.commitChanges()

    # Определить максимальную высоту для каждой линии
    max_z = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": end_y,
            "FIELD_NAME": "max_z",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": 'if("start_z" > "end_z", "start_z", "end_z")',
            "OUTPUT": f"{project_folder}rivers_with_points.gpkg",
        },
    )["OUTPUT"]

    rivers_and_points = QgsVectorLayer(max_z, "rivers_and_points", "ogr")
    QgsProject.instance().addMapLayer(rivers_and_points)

    # Создать слой точек максимальной высоты
    point_layer = QgsVectorLayer("Point?crs=epsg:4326", "MaxHeightPoints", "memory")
    QgsProject.instance().addMapLayer(point_layer)
    layer_provider = point_layer.dataProvider()
    layer_provider.addAttributes(
        [
            QgsField("x", QVariant.Double),
            QgsField("y", QVariant.Double),
            QgsField("z", QVariant.Double),
        ]
    )
    point_layer.updateFields()
    fields = point_layer.fields()

    # Сначала собираем все конечные и начальные точки
    start_points = set()
    end_points = set()

    for feat in rivers_and_points.getFeatures():
        sx, sy = feat["start_x"], feat["start_y"]
        ex, ey = feat["end_x"], feat["end_y"]
        if sx is not None and sy is not None:
            start_points.add((sx, sy))
        if ex is not None and ey is not None:
            end_points.add((ex, ey))

    point_layer.startEditing()
    for feat in rivers_and_points.getFeatures():
        max_z = feat["max_z"]
        if max_z is None:
            continue

        sx, sy, start_z = feat["start_x"], feat["start_y"], feat["start_z"]
        ex, ey, end_z = feat["end_x"], feat["end_y"], feat["end_z"]

        # Проверка начальной точки
        if (
            sx is not None
            and sy is not None
            and start_z is not None
            and start_z == max_z
        ):
            if (sx, sy) not in end_points:
                pt = QgsPointXY(sx, sy)
                new_feat = QgsFeature()
                new_feat.setFields(fields)
                new_feat.setGeometry(QgsGeometry.fromPointXY(pt))
                new_feat["x"] = sx
                new_feat["y"] = sy
                new_feat["z"] = start_z
                point_layer.addFeature(new_feat)

        # Проверка конечной точки
        if ex is not None and ey is not None and end_z is not None and end_z == max_z:
            if (ex, ey) not in start_points:
                pt = QgsPointXY(ex, ey)
                new_feat = QgsFeature()
                new_feat.setFields(fields)
                new_feat.setGeometry(QgsGeometry.fromPointXY(pt))
                new_feat["x"] = ex
                new_feat["y"] = ey
                new_feat["z"] = end_z
                point_layer.addFeature(new_feat)

    # Завершение редактирования и сохранение изменений
    point_layer.commitChanges(True)
    QgsProject.instance().addMapLayer(point_layer)


def load_quickosm_layer(
    layer_name,
    key,
    value,
    extent,
    output_path="TEMPORARY_OUTPUT",
    quickosm_layername="lines",
):
    query = processing.run(
        "quickosm:buildqueryextent",
        {
            "KEY": key,
            "VALUE": value,
            "EXTENT": extent,
            "TIMEOUT": 25,
        },
    )
    download_result = processing.run(
        "native:filedownloader",
        {
            "URL": query["OUTPUT_URL"],
            "OUTPUT": output_path,
        },
    )["OUTPUT"]
    layer = iface.addVectorLayer(
        download_result + f"|layername={quickosm_layername}", layer_name, "ogr"
    )
    return layer
