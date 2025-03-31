from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsRasterLayer,
    QgsApplication
)
from qgis.analysis import QgsNativeAlgorithms
from qgis.PyQt.QtWidgets import QInputDialog, QMessageBox
from pyproj import Transformer
import processing
import requests

# Установить систему координат проекта (EPSG:3857 - Pseudo-Mercator)
def set_project_crs():
    crs = QgsCoordinateReferenceSystem("EPSG:3857")
    QgsProject.instance().setCrs(crs)

# Включить встроенные алгоритмы обработки QGIS
def enable_processing_algorithms():
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

# Добавить слой OpenTopoMap
def add_opentopo_layer():
    opentopo_url = 'type=xyz&zmin=0&zmax=21&url=https://tile.opentopomap.org/{z}/{x}/{y}.png'
    opentopo_layer = QgsRasterLayer(opentopo_url, 'OpenTopoMap', 'wms')
    QgsProject.instance().addMapLayer(opentopo_layer)

# Координаты озера в системе координат EPSG:3857
#x_3857, y_3857 = 4316873, 7711643
def get_coordinates():
    x, ok_x = QInputDialog.getDouble(None, "Координата X", "Введите координату X:", value=4316873, decimals=6)
    if not ok_x:
        QMessageBox.warning(None, "Ошибка", "Неправильная координата X. Работа плагина прекращена.")
        return None, None

    y, ok_y = QInputDialog.getDouble(None, "Координата Y", "Введите координату Y:", value=7711643, decimals=6)
    if not ok_y:
        QMessageBox.warning(None, "Ошибка", "Неправильная координата Y. Работа плагина прекращена.")
        return None, None

    return x, y
# Преобразовать координаты в широту и долготу (EPSG:4326)
def transform_coordinates(x, y):
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)

# Конфигурация запроса к OpenTopography API
def download_dem(bbox, project_folder):
    api_key = 'c1fcbd0b2f691c736e3bf8c43e52a54d'
    url = (
        f"https://portal.opentopography.org/API/globaldem?"
        f"demtype=SRTMGL1"
        f"&south={bbox[1]}&north={bbox[3]}"
        f"&west={bbox[0]}&east={bbox[2]}"
        f"&outputFormat=GTiff"
        f"&API_Key={api_key}"
    )
    response = requests.get(url)

    output_path = f'{project_folder}srtm_output.tif'
    with open(output_path, 'wb') as f:
        f.write(response.content)

    return output_path

# Репроекция DEM из EPSG:4326 в EPSG:3857
def reproject_dem(dem_path):
    return processing.run("gdal:warpreproject", {
        'INPUT': dem_path,
        'SOURCE_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
        'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:3857'),
        'RESAMPLING': 0, 'NODATA': -9999, 'TARGET_RESOLUTION': 30,
        'OPTIONS': '', 'DATA_TYPE': 0, 'TARGET_EXTENT': None,
        'TARGET_EXTENT_CRS': None, 'MULTITHREADING': False,
        'EXTRA': '', 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']

# Добавить скачанный слой DEM в проект QGIS
def add_dem_layer(dem_path):
    dem_layer = QgsRasterLayer(dem_path, "SRTM DEM Layer")
    QgsProject.instance().addMapLayer(dem_layer)
    return dem_layer

def get_main_def(project_folder):
        set_project_crs()
        enable_processing_algorithms()
        add_opentopo_layer()
        x, y = get_coordinates()
        if x is None or y is None:
            return
        longitude, latitude = transform_coordinates(x, y)
        bbox = [longitude - 0.5, latitude - 0.5, longitude + 0.5, latitude + 0.5]
        dem_path = download_dem(bbox, project_folder)
        dem_layer = add_dem_layer(dem_path)
        return reproject_dem(dem_path)