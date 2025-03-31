from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsPointXY,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsGeometry,
    QgsField,
    QgsRaster,
    QgsFeature,
    QgsApplication
)
from processing_saga_nextgen.saga_nextgen_plugin import SagaNextGenAlgorithmProvider
from qgis.PyQt.QtCore import QVariant
import processing
import requests
from qgis.utils import iface
from .common import *

def load_saga_algorithms():
    provider = SagaNextGenAlgorithmProvider()
    provider.loadAlgorithms()

    QgsApplication.processingRegistry().addProvider(provider=provider)


def river(project_folder):    
    set_project_crs()
    enable_processing_algorithms()
    add_opentopo_layer()
    x, y = get_coordinates()
    if x is None or y is None:
        return
    longitude, latitude = transform_coordinates(x, y)
    radius = 0.5
    xmin, ymin = longitude - radius, latitude - radius
    xmax, ymax = longitude + radius, latitude + radius
    # Конфигурация запроса к OpenTopography API
    api_key = 'c1fcbd0b2f691c736e3bf8c43e52a54d'
    url = (
        f"https://portal.opentopography.org/API/globaldem?"
        f"demtype=SRTMGL1"
        f"&south={ymin}&north={ymax}"
        f"&west={xmin}&east={xmax}"
        f"&outputFormat=GTiff"
        f"&API_Key={api_key}"
    )
    response = requests.get(url)

    # Сохранить скачанный DEM на диск
    output_path = f'{project_folder}srtm_output.tif'
    with open(output_path, 'wb') as f:
        f.write(response.content)

    # Добавить скачанный слой DEM в проект QGIS
    dem_layer = QgsRasterLayer(output_path, "SRTM DEM Layer")
    QgsProject.instance().addMapLayer(dem_layer)

    # Репроекция DEM из EPSG:4326 в EPSG:3857
    reprojected_relief = processing.run("gdal:warpreproject", {
        'INPUT': output_path,
        'SOURCE_CRS': QgsCoordinateReferenceSystem('EPSG:4326'),
        'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:3857'),
        'RESAMPLING': 0, 'NODATA': -9999, 'TARGET_RESOLUTION': 30,
        'OPTIONS': '', 'DATA_TYPE': 0, 'TARGET_EXTENT': None,
        'TARGET_EXTENT_CRS': None, 'MULTITHREADING': False,
        'EXTRA': '', 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']


    load_saga_algorithms()
    # Использовать SAGA Fill Sinks для извлечения водосборов
    filled_relief = processing.run("sagang:fillsinkswangliu", {
        'ELEV': reprojected_relief, 'FILLED': 'TEMPORARY_OUTPUT',
        'FDIR': 'TEMPORARY_OUTPUT', 'WSHED': f'{project_folder}basins.sdat',
        'MINSLOPE': 0.01})['WSHED']

    # Сохранить и добавить заполненные области водосбора в проект
    basins = QgsRasterLayer(f'{project_folder}basins.sdat', 'basins')
    QgsProject.instance().addMapLayer(basins)

    # Использовать QuickOSM для запроса данных о водных путях на заданной территории
    bbox = "4261842, 4372940, 7611625, 7813231"

    # Загрузить реки
    query = processing.run('quickosm:buildqueryextent', {
        'KEY': 'waterway', 'VALUE': 'river', 'EXTENT': bbox, 'TIMEOUT': 25})
    file = processing.run("native:filedownloader", {
        'URL': query['OUTPUT_URL'], 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']
    rivers = iface.addVectorLayer(file + '|layername=lines', "rivers", "ogr")

    # Загрузить ручьи
    query = processing.run('quickosm:buildqueryextent', {
        'KEY': 'waterway', 'VALUE': 'stream', 'EXTENT': bbox, 'TIMEOUT': 25})
    file = processing.run("native:filedownloader", {
        'URL': query['OUTPUT_URL'], 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']
    streams = iface.addVectorLayer(file + '|layername=lines', "streams", "ogr")

    # Объединить слои рек и ручьев
    layer1 = QgsProject.instance().mapLayersByName("rivers")[0]
    layer2 = QgsProject.instance().mapLayersByName("streams")[0]
    merge_result = processing.run("qgis:mergevectorlayers", {
        'LAYERS': [layer1, layer2], 'CRS': layer1.crs(),
        'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']
    merge_result = processing.run("native:dissolve",
                            {'INPUT': merge_result, 'FIELD': [], 'SEPARATE_DISJOINT': False, 'OUTPUT': 'TEMPORARY_OUTPUT'})[
        'OUTPUT']
    merge_result = processing.run("native:multiparttosingleparts", {'INPUT': merge_result, 'OUTPUT': f'{project_folder}merge_result.gpkg'})['OUTPUT']

    # Добавить объединенный слой в проект
    rivers_merged = QgsVectorLayer(f'{project_folder}merge_result.gpkg', 'rivers_merged')
    QgsProject.instance().addMapLayer(rivers_merged)

    # Загрузить полигональные данные о водных объектах
    query = processing.run('quickosm:buildqueryextent', {
        'KEY': 'natural', 'VALUE': 'water', 'EXTENT': bbox, 'TIMEOUT': 25})
    file = processing.run("native:filedownloader", {
        'URL': query['OUTPUT_URL'], 'OUTPUT': f'{project_folder}water.gpkg'})['OUTPUT']
    water = iface.addVectorLayer(file + '|layername=multipolygons', "water", "ogr")

    # Рассчитать координаты начальных и конечных точек линий
    start_x = processing.run("native:fieldcalculator", {
        'INPUT': f'{project_folder}merge_result.gpkg', 'FIELD_NAME': 'start_x',
        'FIELD_TYPE': 0, 'FIELD_LENGTH': 0, 'FIELD_PRECISION': 0,
        'FORMULA': 'x(start_point($geometry))', 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']

    start_y = processing.run("native:fieldcalculator", {
        'INPUT': start_x, 'FIELD_NAME': 'start_y',
        'FIELD_TYPE': 0, 'FIELD_LENGTH': 0, 'FIELD_PRECISION': 0,
        'FORMULA': 'y(start_point($geometry))', 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']

    end_x = processing.run("native:fieldcalculator", {
        'INPUT': start_y, 'FIELD_NAME': 'end_x',
        'FIELD_TYPE': 0, 'FIELD_LENGTH': 0, 'FIELD_PRECISION': 0,
        'FORMULA': 'x(end_point($geometry))', 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']

    end_y = processing.run("native:fieldcalculator", {
        'INPUT': end_x, 'FIELD_NAME': 'end_y',
        'FIELD_TYPE': 0, 'FIELD_LENGTH': 0, 'FIELD_PRECISION': 0,
        'FORMULA': 'y(end_point($geometry))', 'OUTPUT': 'TEMPORARY_OUTPUT'})['OUTPUT']

    # Добавить новые поля для хранения высотных данных
    layer_provider = end_y.dataProvider()
    layer_provider.addAttributes([QgsField("start_z", QVariant.Double)])
    layer_provider.addAttributes([QgsField("end_z", QVariant.Double)])
    end_y.updateFields()

    # Начать редактирование и заполнение значений высоты
    end_y.startEditing()
    line_provider = end_y.dataProvider()

    for feature in end_y.getFeatures():
        geom = feature.geometry()
        if geom.isMultipart():
            polyline = geom.asMultiPolyline()[0]
        else:
            polyline = geom.asPolyline()

        start_point = QgsPointXY(polyline[0])
        end_point = QgsPointXY(polyline[-1])

        # Высотные данные начальной точки
        start_z = dem_layer.dataProvider().identify(start_point, QgsRaster.IdentifyFormatValue)

        start_z_idx = line_provider.fields().indexOf('start_z')
        if start_z.isValid():
            start_z_value = start_z.results()[1]
            feature['start_z'] = start_z_value

        # Высотные данные конечной точки
        end_z = dem_layer.dataProvider().identify(end_point, QgsRaster.IdentifyFormatValue)
        end_z_idx = line_provider.fields().indexOf('end_z')
        if end_z.isValid():
            end_z_value = end_z.results()[1]
            feature['end_z'] = end_z_value

        line_provider.changeAttributeValues({feature.id(): {start_z_idx: start_z_value, end_z_idx: end_z_value}})

    end_y.commitChanges()

    # Определить максимальную высоту для каждой линии
    max_z = processing.run("native:fieldcalculator", {
        'INPUT': end_y, 'FIELD_NAME': 'max_z',
        'FIELD_TYPE': 0, 'FIELD_LENGTH': 0, 'FIELD_PRECISION': 0,
        'FORMULA': 'if("start_z" > "end_z", "start_z", "end_z")',
        'OUTPUT': f'{project_folder}rivers_with_points.gpkg'})['OUTPUT']

    rivers_and_points = QgsVectorLayer(max_z, 'rivers_and_points')
    QgsProject.instance().addMapLayer(rivers_and_points)

    # Создать слой точек максимальной высоты
    point_layer = QgsVectorLayer("Point?crs=epsg:4326", "MaxHeightPoints", "memory")
    QgsProject.instance().addMapLayer(point_layer)
    layer_provider = point_layer.dataProvider()
    layer_provider.addAttributes([QgsField("x", QVariant.Double)])
    layer_provider.addAttributes([QgsField("y", QVariant.Double)])
    layer_provider.addAttributes([QgsField("z", QVariant.Double)])
    point_layer.updateFields()
    fields = point_layer.fields()

    # Получить ссылки на слои линий и точек
    line_layer_name = 'rivers_and_points'
    layers = QgsProject.instance().mapLayersByName(line_layer_name)
    layer = layers[0]

    point_layer_name = 'MaxHeightPoints'
    pointLayers = QgsProject.instance().mapLayersByName(point_layer_name)
    pointLayer = pointLayers[0]



    # Сначала собираем все конечные и начальные точки
    start_points = set()
    end_points = set()

    for feature in layer.getFeatures():
        start_x = feature['start_x']
        start_y = feature['start_y']
        end_x = feature['end_x']
        end_y = feature['end_y']

        if start_x is not None and start_y is not None:
            start_points.add((start_x, start_y))

        if end_x is not None and end_y is not None:
            end_points.add((end_x, end_y))

    pointLayer.startEditing()

    for feature in layer.getFeatures():
        if feature['max_z'] is not None:
            start_x = feature['start_x']
            start_y = feature['start_y']
            start_z = feature['start_z']
            end_x = feature['end_x']
            end_y = feature['end_y']
            end_z = feature['end_z']

            # Проверка начальной точки
            if start_x is not None and start_y is not None and start_z is not None:
                start_point = (start_x, start_y)
                if start_z == feature['max_z'] and start_point not in end_points:
                    point = QgsPointXY(start_x, start_y)
                    new_feature = QgsFeature()
                    new_feature.setFields(fields)
                    new_feature.setGeometry(QgsGeometry.fromPointXY(point))
                    new_feature['x'] = start_x
                    new_feature['y'] = start_y
                    new_feature['z'] = start_z
                    pointLayer.addFeature(new_feature)

            # Проверка конечной точки
            if end_x is not None and end_y is not None and end_z is not None:
                end_point = (end_x, end_y)
                if end_z == feature['max_z'] and end_point not in start_points:
                    point = QgsPointXY(end_x, end_y)
                    new_feature = QgsFeature()
                    new_feature.setFields(fields)
                    new_feature.setGeometry(QgsGeometry.fromPointXY(point))
                    new_feature['x'] = end_x
                    new_feature['y'] = end_y
                    new_feature['z'] = end_z
                    pointLayer.addFeature(new_feature)

    # Завершение редактирования и сохранение изменений
    pointLayer.commitChanges(True)