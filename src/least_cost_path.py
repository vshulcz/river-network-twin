from qgis.core import (
    QgsProject,
    QgsPointXY,
    QgsGeometry,
    QgsSpatialIndex,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsMessageLog
)
from qgis.PyQt.QtWidgets import QMessageBox
import processing
import os

# Возвращает путь к полученному файлу
def generate_slope_layer(dem_path: str, project_folder: str) -> str:
    os.makedirs(f"{project_folder}/tmp/", exist_ok=True)

    dem_layer = QgsRasterLayer(dem_path, "DEM")
    crs = dem_layer.crs().authid()
    extent = dem_layer.extent()
    center_longitude = (extent.xMinimum() + extent.xMaximum()) / 2
    utm_zone = int((center_longitude + 180) / 6) + 1
    target_crs = f"EPSG:{32600 + utm_zone}" if extent.yMinimum() >= 0 else f"EPSG:{32700 + utm_zone}"

    # 2. Перепроектируем DEM в UTM
    params_reproject = {
        'INPUT': dem_path,
        'TARGET_CRS': target_crs,
        'RESAMPLING': 1,  # 0 = Nearest Neighbour (для категорийных), 1 = Bilinear (лучше для DEM)
        'OUTPUT': 'TEMPORARY_OUTPUT'
    }
    output_reprojected = processing.run("gdal:warpreproject", params_reproject)['OUTPUT']

    # 3. Рассчитываем уклон (в градусах)
    params_slope = {
        'INPUT': output_reprojected,
        'BAND': 1,
        'SCALE': 1,  # Масштаб (оставьте 1, если пиксели в метрах)
        'AS_PERCENT': False,  # False = градусы, True = проценты
        'ZEVENBERGEN': False,  # False = стандартный алгоритм, True = более гладкий
        'OUTPUT': 'TEMPORARY_OUTPUT'
    }
    output_slope_utm = processing.run("gdal:slope", params_slope)['OUTPUT']

    # возвращаем в исходные координаты
    output_slope = f"{project_folder}/slope_layer.tif"
    params = {
        'INPUT': output_slope_utm,
        'TARGET_CRS': crs,  # Целевая система координат
        'RESAMPLING': 1,  # 1 = Bilinear (лучше для DEM)
        'OUTPUT': output_slope
    }
    processing.run("gdal:warpreproject", params)

    return output_slope

def least_cost_path_analysis(project_folder):
    # Получение необходимых слоев
    try:
        points_layer = QgsProject.instance().mapLayersByName('MaxHeightPoints')[0]
    except IndexError:
        QMessageBox.warning(None, "Ошибка", "Слой 'MaxHeightPoints' не найден.")
        return

    try:
        cost_layer = QgsProject.instance().mapLayersByName('Slope Layer')[0]
    except IndexError:
        slope_layer_path = generate_slope_layer(f"{project_folder}/srtm_output.tif", project_folder)
        cost_layer = QgsRasterLayer(slope_layer_path, 'Slope Layer')
        if cost_layer.isValid():
            QgsProject.instance().addMapLayer(cost_layer)
            print("Создан слой уклона", flush=True)
        else:
            QMessageBox.warning(None, "Ошибка", "Не удалось загрузить слой крутизны.")
            return


    start_layer = QgsVectorLayer("Point", "point", "memory")
    start_layer.setCrs(points_layer.sourceCrs()) # set the coordinate system from layer to the new layer
    pr1 = start_layer.dataProvider() # get the layers data provider (can't explain what it is exactly)
    for p in points_layer.getFeatures():
        if p['z'] != None:
            point = p
            break
    pr1.addFeature(point) # add the feature to the layer via data provider
    start_layer.updateFeature(point)

    print("Построен слой с точкой", flush=True)

    # Параметры для алгоритма Least Cost Path
    params = {
        'INPUT_COST_RASTER': cost_layer,
        'INPUT_RASTER_BAND': 1,
        'INPUT_START_LAYER': start_layer,
        'INPUT_END_LAYER': points_layer,
        'BOOLEAN_FIND_LEAST_PATH_TO_ALL_ENDS': False,
        'BOOLEAN_OUTPUT_LINEAR_REFERENCE': False,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    }
    try:
        result = processing.run("Cost distance analysis:Least Cost Path", params)
        path_layer = result['OUTPUT']
        path_layer.setName("Output least cost path")
        QgsProject.instance().addMapLayer(path_layer)
        print("Создан слой с путями", flush=True)
    except Exception as e:
        QMessageBox.critical(None, "Ошибка", f"Ошибка при выполнении анализа: {e}")
        return

    # Вспомогательная функция для расчёта минимальной высоты вдоль линии
    def calculate_minimum_elevation(raster_layer, line_geom):
        provider = raster_layer.dataProvider()
        min_elev = float('inf')
        if line_geom.isMultipart():
            lines = line_geom.asMultiPolyline()
        else:
            lines = [line_geom.asPolyline()]
        for line in lines:
            for pt in line:
                sample = provider.sample(QgsPointXY(pt.x(), pt.y()), 1)
                if sample:
                    value, valid = sample
                    if valid and value is not None:
                        min_elev = min(min_elev, value)
        return min_elev if min_elev != float('inf') else None

    try:
        elevation_layer = QgsProject.instance().mapLayersByName('SRTM DEM Layer')[0]
    except IndexError:
        QMessageBox.warning(None, "Ошибка", "Слой 'SRTM DEM Layer' не найден.")
        return
    return

    # Фильтрация путей по критерию разницы высот
    paths_to_delete = []
    for feature in path_layer.getFeatures():
        geom = feature.geometry()
        min_elev = calculate_minimum_elevation(elevation_layer, geom)
        if min_elev is None:
            continue
        if geom.isMultipart():
            polyline = geom.asMultiPolyline()[0]
        else:
            polyline = geom.asPolyline()
        if not polyline:
            continue
        first_point = polyline[0]
        last_point = polyline[-1]
        sample_start = elevation_layer.dataProvider().sample(QgsPointXY(first_point.x(), first_point.y()), 1)
        sample_end = elevation_layer.dataProvider().sample(QgsPointXY(last_point.x(), last_point.y()), 1)
        if sample_start and sample_end:
            z_start, valid_start = sample_start
            z_end, valid_end = sample_end
            if not (valid_start and valid_end):
                continue
            z1 = min(z_start, z_end)
            if min_elev < z1 - 15:
                paths_to_delete.append(feature.id())
    if paths_to_delete:
        path_layer.startEditing()
        for fid in paths_to_delete:
            path_layer.deleteFeature(fid)
        path_layer.commitChanges()
        QMessageBox.information(None, "Информация", f"Удалено {len(paths_to_delete)} путей по критерию высоты.")

    # Фильтрация путей, пересекающих реки (исключая совпадающие начала/концы)
    try:
        rivers_layer = QgsProject.instance().mapLayersByName('rivers_and_points')[0]
    except IndexError:
        QMessageBox.warning(None, "Ошибка", "Слой 'rivers_and_points' не найден.")
        return
    spatial_index = QgsSpatialIndex(rivers_layer.getFeatures())
    paths_to_delete = []
    for feature in path_layer.getFeatures():
        geom = feature.geometry()
        if geom.isEmpty():
            continue
        pts = geom.asPolyline()
        if not pts:
            continue
        start_geom = QgsGeometry.fromPointXY(pts[0])
        end_geom = QgsGeometry.fromPointXY(pts[-1])
        candidate_ids = spatial_index.intersects(geom.boundingBox())
        for cid in candidate_ids:
            candidate = rivers_layer.getFeature(cid)
            if geom.intersects(candidate.geometry()):
                if start_geom.intersects(candidate.geometry()) or end_geom.intersects(candidate.geometry()):
                    continue
                paths_to_delete.append(feature.id())
                break
    if paths_to_delete:
        path_layer.startEditing()
        for fid in paths_to_delete:
            path_layer.deleteFeature(fid)
        path_layer.commitChanges()
        QMessageBox.information(None, "Информация", f"Удалено {len(paths_to_delete)} путей, пересекающих реки.")