from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsPointXY,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsGeometry,
    QgsField,
    QgsFields,
    QgsFeature,
    QgsSymbol,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
    QgsProcessingFeatureSourceDefinition
)
from qgis.PyQt.QtCore import QVariant, QEventLoop
from qgis.PyQt.QtGui import QColor
from PyQt5.QtWidgets import QPushButton
from osgeo import gdal
from pyproj import Transformer
import processing
import os
import numpy as np
from qgis.utils import iface
import math
from .common import *
from .point_collector import PointCollector

# Инициализация сбора точек
def start_point_collection():
    
    collector = PointCollector(iface.mapCanvas())
    iface.mapCanvas().setMapTool(collector)
    print("Щелкните по карте, чтобы выбрать точки.")
    
    return collector

# 2. Преобразование точек в полигон
def create_polygon_from_points(points):
    # Преобразуем список точек в полигон (замкнутую геометрию)
    qgs_points = [QgsPointXY(pt.x(), pt.y()) for pt in points]  # Преобразуем в QgsPointXY
    polygon = QgsGeometry.fromPolygonXY([qgs_points])  # Обратите внимание на вложение в список
    return polygon

# 3. Добавление полигона в слой
def add_polygon_to_layer(polygon):
    # Создаем слой в памяти и добавляем полигон
    polygon_layer = QgsVectorLayer('Polygon?crs=EPSG:3857', 'Selected Region', 'memory')
    polygon_layer_data = polygon_layer.dataProvider()
    
    polygon_feature = QgsFeature()
    polygon_feature.setGeometry(polygon)
    polygon_layer_data.addFeatures([polygon_feature])

    QgsProject.instance().addMapLayer(polygon_layer)
    print("Полигон добавлен в проект.")
    return polygon_layer

# 4. Обрезка DEM с использованием полигона
def clip_dem_with_polygon(dem_layer, polygon_layer, masked_dem_output_path, project_folder):
    # Параметры буфера
    buffer_distance = 100  # Задайте отступ в метрах или единицах CRS маски
    buffered_mask = os.path.join(project_folder, "buffered_mask.shp")

    # Создание буфера
    processing.run("native:buffer", {
        'INPUT': polygon_layer,  # Исходная маска
        'DISTANCE': buffer_distance,  # Отступ
        'SEGMENTS': 5,  # Количество сегментов для сглаживания (можно увеличить для кругов)
        'END_CAP_STYLE': 0,  # Открытая граница
        'JOIN_STYLE': 0,  # Стыковка углов (например, круглая)
        'MITER_LIMIT': 2,
        'DISSOLVE': False,
        'OUTPUT': buffered_mask
    })
    # Убедитесь, что буфер был успешно создан
    print(f'Буфер сохранен в: {buffered_mask}')
    # Используем инструмент GDAL для обрезки DEM с использованием полигона
    processing.run("gdal:cliprasterbymasklayer", {
        'INPUT': dem_layer,
        'MASK': QgsProcessingFeatureSourceDefinition(buffered_mask, selectedFeaturesOnly=False),
        'CROP_TO_CUTLINE': True,       # Оставляем все пиксели, пересекающие границу полигона
        'ALL_TOUCHED': True,            # Включаем все пиксели, хотя бы частично затронутые полигоном
        'KEEP_RESOLUTION': True,       # Preserve the resolution of the DEM
        'OUTPUT': masked_dem_output_path
    })
    print(f"Обрезка завершена. Выходной файл: {masked_dem_output_path}")

# Обработчик завершения сбора точек
def handle_points_collection(collector, dem_layer, masked_dem_output_path, project_folder):
    selected_points = collector.get_points()
    
    if len(selected_points) < 3:
        print("Для создания полигона необходимо выбрать хотя бы 3 точки.")
        return

    # 5. Создаем полигон из точек
    polygon = create_polygon_from_points(selected_points)
    
    # 6. Добавляем полигон в слой
    polygon_layer = add_polygon_to_layer(polygon)
    
    # 7. Обрезаем DEM с использованием выбранного полигона
    clip_dem_with_polygon(dem_layer, polygon_layer, masked_dem_output_path, project_folder)

# Добавление кнопки завершения в интерфейс
def add_finish_button(collector, dem_layer, masked_dem_output_path, project_folder):
    finish_button = QPushButton("Завершить сбор точек", iface.mainWindow())
    finish_button.setGeometry(10, 50, 200, 30)  # Позиция и размер кнопки
    finish_button.show()

    def on_button_click():
        collector.complete_collection()
        finish_button.deleteLater()  # Убираем кнопку после завершения
        handle_points_collection(collector, dem_layer, masked_dem_output_path, project_folder)
        process_complete()  # Сигнализируем об окончании процесса

    finish_button.clicked.connect(on_button_click)

# Ожидание завершения процесса
def wait_for_process():
    loop = QEventLoop()

    def stop_waiting():
        loop.quit()

    global process_complete
    process_complete = stop_waiting
    loop.exec_()

# Основная функция для выполнения всего процесса
def process_dem_with_polygon(main_dem_file_path, masked_dem_output_path, project_folder):
    # 1. Загружаем DEM слой
    dem_layer = QgsRasterLayer(main_dem_file_path, "SRTM DEM Layer")
    if not dem_layer.isValid():
        print("Ошибка загрузки DEM слоя.")
        return
    
    QgsProject.instance().addMapLayer(dem_layer)
    print("DEM слой загружен.")
    
    # 2. Начинаем сбор точек
    collector = start_point_collection()

    # 3. Добавляем кнопку завершения
    add_finish_button(collector, dem_layer, masked_dem_output_path, project_folder)
    
    # 4. Ожидаем завершения обработки
    wait_for_process()

def add_dem_layer(output_path):
    # Добавить скачанный слой DEM в проект QGIS
    dem_layer = QgsRasterLayer(output_path, "SRTM DEM Layer")
    if not dem_layer.isValid():
        print(f"Ошибка: Файл {output_path} недоступен или поврежден.")
    QgsProject.instance().addMapLayer(dem_layer)
    return

def reproject_dem2(project_folder):
    # Репроекция DEM из EPSG:4326 в EPSG:3857
    output_reprojected_path =  os.path.join(project_folder, "reprojected_dem.tif")
    output_path =  os.path.join(project_folder, "masked_dem.tif")
    reprojected_dem = processing.run("gdal:warpreproject", {
        'INPUT': output_path,
        'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:3857'),
        'RESAMPLING': 0,
        'NODATA': -9999,
        'TARGET_RESOLUTION': 30,
        'OPTIONS': '',
        'DATA_TYPE': 0,
        'OUTPUT': output_reprojected_path
    })['OUTPUT']
    print(f"Репроекция выполнена. Файл сохранен в {output_reprojected_path}")

    return reprojected_dem

def load_dem_to_numpy(project_folder):
    output_path =  os.path.join(project_folder, "masked_dem.tif")
    # Загрузка DEM в numpy массив для анализа
    dem_raster = gdal.Open(output_path)
    dem_band = dem_raster.GetRasterBand(1)
    dem_data = dem_band.ReadAsArray()
    return dem_data, dem_raster

def setting_dem_coordinates(dem_data, dem_raster):
    # Находим максимальную и минимальную высоту
    max_height = np.max(dem_data)
    min_height = np.min(dem_data)

    # Координаты точек с максимальной и минимальной высотой
    max_coords = np.unravel_index(np.argmax(dem_data), dem_data.shape)
    min_coords = np.unravel_index(np.argmin(dem_data), dem_data.shape)

    # Преобразуем индексы пикселей в географические координаты (EPSG:4326)
    transform = dem_raster.GetGeoTransform()
    max_x_4326 = transform[0] + max_coords[1] * transform[1] + max_coords[0] * transform[2]
    max_y_4326 = transform[3] + max_coords[0] * transform[4] + max_coords[1] * transform[5]

    min_x_4326 = transform[0] + min_coords[1] * transform[1] + min_coords[0] * transform[2]
    min_y_4326 = transform[3] + min_coords[0] * transform[4] + min_coords[1] * transform[5]

    transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Преобразование координат точек с максимальной и минимальной высотой в EPSG:3857
    max_x_3857, max_y_3857 = transformer_to_3857.transform(max_x_4326, max_y_4326)
    min_x_3857, min_y_3857 = transformer_to_3857.transform(min_x_4326, min_y_4326)
    coordinates = [
        (max_x_3857, max_y_3857),
        (min_x_3857, min_y_3857),
    ]
    return coordinates, min_height, max_height

def create_temp_vector_layer():
    # Создаем временный векторный слой для точек
    layer = QgsVectorLayer("Point?crs=EPSG:3857", "points1", "memory")
    QgsProject.instance().addMapLayer(layer)
    return layer

def set_attribute_fields(layer):
    # Добавляем поля для атрибутов (опционально)
    fields = QgsFields()
    fields.append(QgsField("ID", QVariant.Int))

    layer.dataProvider().addAttributes(fields)
    layer.updateFields()
    return layer

def add_points(coordinates, layer):
    features = []
    for i, coord in enumerate(coordinates, start=1):
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(coord[0], coord[1])))
        feature.setAttributes([i])  # ID точки
        features.append(feature)

    layer.dataProvider().addFeatures(features)
    layer.updateExtents()

    print("Точки успешно добавлены!")
    return

def create_points(project_folder):
    output_path =  os.path.join(project_folder, "masked_dem.tif")  # Путь к выходному файлу после обрезки
    add_dem_layer(output_path)
    reprojected_dem_mask = reproject_dem2(project_folder)
    dem_data, dem_raster = load_dem_to_numpy(project_folder)
    coordinates, min_height, max_height = setting_dem_coordinates(dem_data, dem_raster)
    layer = create_temp_vector_layer()
    layer = set_attribute_fields(layer)
    add_points(coordinates, layer)
    return reprojected_dem_mask, min_height, max_height

def calculate(H, J, angle):
    # Расчёты
    L = H * J  # Идеальное расстояние между лесополосами
    hop = L * math.tan(math.radians(angle))  # Шаг по высоте
    print(f"Идеальное расстояние L: {L}")
    print(f"Шаг по высоте hop: {hop}")

    return L, hop

def construct_isolines(reprojected_dem_mask, hop, max_height, project_folder):
    # Построение изолиний от max_z до min_z
    contours_output_path =  os.path.join(project_folder, "contours_output.shp")  # Укажите путь для сохранения изолиний

    try:
        result = processing.run("gdal:contour", {
            'INPUT': reprojected_dem_mask,
            'BAND': 1,
            'INTERVAL': hop,
            'FIELD_NAME': 'ELEV',
            'BASE': max_height,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        })
        print(f"Изолинии успешно созданы: {result}")
    except Exception as e:
        print(f"Ошибка при создании изолиний: {e}")

    return contours_output_path, result

def add_isolines_to_a_layer(contours_output_path, result):
    # Проверка, создан ли временный файл
    contours_output_path = result['OUTPUT']
    print(f"Временный файл изолиний создан: {contours_output_path}")

    if os.path.getsize(contours_output_path) == 0:
        print("Ошибка: Файл изолиний пуст.")

    # Загрузка изолиний в слой
    contours_layer = QgsVectorLayer(contours_output_path, 'Contours', 'ogr')

    print(f"CRS слоя: {contours_layer.crs().authid()}")
    print(f"CRS проекта: {QgsProject.instance().crs().authid()}")


    if not contours_layer.isValid():
        raise Exception("Не удалось загрузить слой изолиний.")
    QgsProject.instance().addMapLayer(contours_layer)

    return contours_layer

def filter_isoline(contours_layer):
    # Фильтрация изолиний по диапазону высот
    filtered_layer = QgsVectorLayer("LineString?crs=EPSG:3857", "Filtered Contours", "memory")  # Используем ту же CRS, что и DEM
    filtered_provider = filtered_layer.dataProvider()
    filtered_provider.addAttributes(contours_layer.fields())  # Копируем атрибуты из изолиний
    filtered_layer.updateFields()
    return filtered_provider, filtered_layer

def adding_isolines_by_height(contours_layer, min_height, max_height, filtered_provider, filtered_layer):
    # Добавляем только изолинии в диапазоне высот
    for feature in contours_layer.getFeatures():
        elevation = feature['ELEV']
        if min_height <= elevation <= max_height:  # Диапазон высот
            filtered_provider.addFeatures([feature])  # Добавляем отфильтрованные изолинии

    filtered_layer.updateExtents()  # Обновляем границы слоя
    QgsProject.instance().addMapLayer(filtered_layer)
    return

def create_isolines(reprojected_dem_mask, min_height, max_height, project_folder):
    # Параметры задачи
    H = 15  # Высота лесополосы
    J = 20  # Коэффициент влияния
    angle = 3  # Угол наклона в градусах

    L, hop = calculate(H, J, angle)
    contours_output_path, result = construct_isolines(reprojected_dem_mask, hop, max_height, project_folder)
    contours_layer = add_isolines_to_a_layer(contours_output_path, result)
    filtered_provider, filtered_layer = filter_isoline(contours_layer)
    adding_isolines_by_height(contours_layer, min_height, max_height, filtered_provider, filtered_layer)
    return filtered_layer

def add_forests_layer():
    # Создание слоя для лесополос
    forest_layer = QgsVectorLayer("LineString?crs=EPSG:3857", "Forest Belts", "memory")  # Совместимый CRS
    forest_provider = forest_layer.dataProvider()
    forest_provider.addAttributes([QgsField("Step", QVariant.Int)])  # Добавляем поле Step
    forest_layer.updateFields()

    return forest_layer, forest_provider

def generate_shades(base_color, steps):
    shades = []
    for i in range(steps):
        factor = i / (steps - 1)  # Нормализуем в диапазон [0, 1]
            
        # Увеличиваем интенсивность цвета для яркости
        r = int(base_color.red() * (1 - factor) + 255 * factor)  # Плавное увеличение от base к яркому
        g = int(base_color.green() * (1 - factor) + 255 * factor)
        b = int(base_color.blue() * (1 - factor) + 255 * factor)
        
        shades.append(QColor(r, g, b))
    return shades


def generate_color_pallete():
    # Базовый цвет (чистый красный)
    base_color = QColor(255, 0, 0)
    # Базовый цвет (чистый зеленый)
    base_color1 = QColor(0, 255, 0)
    # Базовый цвет (чистый синий)
    base_color2 = QColor(0, 0, 255)

    grad_steps = 255

    # Генерируем оттенки
    colors = generate_shades(base_color2, grad_steps) + generate_shades(base_color1, grad_steps) + generate_shades(base_color, grad_steps)  
    return colors

def add_forest_feature(filtered_layer, forest_provider,forest_layer,  colors):
    # Добавление лесополос с градиентом цветов
    step_index = 1

    categories = []  # Список категорий для рендера
    # Добавление лесополос
    for feature in filtered_layer.getFeatures():
        geometry = feature.geometry()
        if not geometry.isEmpty():  # Проверяем, что геометрия не пустая
            print(f"Feature {step_index}: Geometry is empty = {geometry.isEmpty()}")

            forest_feature = QgsFeature()
            forest_feature.setGeometry(geometry)
            forest_feature.setAttributes([step_index])  # Присваиваем шаг
            forest_provider.addFeatures([forest_feature])
            
            # Создаем символ с градиентным цветом
            color = colors[step_index % len(colors)]
            symbol = QgsSymbol.defaultSymbol(forest_layer.geometryType())
            symbol.setColor(color)

            # Добавляем категорию для рендера
            category = QgsRendererCategory(step_index, symbol, f"Лесополоса {step_index}")
            categories.append(category)

            
            step_index += 1
    return categories

def config_render(forest_layer, categories):
    # Настраиваем рендерер для слоя
    renderer = QgsCategorizedSymbolRenderer('Step', categories)
    forest_layer.setRenderer(renderer)

    forest_layer.updateExtents()  # Обновляем границы слоя
    QgsProject.instance().addMapLayer(forest_layer)
    return 

def check_forest_layers():
    # Получение слоев
    forest_layer = QgsProject.instance().mapLayersByName('Forest Belts')[0]
    selected_region_layer = QgsProject.instance().mapLayersByName('Selected Region')[0]
    
    # Проверка наличия слоев
    if not forest_layer or not selected_region_layer:
        print("Не найдены слои 'Forest Belts' или 'Selected Region'")
        exit()
    return forest_layer, selected_region_layer

def create_boundary_layer(selected_region_layer):
    # Создание слоя границ для замыкания
    boundary_result = processing.run("native:boundary", {
        'INPUT': selected_region_layer,
        'OUTPUT': 'memory:'
    })
    boundary_layer = boundary_result['OUTPUT']
    return boundary_layer

def setting_up_render(forest_layer):
        # Проверяем наличие поля 'Step' в слое лесополос
        if 'Step' not in [field.name() for field in forest_layer.fields()]:
            forest_layer.dataProvider().addAttributes([QgsField('Step', QVariant.Int)])
            forest_layer.updateFields()

        # Подготовка для рендеринга
        categories = []
        step_index = 0

        # Уникальные шаги (STEPS)
        unique_steps = sorted(set(f['Step'] for f in forest_layer.getFeatures()))
        return categories, step_index, unique_steps

def process_step(unique_steps, forest_layer, boundary_layer, colors, categories, step_index):
    for step in unique_steps:
        print(f"Обработка шага {step}")
        
        step_features = [f for f in forest_layer.getFeatures() if f['Step'] == step]
        
        for feature in step_features:
            geometry = feature.geometry()
            if geometry.isEmpty():
                print(f"Пропуск пустой геометрии на шаге {step}")
                continue
            
            # Поиск ближайшего контура
            shortest_distance = float('inf')
            best_contour = None
            
            for boundary_feature in boundary_layer.getFeatures():
                boundary_geom = boundary_feature.geometry()
                distance = geometry.distance(boundary_geom)
                if distance < shortest_distance:
                    shortest_distance = distance
                    best_contour = boundary_geom

            if best_contour:
                # Создание новой замкнутой геометрии
                closed_geometry = geometry.combine(best_contour)
                new_feature = QgsFeature(forest_layer.fields())
                new_feature.setGeometry(closed_geometry)
                new_feature.setAttributes(feature.attributes())
                
                forest_layer.dataProvider().addFeatures([new_feature])
                
                # Создание символа для рендеринга
                color = colors[step_index % len(colors)]
                symbol = QgsSymbol.defaultSymbol(forest_layer.geometryType())
                symbol.setColor(color)
                
                category = QgsRendererCategory(step, symbol, f"Лесополоса {step}")
                categories.append(category)

            step_index += 1

    return forest_layer

def render_forest_belts(categories, forest_layer):
    renderer = QgsCategorizedSymbolRenderer('Step', categories)
    forest_layer.setRenderer(renderer)
    forest_layer.updateExtents()  # Обновление слоя
    QgsProject.instance().addMapLayer(forest_layer)

def generate_forest_belts_layer(filtered_layer,):
    forest_layer, forest_provider = add_forests_layer()
    colors = generate_color_pallete()
    categories = add_forest_feature(filtered_layer, forest_provider,forest_layer, colors)
    config_render(forest_layer, categories)
    forest_layer, selected_region_layer = check_forest_layers()
    boundary_layer = create_boundary_layer(selected_region_layer)
    categories, step_index, unique_steps = setting_up_render(forest_layer)
    base_color = QColor(0, 255, 0)
    colors = generate_shades(base_color, 255)
    forest_layer = process_step(unique_steps, forest_layer, boundary_layer, colors, categories, step_index)
    render_forest_belts(categories, forest_layer)

    return

def create_forest_belts(project_folder):
    reprojected_dem_mask, min_height, max_height = create_points(project_folder) 
    filtered_layer = create_isolines(reprojected_dem_mask, min_height, max_height)
    generate_forest_belts_layer(filtered_layer)
    print("Лесополосы успешно замкнуты и добавлены в проект.")

    return

def forest(project_folder):
        get_main_def(project_folder)

        main_dem_file_path = os.path.join(project_folder, "srtm_output.tif")  # Путь к вашему DEM файлу
        masked_dem_output_path = os.path.join(project_folder, "masked_dem.tif")  # Путь к выходному файлу после обрезки

        process_dem_with_polygon(main_dem_file_path, masked_dem_output_path, project_folder)
        create_forest_belts()

        return
