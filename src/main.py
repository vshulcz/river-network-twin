from qgis.core import QgsProject
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtWidgets import QPushButton, QDialog, QVBoxLayout, QLabel, QCheckBox
import glob
import os
from qgis.utils import iface
from common import *
from river import river
from least_cost_path import least_cost_path_analysis
from forest import forest


class CustomDEMPlugin:

    project_folder = None

    def __init__(self, iface):
        self.iface = iface
        self.plugin_name = "RiverNETWORK"

    def initGui(self):
        #Для запуска плагина
        from qgis.PyQt.QtWidgets import QAction
        self.action = QAction(self.plugin_name, self.iface.mainWindow())
        self.action.triggered.connect(self.run_plugin)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&RiverNETWORK", self.action)

    def unload(self):
        #Удаление плагина
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&RiverNETWORK", self.action)

    def run_plugin(self):
        #Код плагина
        CustomDEMPlugin.project_folder = QFileDialog.getExistingDirectory(
            None, "Выберите рабочую папку"
        )
        if not CustomDEMPlugin.project_folder:
            QMessageBox.warning(None, "Ошибка", "Рабочая папка не выбрана. Работа плагина прекращена.")
            return
        
        # Создать папку "work" внутри выбранной папки
        CustomDEMPlugin.project_folder = os.path.join(CustomDEMPlugin.project_folder, "work/")
        if not os.path.exists(CustomDEMPlugin.project_folder):
            os.makedirs(CustomDEMPlugin.project_folder)
        QMessageBox.information(None, "Папка установлена", f"Рабочая папка: {CustomDEMPlugin.project_folder}")
        CustomDEMPlugin.run_programm()
    
    def show_layer_visibility_dialog():
        # Создать диалоговое окно
        dialog = QDialog()
        dialog.setWindowTitle("Выбор слоев")
        layout = QVBoxLayout()

        # Добавить лейбл
        label = QLabel("Выберите слои для отображения:")
        layout.addWidget(label)

        # Получить доступ к дереву слоев проекта
        root = QgsProject.instance().layerTreeRoot()

        # Создать чекбокс для каждого слоя
        checkboxes = {}
        for layer in QgsProject.instance().mapLayers().values():
            layer_tree_node = root.findLayer(layer.id())
            if layer_tree_node:  # Проверка наличия слоя в дереве
                checkbox = QCheckBox(layer.name())
                checkbox.setChecked(layer_tree_node.isVisible())  # Получить настоящую видимость
                layout.addWidget(checkbox)
                checkboxes[layer_tree_node] = checkbox

        # Добавить кнопку 
        apply_button = QPushButton("Применить")
        layout.addWidget(apply_button)

        def apply_layer_visibility():
            for layer_tree_node, checkbox in checkboxes.items():
                layer_tree_node.setItemVisibilityChecked(checkbox.isChecked())
            dialog.close()

        # Соединить кнопку подтверждения с функцией
        apply_button.clicked.connect(apply_layer_visibility)

        # Настройка макета и отображение диалогового окна
        dialog.setLayout(layout)
        dialog.exec_()

    # Определите основную функцию для диалога
    def show_choice_dialog():
        # Создание диалогового окна
        dialog = QDialog()
        dialog.setWindowTitle("Выбор функции")
        layout = QVBoxLayout()

        # Добавление метки
        label = QLabel("Что вы хотите сделать?")
        layout.addWidget(label)

        # Добавляйте кнопки для различных вариантов
        waterlines_button = QPushButton("Создать речную сеть")
        forest_belts_button = QPushButton("Создать лесополосы")
        cost_path_button = QPushButton("Вычислить путь наименьшей стоимости")

        layout.addWidget(waterlines_button)
        layout.addWidget(forest_belts_button)
        layout.addWidget(cost_path_button)

        # Определение действий для кнопок
        def create_waterlines():
            dialog.close()
            river(CustomDEMPlugin.project_folder)

        def create_forest_belts():
            dialog.close()
            forest(CustomDEMPlugin.project_folder)
        
        def create_cost_path():
            dialog.close()
            river(CustomDEMPlugin.project_folder)
            least_cost_path_analysis(CustomDEMPlugin.project_folder)

        # Свяжите кнопки с их действиями
        waterlines_button.clicked.connect(create_waterlines)
        forest_belts_button.clicked.connect(create_forest_belts)
        cost_path_button.clicked.connect(create_cost_path)

        # Настройка макета и отображение диалогового окна
        dialog.setLayout(layout)
        dialog.exec_()

    def clear_cache():
        project = QgsProject.instance()
        # Перебор всеч слоев в проекте
        for layer in project.mapLayers().values():
            # Проверка, содержит ли имя слоя слово "buffer" 
            if "buffer" in layer.name().lower():
                # Удалить слой из проекта
                project.removeMapLayer(layer)
        for file_pattern in ["*.shp", "*.shx", "*.dbf", "*.prj", "*.tif"]:
            for file_path in glob.glob(os.path.join(CustomDEMPlugin.project_folder, file_pattern)):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
                    
        # Очистка кэша холста карты
        iface.mapCanvas().refreshAllLayers()
        # Очистка кэша рендеринга
        QgsProject.instance().reloadAllLayers()
        return

    def prepare():
        project = QgsProject.instance()
        # Удалить все слои
        for layer in list(project.mapLayers().values()): 
            project.removeMapLayer(layer)

        # Удалить определенные форматы файлов (e.g., shapefiles, GeoTIFFs, etc.)
        for file_pattern in ["*.shp", "*.shx", "*.dbf", "*.prj", "*.tif"]:
            for file_path in glob.glob(os.path.join(CustomDEMPlugin.project_folder, file_pattern)):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
                    
        # Очистка кэша холста карты
        iface.mapCanvas().refreshAllLayers()

        # Очистка кэша рендеринга
        QgsProject.instance().reloadAllLayers()
        return

    def run_programm():
        #Подготовка к работе
        CustomDEMPlugin.clear_cache()
        CustomDEMPlugin.prepare()
        
        #Запуск диалогового окна
        CustomDEMPlugin.show_choice_dialog()
        CustomDEMPlugin.show_layer_visibility_dialog()
        
        return