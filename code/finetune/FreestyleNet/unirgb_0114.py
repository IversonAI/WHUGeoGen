# https://github.com/omipan/svl_adapter/blob/main/data_preparation.py
import os

import cv2
import numpy as np
from PIL import Image
import argparse
import vthread
from global_var import RAW_DATA, UNIRGB_DATA
from osgeo import gdal

Image.MAX_IMAGE_PIXELS = 1000000000

'''
1、调整处理的顺序   原始图像——RGB/Gray统一——分辨率切割——清洗——训练测试验证划分——json数据
2、新增灰度值处理
3、将labels_dict拓展为三个分类体系的颜色
'''

"""
mapping2rgb记录现有的所有类别及其对应的rgb颜色
mapping2rgb记录现有的所有类别及其对应的灰度颜色
"""
mapping2rgb = {
    # 土地利用类型
    'Unlabeled': (255, 255, 255),
    'Wetland': (240, 248, 255),
    'Cultivated_land': (250, 235, 215),
    'Plantation_land': (238, 130, 238),
    'Woodland': (127, 255, 212),
    'Grassland': (240, 255, 255),
    'Commercial_land': (245, 245, 220),
    'Industrial_land': (255, 228, 196),
    'Residential_land': (255, 235, 205),
    'Public_land': (0, 0, 255),
    'Special_land': (138, 43, 226),
    'Transportation_land': (165, 42, 42),
    'Water_land': (222, 184, 135),
    'Other_land': (95, 158, 160),
    # 地形分类（粗）
    'Road': (127, 255, 0),
    'Water': (210, 105, 30),
    'Vegetation': (255, 127, 80),
    'Settlement': (100, 149, 237),
    'Amenity': (255, 248, 220),
    'Soil': (220, 20, 60),
    'Landform': (0, 255, 255),
    'Object': (0, 0, 139),
    # 地形分类（细）
    'Railway': (0, 139, 139),
    'Motorway': (184, 134, 11),
    'Ordinary_highway': (169, 169, 169),
    'River': (0, 100, 0),
    'Lake': (72, 61, 139),
    'Reservoir': (189, 183, 107),
    'Sea': (139, 0, 139),
    'Island': (85, 107, 47),
    'Swamp': (255, 140, 0),
    'Farmland': (153, 50, 204),
    'Forest': (139, 0, 0),
    'Urban_green': (233, 150, 122),
    'Meadow': (143, 188, 143),
    'Space': (47, 79, 79),
    'Other_buildings': (119, 136, 153),
    'Port': (0, 206, 209),
    'Parking_lot': (148, 0, 211),
    'Airport': (255, 20, 147),
    'Overpass': (0, 191, 255),
    'Bridge': (105, 105, 105),
    'Railway_station': (176, 196, 222),
    'Crossroads': (30, 144, 255),
    'Roundabout': (178, 34, 34),
    'Heliport': (255, 250, 240),
    'Terminal': (34, 139, 34),
    'Lighthouse': (255, 0, 255),
    'Fuel': (220, 220, 220),
    'Solar_panel': (248, 248, 255),
    'Generator': (255, 215, 0),
    'Electricity_tower': (218, 165, 32),
    'Communications_tower': (128, 128, 128),
    'School': (0, 128, 0),
    'Sport_field': (173, 255, 47),
    'Water_tower': (255, 255, 224),
    'Amusementpark': (240, 255, 240),
    'Park': (255, 105, 180),
    'Grave_yard': (205, 92, 92),
    'Hospital': (75, 0, 130),
    'Church': (0, 255, 0),
    'Palace': (240, 230, 140),
    'Firestation': (230, 230, 250),
    'Policestation': (255, 240, 245),
    'Prison': (124, 252, 0),
    'Tank': (255, 250, 205),
    'Bare_land': (173, 216, 230),
    'Sand': (240, 128, 128),
    'Glacier': (224, 255, 255),
    'Mountain': (250, 250, 210),
    'Snowfield': (211, 211, 211),
    'Plane': (144, 238, 144),
    'Vehicle': (255, 182, 193),
    'Ship': (255, 160, 122),
    'Person': (32, 178, 170),
    'Animal': (135, 206, 250),
}
mapping2gray = {
    # 土地利用类型
    'Unlabeled': 0,
    'Wetland': 101,
    'Cultivated_land': 102,
    'Plantation_land': 103,
    'Woodland': 104,
    'Grassland': 105,
    'Commercial_land': 106,
    'Industrial_land': 107,
    'Residential_land': 108,
    'Public_land': 109,
    'Special_land': 110,
    'Transportation_land': 111,
    'Water_land': 112,
    'Other_land': 113,
    # 地形分类（粗）
    'Road': 201,
    'Water': 202,
    'Vegetation': 203,
    'Settlement': 204,
    'Amenity': 205,
    'Soil': 206,
    'Landform': 207,
    'Object': 208,
    # 地形分类（细）
    'Railway': 1,
    'Motorway': 2,
    'Ordinary_highway': 3,
    'River': 4,
    'Lake': 5,
    'Reservoir': 6,
    'Sea': 7,
    'Island': 8,
    'Swamp': 9,
    'Farmland': 10,
    'Forest': 11,
    'Urban_green': 12,
    'Meadow': 13,
    'Space': 14,
    'Other_buildings': 15,
    'Port': 16,
    'Parking_lot': 17,
    'Airport': 18,
    'Overpass': 19,
    'Bridge': 20,
    'Railway_station': 21,
    'Crossroads': 22,
    'Roundabout': 23,
    'Heliport': 24,
    'Terminal': 25,
    'Lighthouse': 26,
    'Fuel': 27,
    'Solar_panel': 28,
    'Generator': 29,
    'Electricity_tower': 30,
    'Communications_tower': 31,
    'School': 32,
    'Sport_field': 33,
    'Water_tower': 34,
    'Amusementpark': 35,
    'Park': 36,
    'Grave_yard': 37,
    'Hospital': 38,
    'Church': 39,
    'Palace': 40,
    'Firestation': 41,
    'Policestation': 42,
    'Prison': 43,
    'Tank': 44,
    'Bare_land': 45,
    'Sand': 46,
    'Glacier': 47,
    'Mountain': 48,
    'Snowfield': 49,
    'Plane': 50,
    'Vehicle': 51,
    'Ship': 52,
    'Person': 53,
    'Animal': 54
}

""" 
The temporary dictionary used to store details of the currently collected datasets 
like the number of classes, the image directory name, the RGB values corresponding 
to the original image categories etc
"""
"""
labels_dict现有数据集原始rgb及其现在对应的类别名称
"""
labels_dict = {
    'FLAIR':
        {'num_classes': 19, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['train','val'],  # ['2006','2012']
         'mapping':  # 即使是灰度值也先转换成rgb进行处理，所以这里写的是三个通道
             {
                 'landuse': {
                     (0, 0, 0): 'Unlabeled', (1, 1, 1): 'Residential_land', (2, 2, 2): 'Other_land',
                     (3, 3, 3): 'Other_land', (4, 4, 4): 'Other_land', (5, 5, 5): 'Water_land',
                     (6, 6, 6): 'Woodland', (7, 7, 7): 'Woodland', (8, 8, 8): 'Grassland',
                     (9, 9, 9): 'Plantation_land', (10, 10, 10): 'Plantation_land', (11, 11, 11): 'Cultivated_land',
                     (12, 12, 12): 'Cultivated_land', (13, 13, 13): 'Public_land',
                     (14, 14, 14): 'Other_land', (15, 15, 15): 'Other_land', (16, 16, 16): 'Other_land',
                     (17, 17, 17): 'Woodland', (18, 18, 18): 'Plantation_land', (19, 19, 19): 'Unlabeled',

                 },
                 'coarse': {
                     (0, 0, 0): 'Unlabeled', (1, 1, 1): 'Settlement', (2, 2, 2): 'Unlabeled',
                     (3, 3, 3): 'Unlabeled', (4, 4, 4): 'Soil', (5, 5, 5): 'Water',
                     (6, 6, 6): 'Vegetation', (7, 7, 7): 'Vegetation', (8, 8, 8): 'Vegetation',
                     (9, 9, 9): 'Vegetation', (10, 10, 10): 'Vegetation', (11, 11, 11): 'Vegetation',
                     (12, 12, 12): 'Vegetation', (13, 13, 13): 'Amenity',
                     (14, 14, 14): 'Landform', (15, 15, 15): 'Unlabeled', (16, 16, 16): 'Unlabeled',
                     (17, 17, 17): 'Vegetation', (18, 18, 18): 'Vegetation', (19, 19, 19): 'Unlabeled',
                 },
                 'fine': {
                     (0, 0, 0): 'Unlabeled', (1, 1, 1): 'Other_buildings', (2, 2, 2): 'Unlabeled',
                     (3, 3, 3): 'Unlabeled', (4, 4, 4): 'Bare_land', (5, 5, 5): 'Unlabeled',
                     (6, 6, 6): 'Forest', (7, 7, 7): 'Forest', (8, 8, 8): 'Farmland',
                     (9, 9, 9): 'Meadow', (10, 10, 10): 'Farmland', (11, 11, 11): 'Farmland',
                     (12, 12, 12): 'Sport_field', (13, 13, 13): 'Snowfield',
                     (14, 14, 14): 'Unlabeled', (15, 15, 15): 'Unlabeled', (16, 16, 16): 'Other_land',
                     (17, 17, 17): 'Forest', (18, 18, 18): 'Unlabeled', (19, 19, 19): 'Unlabeled',
                 }
             }
         },
    'iSAID':
        {'num_classes': 16, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['train', 'val'],
         'mapping':
             {
                 'landuse': {
                     (0, 0, 63): 'Unlabeled', (0, 63, 0): 'Public_land',
                     (0, 63, 63): 'Industrial_land', (0, 63, 127): 'Public_land',
                     (0, 63, 191): 'Public_land', (0, 63, 255): 'Public_land', (0, 127, 63): 'Transportation_land',
                     (0, 127, 127): 'Unlabeled', (0, 0, 127): 'Unlabeled', (0, 0, 191): 'Unlabeled',
                     (0, 0, 255): 'Public_land', (0, 191, 127): 'Public_land', (0, 127, 191): 'Public_land',
                     (0, 127, 255): 'Unlabeled', (0, 100, 155): 'Transportation_land', (0, 0, 0): 'Unlabeled',

                 },
                 'coarse': {
                     (0, 0, 63): 'Object', (0, 63, 0): 'Amenity',
                     (0, 63, 63): 'Amenity', (0, 63, 127): 'Amenity',
                     (0, 63, 191): 'Amenity', (0, 63, 255): 'Amenity', (0, 127, 63): 'Amenity',
                     (0, 127, 127): 'Object', (0, 0, 127): 'Object', (0, 0, 191): 'Object',
                     (0, 0, 255): 'Amenity', (0, 191, 127): 'Amenity', (0, 127, 191): 'Amenity',
                     (0, 127, 255): 'Object', (0, 100, 155): 'Amenity', (0, 0, 0): 'Unlabeled',
                 },
                 'fine': {
                     (0, 0, 63): 'Ship', (0, 63, 0): 'Sport_field',
                     (0, 63, 63): 'Tank', (0, 63, 127): 'Sport_field',
                     (0, 63, 191): 'Sport_field', (0, 63, 255): 'Sport_field', (0, 127, 63): 'Bridge',
                     (0, 127, 127): 'Vehicle', (0, 0, 127): 'Vehicle', (0, 0, 191): 'Plane',
                     (0, 0, 255): 'Sport_field', (0, 191, 127): 'Roundabout', (0, 127, 191): 'Sport_field',
                     (0, 127, 255): 'Plane', (0, 100, 155): 'Port', (0, 0, 0): 'Unlabeled',
                 }
             }
         },
    'loveDA':
        {'num_classes': 7, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['train', 'val'],#['val'],
         'mapping':  # 即使是灰度值也先转换成rgb进行处理，所以这里写的是三个通道
             {
                 'landuse': {
                     (1, 1, 1): 'Unlabeled', (2, 2, 2): 'Residential_land', (3, 3, 3): 'Unlabeled',
                     (4, 4, 4): 'Water_land',
                     (5, 5, 5): 'Other_land', (6, 6, 6): 'Woodland', (7, 7, 7): 'Cultivated_land'
                 },
                 'coarse': {
                     (1, 1, 1): 'Unlabeled', (2, 2, 2): 'Settlement', (3, 3, 3): 'Road', (4, 4, 4): 'Water',
                     (5, 5, 5): 'Soil', (6, 6, 6): 'Vegetation', (7, 7, 7): 'Vegetation'
                 },
                 'fine': {
                     (1, 1, 1): 'Unlabeled', (2, 2, 2): 'Other_buildings', (3, 3, 3): 'Unlabeled',
                     (4, 4, 4): 'Unlabeled',
                     (5, 5, 5): 'Bare_land', (6, 6, 6): 'Forest', (7, 7, 7): 'Farmland'
                 }
             }
         },
    'HRSCD':
        {'num_classes': 6, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['2012'],  # ['2006','2012']
         'mapping':  # 即使是灰度值也先转换成rgb进行处理，所以这里写的是三个通道
             {
                 'landuse': {
                     (0, 0, 0): 'Unlabeled', (1, 1, 1): 'Other_land', (2, 2, 2): 'Cultivated_land',
                     (3, 3, 3): 'Woodland', (4, 4, 4): 'Wetland', (5, 5, 5): 'Water_land'
                 },
                 'coarse': {
                     (0, 0, 0): 'Unlabeled', (1, 1, 1): 'Settlement', (2, 2, 2): 'Vegetation',
                     (3, 3, 3): 'Vegetation', (4, 4, 4): 'Water', (5, 5, 5): 'Water'
                 },
                 'fine': {
                     (0, 0, 0): 'Unlabeled', (1, 1, 1): 'Space', (2, 2, 2): 'Farmland',
                     (3, 3, 3): 'Forest', (4, 4, 4): 'Swamp', (5, 5, 5): 'Unlabeled'
                 }
             }
         },
    'enviroatlas_landcover_nlcd':
        {'num_classes': 21, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['all'],  # ['2006','2012']
         'mapping':  # 即使是灰度值也先转换成rgb进行处理，所以这里写的是三个通道
             {
                 'landuse': {
                     (1, 1, 1): 'Water_land', (2, 2, 2): 'Unlabeled', (3, 3, 3): 'Residential_land',
                     (4, 4, 4): 'Industrial_land', (5, 5, 5): 'Commercial_land', (6, 6, 6): 'Other_land',
                     (7, 7, 7): 'Woodland', (8, 8, 8): 'Woodland', (9, 9, 9): 'Woodland',
                     (10, 10, 10): 'Woodland', (11, 11, 11): 'Grassland', (12, 12, 12): 'Grassland',
                     (13, 13, 13): 'Cultivated_land', (14, 14, 14): 'Wetland', (15, 15, 15): 'Wetland',
                     (16, 16, 16): 'Unlabeled', (17, 17, 17): 'Unlabeled', (18, 18, 18): 'Unlabeled',
                     (19, 19, 19): 'Unlabeled', (20, 20, 20): 'Unlabeled', (21, 21, 21): 'Unlabeled'
                 },
                 'coarse': {
                     (1, 1, 1): 'Water', (2, 2, 2): 'Settlement', (3, 3, 3): 'Settlement',
                     (4, 4, 4): 'Settlement', (5, 5, 5): 'Settlement', (6, 6, 6): 'Soil',
                     (7, 7, 7): 'Vegetation', (8, 8, 8): 'Vegetation', (9, 9, 9): 'Vegetation',
                     (10, 10, 10): 'Vegetation', (11, 11, 11): 'Vegetation', (12, 12, 12): 'Vegetation',
                     (13, 13, 13): 'Vegetation', (14, 14, 14): 'Vegetation', (15, 15, 15): 'Vegetation',
                     (16, 16, 16): 'Unlabeled', (17, 17, 17): 'Unlabeled', (18, 18, 18): 'Unlabeled',
                     (19, 19, 19): 'Unlabeled', (20, 20, 20): 'Unlabeled', (21, 21, 21): 'Unlabeled'
                 },
                 'fine': {
                     (1, 1, 1): 'Unlabeled', (2, 2, 2): 'Other_buildings', (3, 3, 3): 'Other_buildings',
                     (4, 4, 4): 'Other_buildings', (5, 5, 5): 'Other_buildings', (6, 6, 6): 'Bare_land',
                     (7, 7, 7): 'Forest', (8, 8, 8): 'Forest', (9, 9, 9): 'Forest',
                     (10, 10, 10): 'Forest', (11, 11, 11): 'Meadow', (12, 12, 12): 'Meadow',
                     (13, 13, 13): 'Farmland', (14, 14, 14): 'Unlabeled', (15, 15, 15): 'Unlabeled',
                     (16, 16, 16): 'Unlabeled', (17, 17, 17): 'Unlabeled', (18, 18, 18): 'Unlabeled',
                     (19, 19, 19): 'Unlabeled', (20, 20, 20): 'Unlabeled', (21, 21, 21): 'Unlabeled'
                 }
             }
         },
    'enviroatlas_landcover_highres':
        {'num_classes': 8, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['all'],  # ['2006','2012']
         'mapping':  # 即使是灰度值也先转换成rgb进行处理，所以这里写的是三个通道
             {
                 'landuse': {
                     (1, 1, 1): 'Water_land', (2, 2, 2): 'Residential_land', (3, 3, 3): 'Other_land',
                     (4, 4, 4): 'Woodland', (5, 5, 5): 'Vegetation', (6, 6, 6): 'Grassland',
                     (7, 7, 7): 'Cultivated_land', (8, 8, 8): 'Wetland'
                 },
                 'coarse': {
                     (1, 1, 1): 'Water', (2, 2, 2): 'Settlement', (3, 3, 3): 'Soil',
                     (4, 4, 4): 'Vegetation', (5, 5, 5): 'Vegetation', (6, 6, 6): 'Vegetation',
                     (7, 7, 7): 'Vegetation', (8, 8, 8): 'Water'
                 },
                 'fine': {
                     (1, 1, 1): 'Unlabeled', (2, 2, 2): 'Other_buildings', (3, 3, 3): 'Bare_land',
                     (4, 4, 4): 'Forest', (5, 5, 5): 'Forest', (6, 6, 6): 'Meadow',
                     (7, 7, 7): 'Farmland', (8, 8, 8): 'Swamp'
                 }
             }
         },
    'Geonrw':
        {'num_classes': 10, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['all'],  # ['2006','2012']
         'mapping':  # 即使是灰度值也先转换成rgb进行处理，所以这里写的是三个通道
             {
                 'landuse': {
                     (1, 1, 1): 'Woodland', (2, 2, 2): 'Water_land', (3, 3, 3): 'Cultivated_land',
                     (4, 4, 4): 'Residential_land', (5, 5, 5): 'Grassland', (6, 6, 6): 'Unlabeled',
                     (7, 7, 7): 'Transportation_land', (8, 8, 8): 'Unlabeled', (9, 9, 9): 'Unlabeled',
                     (10, 10, 10): 'Residential_land'
                 },
                 'coarse': {
                     (1, 1, 1): 'Vegetation', (2, 2, 2): 'Water', (3, 3, 3): 'Vegetation',
                     (4, 4, 4): 'Unlabeled', (5, 5, 5): 'Vegetation', (6, 6, 6): 'Road',
                     (7, 7, 7): 'Road', (8, 8, 8): 'Amenity', (9, 9, 9): 'Road',
                     (10, 10, 10): 'Settlement'
                 },
                 'fine': {
                     (1, 1, 1): 'Forest', (2, 2, 2): 'Unlabeled', (3, 3, 3): 'Farmland',
                     (4, 4, 4): 'Unlabeled', (5, 5, 5): 'Meadow', (6, 6, 6): 'Railway',
                     (7, 7, 7): 'Unlabeled', (8, 8, 8): 'Unlabeled', (9, 9, 9): 'Unlabeled',
                     (10, 10, 10): 'Other_buildings'
                 }
             }
         },
    'cvpr_chesapeake_landcover_lc':
        {'num_classes': 7, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['all'],
         'mapping':  # 即使是灰度值也先转换成rgb进行处理，所以这里写的是三个通道
             {
                 'landuse': {
                     (1, 1, 1): 'Water_land', (2, 2, 2): 'Woodland', (3, 3, 3): 'Cultivated_land',
                     (4, 4, 4): 'Other_land', (5, 5, 5): 'Unlabeled', (6, 6, 6): 'Unlabeled',
                     (15, 15, 15): 'Unlabeled'
                 },
                 'coarse': {
                     (1, 1, 1): 'Water', (2, 2, 2): 'Vegetation', (3, 3, 3): 'Vegetation',
                     (4, 4, 4): 'Soil', (5, 5, 5): 'Unlabeled', (6, 6, 6): 'Road',
                     (15, 15, 15): 'Unlabeled'
                 },
                 'fine': {
                     (1, 1, 1): 'Unlabeled', (2, 2, 2): 'Forest', (3, 3, 3): 'Unlabeled',
                     (4, 4, 4): 'Bare_land', (5, 5, 5): 'Unlabeled', (6, 6, 6): 'Road',
                     (15, 15, 15): 'Unlabeled'
                 }
             }
         },
    'cvpr_chesapeake_landcover_nlcd':
        {'num_classes': 21, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['all'],  # ['2006','2012']
         'mapping':  # 即使是灰度值也先转换成rgb进行处理，所以这里写的是三个通道
             {
                 'landuse': {
                     (0, 0, 0): 'Unlabeled', (11, 11, 11): 'Water_land', (12, 12, 12): 'Other_land',
                     (21, 21, 21): 'Unlabeled', (22, 22, 22): 'Residential_land', (23, 23, 23): 'Industrial_land',
                     (24, 24, 24): 'Commercial_land', (31, 31, 31): 'Other_land', (41, 41, 41): 'Woodland',
                     (42, 42, 42): 'Woodland', (43, 43, 43): 'Woodland', (51, 51, 51): 'Woodland',
                     (52, 52, 52): 'Woodland', (71, 71, 71): 'Grassland', (72, 72, 72): 'Grassland',
                     (73, 73, 73): 'Grassland', (74, 74, 74): 'Grassland', (81, 81, 81): 'Grassland',
                     (82, 82, 82): 'Cultivated_land', (90, 90, 90): 'Wetland', (95, 95, 95): 'Wetland',
                 },
                 'coarse': {
                     (0, 0, 0): 'Unlabeled', (11, 11, 11): 'Water', (12, 12, 12): 'Landform',
                     (21, 21, 21): 'Settlement', (22, 22, 22): 'Settlement', (23, 23, 23): 'Settlement',
                     (24, 24, 24): 'Settlement', (31, 31, 31): 'Soil', (41, 41, 41): 'Vegetation',
                     (42, 42, 42): 'Vegetation', (43, 43, 43): 'Vegetation', (51, 51, 51): 'Vegetation',
                     (52, 52, 52): 'Vegetation', (71, 71, 71): 'Vegetation', (72, 72, 72): 'Vegetation',
                     (73, 73, 73): 'Vegetation', (74, 74, 74): 'Vegetation', (81, 81, 81): 'Vegetation',
                     (82, 82, 82): 'Vegetation', (90, 90, 90): 'Vegetation', (95, 95, 95): 'Vegetation',
                 },
                 'fine': {
                     (0, 0, 0): 'Unlabeled', (11, 11, 11): 'Unlabeled', (12, 12, 12): 'Snowfield',
                     (21, 21, 21): 'Other_buildings', (22, 22, 22): 'Other_buildings', (23, 23, 23): 'Other_buildings',
                     (24, 24, 24): 'Other_buildings', (31, 31, 31): 'Bare_land', (41, 41, 41): 'Forest',
                     (42, 42, 42): 'Forest', (43, 43, 43): 'Forest', (51, 51, 51): 'Forest',
                     (52, 52, 52): 'Forest', (71, 71, 71): 'Meadow', (72, 72, 72): 'Meadow',
                     (73, 73, 73): 'Unlabeled', (74, 74, 74): 'Unlabeled', (81, 81, 81): 'Meadow',
                     (82, 82, 82): 'Farmland', (90, 90, 90): 'Unlabeled', (95, 95, 95): 'Unlabeled',
                 }
             }
         },
    'DroneDeploy':
        {'num_classes': 7, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['all'],
         'mapping':  # 即使是灰度值也先转换成rgb进行处理，所以这里写的是三个通道
             {
                 'landuse': {
                     (230, 25, 75): 'Residential_land', (145, 30, 180): 'Unlabeled', (60, 180, 75): 'Cultivated_land',
                     (245, 130, 48): 'Water_land', (255, 255, 255): 'Unlabeled', (0, 130, 200): 'Unlabeled',
                     (255, 0, 255): 'Unlabeled'
                 },
                 'coarse': {
                     (230, 25, 75): 'Settlement', (145, 30, 180): 'Unlabeled', (60, 180, 75): 'Vegetation',
                     (245, 130, 48): 'Water', (255, 255, 255): 'Settlement', (0, 130, 200): 'Object',
                     (255, 0, 255): 'Unlabeled'
                 },
                 'fine': {
                     (230, 25, 75): 'Other_buildings', (145, 30, 180): 'Unlabeled', (60, 180, 75): 'Unlabeled',
                     (245, 130, 48): 'Unlabeled', (255, 255, 255): 'Space', (0, 130, 200): 'Vehicle',
                     (255, 0, 255): 'Unlabeled'
                 }
             }

         },
    'MiniFrance':
        {'num_classes': 16, 'label_dir': 'labels/', 'image_dir': 'images/', 'split_names': ['all'],
         'mapping':  # 即使是灰度值也先转换成rgb进行处理，所以这里写的是三个通道
             {
                 'landuse': {
                     (0, 0, 0): 'Unlabeled', (1, 1, 1): 'Residential_land', (2, 2, 2): 'Unlabeled',
                     (3, 3, 3): 'Industrial_land',
                     (4, 4, 4): 'Grassland', (5, 5, 5): 'Cultivated_land', (6, 6, 6): 'Cultivated_land',
                     (7, 7, 7): 'Grassland', (8, 8, 8): 'Cultivated_land', (9, 9, 9): 'Cultivated_land',
                     (10, 10, 10): 'Woodland', (11, 11, 11): 'Cultivated_land', (12, 12, 12): 'Other_land',
                     (13, 13, 13): 'Wetland', (14, 14, 14): 'Water_land', (15, 15, 15): 'Unlabeled',
                 },
                 'coarse': {
                     (0, 0, 0): 'Unlabeled', (1, 1, 1): 'Settlement', (2, 2, 2): 'Settlement', (3, 3, 3): 'Unlabeled',
                     (4, 4, 4): 'Vegetation', (5, 5, 5): 'Vegetation', (6, 6, 6): 'Vegetation',
                     (7, 7, 7): 'Vegetation', (8, 8, 8): 'Vegetation', (9, 9, 9): 'Vegetation',
                     (10, 10, 10): 'Vegetation', (11, 11, 11): 'Vegetation', (12, 12, 12): 'Settlement',
                     (13, 13, 13): 'Water', (14, 14, 14): 'Water', (15, 15, 15): 'Unlabeled',
                 },
                 'fine': {
                     (0, 0, 0): 'Unlabeled', (1, 1, 1): 'Other_buildings', (2, 2, 2): 'Unlabeled',
                     (3, 3, 3): 'Unlabeled',
                     (4, 4, 4): 'Unlabeled', (5, 5, 5): 'Farmland', (6, 6, 6): 'Farmland',
                     (7, 7, 7): 'Meadow', (8, 8, 8): 'Farmland', (9, 9, 9): 'Farmland',
                     (10, 10, 10): 'Forest', (11, 11, 11): 'Meadow', (12, 12, 12): 'Space',
                     (13, 13, 13): 'Swamp', (14, 14, 14): 'Unlabeled', (15, 15, 15): 'Unlabeled',
                 }
             }
         }
}


def get_unirgb_label(dataset, label_path):
    '''
    输入处理的数据集和语义图的路径，输出改名后的语义图路径字典
    :param dataset:     处理的目标数据集
    :param label_path:  现有语义图路径
    :return:            改名后的语义图路径的字典
    '''
    result_paths = {
        "rgb_landuse": "",
        "rgb_coarse": "",
        "rgb_fine": "",
        "gray_landuse": "",
        "gray_coarse": "",
        "gray_fine": ""
    }
    if dataset == 'iSAID':
        # label_path = 'F:/DATA1/0RAWDATA/iSAID/labels/val/P0060_instance_color_RGB.png'
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-1] = label_path_parts_raw[-1].replace("_instance_color_RGB", "")
        label_path_parts[-5] = UNIRGB_DATA
        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    elif dataset == 'FLAIR':
        # label_path ='/home/wyd/dl/data/0RAWDATA/FLAIR/labels/train/MSK_000002.tif'
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-5] = UNIRGB_DATA
        label_path_parts[-1] = label_path_parts_raw[-1].replace('MSK_', '')
        label_path_parts[-1] = label_path_parts_raw[-1].replace('tif', 'png')

        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    elif dataset == 'loveDA':
        # label_path = 'D:/bqp_home/Data_process/loveDA/train/1234.png'
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-5] = UNIRGB_DATA
        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    elif dataset == 'HRSCD':
        # label_path = 'F:/Codefield/Data_process/0RAWDATA/HRSCD/labels/2006/14-2012-0415-6890-LA93-0M50-E080.tif'
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-5] = UNIRGB_DATA
        label_path_parts[-1] = label_path_parts_raw[-1].replace('tif', 'png')
        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    elif dataset == 'enviroatlas_landcover_nlcd':
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-5] = UNIRGB_DATA
        label_path_parts[-1] = label_path_parts_raw[-1].replace('tif', 'png')
        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    elif dataset == 'enviroatlas_landcover_highres':
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-5] = UNIRGB_DATA
        label_path_parts[-1] = label_path_parts_raw[-1].replace('tif', 'png')
        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    elif dataset == 'Geonrw':
        # label_path = 'F:\AIGC\Data_process\0RAWDATA\Geonrw\labels/286_5630.jp2'
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-5] = UNIRGB_DATA
        label_path_parts[-1] = label_path_parts_raw[-1].replace('jp2', 'png')
        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    elif dataset == 'cvpr_chesapeake_landcover_nlcd':
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-5] = UNIRGB_DATA
        label_path_parts[-1] = label_path_parts_raw[-1].replace('tif', 'png')
        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    elif dataset == 'cvpr_chesapeake_landcover_lc':
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-5] = UNIRGB_DATA
        label_path_parts[-1] = label_path_parts_raw[-1].replace('tif', 'png')
        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    elif dataset == 'DroneDeploy':
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-5] = UNIRGB_DATA
        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    elif dataset == 'MiniFrance':
        label_path_parts_raw = label_path.split("/")
        label_path_parts = label_path_parts_raw.copy()
        label_path_parts[-5] = UNIRGB_DATA
        label_path_parts[-1] = label_path_parts_raw[-1].replace('.tif', '.png')
        for key in result_paths.keys():
            label_path_parts[-2] = label_path_parts_raw[-2] + "_" + key
            result_paths[key] = "/".join(label_path_parts)
    print(result_paths)
    return result_paths



def get_unirgb_image(dataset, image_path):
    '''
    输入处理的数据集和遥感图的路径，输出改名后的遥感图路径字典
    :param dataset:     处理的目标数据集
    :param image_path:  现有遥感图路径
    :return:            改名后的遥感图路径
    '''
    result_path = ""
    if dataset == 'iSAID':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('png', 'jpg')
    elif dataset == 'FLAIR':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('png', 'jpg')
    elif dataset == 'loveDA':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('png', 'jpg')
    elif dataset == 'HRSCD':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('tif', 'jpg')
    elif dataset == 'enviroatlas_landcover_nlcd':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('tif', 'jpg')
    elif dataset == 'enviroatlas_landcover_highres':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('tif', 'jpg')
    elif dataset == 'Geonrw':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('jp2', 'jpg')
    elif dataset == 'cvpr_chesapeake_landcover_nlcd':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('tif', 'jpg')
    elif dataset == 'cvpr_chesapeake_landcover_lc':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('tif', 'jpg')
    elif dataset == 'DroneDeploy':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('tif', 'jpg')
    elif dataset == 'MiniFrance':
        result_path = image_path.replace(RAW_DATA, UNIRGB_DATA).replace('tif', 'jpg')
    print(result_path)
    return result_path


def read_image_file_path(dataset, image_file_path):
    '''
    用cv2库读取image_file_path文件
    :param dataset              处理的目标文件夹
    :param image_file_path:     文件路径
    :return:                    读取的image对象
    '''
    # # 以RGB图像的形式打开待处理文件，conver("RGB")的作用是将所有格式的图像转换为RGB三通道的
    # raw_image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    # # 检查图像通道数
    # if len(raw_image.shape) == 2:  # 灰度图
    #     # 将灰度图转为RGB
    #     raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
    # raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    raw_image = ""
    if dataset == 'iSAID':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    elif dataset == 'FLAIR':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        if len(raw_image.shape) == 2:  # 灰度图
            # # 将灰度图转为RGB
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        raw_image = raw_image.astype(np.uint8)
    elif dataset == 'loveDA':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        if len(raw_image.shape) == 2:  # 灰度图
            # # 将灰度图转为RGB
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
    elif dataset == 'HRSCD':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        if len(raw_image.shape) == 2:  # 灰度图
            # # 将灰度图转为RGB
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        raw_image = raw_image.astype(np.uint8)
    elif dataset == 'enviroatlas_landcover_nlcd':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        if len(raw_image.shape) == 2:  # 灰度图
            # # 将灰度图转为RGB
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        raw_image = raw_image.astype(np.uint8)
    elif dataset == 'enviroatlas_landcover_highres':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        if len(raw_image.shape) == 2:  # 灰度图
            # # 将灰度图转为RGB
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        raw_image = raw_image.astype(np.uint8)
    elif dataset == 'Geonrw':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        if len(raw_image.shape) == 2:  # 灰度图
            # # 将灰度图转为RGB
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
    elif dataset == 'cvpr_chesapeake_landcover_nlcd':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        if len(raw_image.shape) == 2:  # 灰度图
            # # 将灰度图转为RGB
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        raw_image = raw_image.astype(np.uint8)
    elif dataset == 'cvpr_chesapeake_landcover_lc':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        if len(raw_image.shape) == 2:  # 灰度图
            # # 将灰度图转为RGB
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        raw_image = raw_image.astype(np.uint8)
    elif dataset == 'DroneDeploy':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    elif dataset == 'MiniFrance':
        raw_image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        if len(raw_image.shape) == 2:  # 灰度图
            # # 将灰度图转为RGB
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        raw_image = raw_image.astype(np.uint8)
    return raw_image

# @vthread.thread(10)  # 多线程处理 by dailei
def main(args):
    # ---------------------------一、准备工作-------------------------------------------
    # 获取0RAWDATA中目标数据集的完整路径
    input_dir = '{}/{}/{}/'.format(args['root_data_dir'], RAW_DATA, args['dataset'])
    print("正在处理的数据集为：" + input_dir)
    label_dir = labels_dict[args['dataset']]['label_dir']  # 获取目标数据集内的语义图目录 --'labels/'
    image_dir = labels_dict[args['dataset']]['image_dir']  # 获取目标数据集内的遥感图目录
    # 在1UNIRGBDATA文件夹中创建一个目标数据集对应的树结构
    output_dir = '{}/{}/{}'.format(args['root_data_dir'], UNIRGB_DATA, args['dataset'])
    if os.path.exists(output_dir):
        print("已经存在：" + output_dir)
    else:
        os.mkdir(output_dir)
    if os.path.exists(output_dir + "/" + label_dir):
        print("已经存在：" + output_dir + "/" + label_dir)
    else:
        os.mkdir(output_dir + "/" + label_dir)
    if os.path.exists(output_dir + "/" + image_dir):
        print("已经存在：" + output_dir + "/" + image_dir)
    else:
        os.mkdir(output_dir + "/" + image_dir)
    for split_file in labels_dict[args['dataset']]['split_names']:  # ['train','val','test']:
        if not os.path.exists(output_dir + "/" + label_dir + "/" + split_file + "_rgb_landuse"):
            os.mkdir(output_dir + "/" + label_dir + "/" + split_file + "_rgb_landuse")
        if not os.path.exists(output_dir + "/" + label_dir + "/" + split_file + "_rgb_coarse"):
            os.mkdir(output_dir + "/" + label_dir + "/" + split_file + "_rgb_coarse")
        if not os.path.exists(output_dir + "/" + label_dir + "/" + split_file + "_rgb_fine"):
            os.mkdir(output_dir + "/" + label_dir + "/" + split_file + "_rgb_fine")
        if not os.path.exists(output_dir + "/" + label_dir + "/" + split_file + "_gray_landuse"):
            os.mkdir(output_dir + "/" + label_dir + "/" + split_file + "_gray_landuse")
        if not os.path.exists(output_dir + "/" + label_dir + "/" + split_file + "_gray_coarse"):
            os.mkdir(output_dir + "/" + label_dir + "/" + split_file + "_gray_coarse")
        if not os.path.exists(output_dir + "/" + label_dir + "/" + split_file + "_gray_fine"):
            os.mkdir(output_dir + "/" + label_dir + "/" + split_file + "_gray_fine")
        if not os.path.exists(output_dir + "/" + image_dir + "/" + split_file):
            os.mkdir(output_dir + "/" + image_dir + "/" + split_file)
    # ---------------------------二、处理工作（语义图像部分）-------------------------------------------
    image_files = os.path.join(input_dir, label_dir)
    for split_file in labels_dict[args['dataset']]['split_names']:  # ['train','val','test']:
        # 获取处理图像的直接父目录
        image_path = os.path.join(image_files,
                                  split_file)  # 根目录/iSAID/labels/train、根目录/iSAID/labels/val、根目录/iSAID/labels/test分别处理
        '''
        方法一：用PIL库进行处理，对每一个像素依次进行处理
        # 对每一张图片分别进行处理
        for image_file in os.listdir(image_path):
            image_file_path = os.path.join(image_path, image_file)  # 每一张语义图片的完整路径
            print("正在处理：" + image_file_path)
            if os.path.isfile(image_file_path):   # 判断待处理的图像是否是文件而不是文件夹
                # 以RGB图像的形式打开待处理文件，conver("RGB")的作用是将所有格式的图像转换为RGB三通道的
                raw_image = Image.open(image_file_path).convert("RGB")  # 这里不适用with打开的方式是因为一张原始的语义图像要保存为六张不同的语义图片，不使用with方便拷贝
                width, height = raw_image.size  # 获取图像长宽像素数量以方便下面进行遍历
                rgb_landuse_image = raw_image.copy()              # ①处理转换rgb的landuse分类
                rgb_coarse_image = raw_image.copy()         # ②处理转换rgb的landform_rough分类
                rgb_fine_image = raw_image.copy()      # ③处理转换rgb的landform_detailed分类
                gray_landuse_image = raw_image.copy()             # ④处理转换gray的landuse分类
                gray_coarse_image = raw_image.copy()        # ⑤处理转换gray的landfrom_rough分类
                gray_fine_image = raw_image.copy()     # ⑥处理转换gray的landform_detailed分类
                # 根据原有像素RGB对应的类别赋值转换后的RGB
                for x in range(width):
                    for y in range(height):
                        pixel = raw_image.getpixel((x, y))   # 获取每一个坐标的像素值
                        rgb_landuse_image.putpixel((x, y), mapping2rgb[labels_dict[args['dataset']]['mapping']['landuse'][pixel]])
                        rgb_coarse_image.putpixel((x, y), mapping2rgb[labels_dict[args['dataset']]['mapping']['coarse'][pixel]])
                        rgb_fine_image.putpixel((x, y), mapping2rgb[labels_dict[args['dataset']]['mapping']['fine'][pixel]])
                        gray_landuse_image.putpixel((x, y), (mapping2gray[labels_dict[args['dataset']]['mapping']['landuse'][pixel]],) * 3)
                        gray_coarse_image.putpixel((x, y), (mapping2gray[labels_dict[args['dataset']]['mapping']['coarse'][pixel]],) * 3)
                        gray_fine_image.putpixel((x, y), (mapping2gray[labels_dict[args['dataset']]['mapping']['fine'][pixel]],) * 3)
                # 设计六个处理后的图像保存路径
                rgb_landuse_path = output_dir + "/" + label_dir + "/" + split_file + '_rgb_landuse' + "/" + os.path.splitext(image_file)[0] + '_rgb_landuse' + '.png'
                rgb_coarse_path = output_dir + "/" + label_dir + "/" + split_file + '_rgb_coarse' + "/" + os.path.splitext(image_file)[0] + '_rgb_coarse' + '.png'
                rgb_fine_path = output_dir + "/" + label_dir + "/" + split_file + '_rgb_fine' + "/" + os.path.splitext(image_file)[0] + '_rgb_fine' + '.png'
                gray_landuse_path = output_dir + "/" + label_dir + "/" + split_file + '_gray_landuse' + "/" + os.path.splitext(image_file)[0] + '_gray_landuse' + '.png'
                gray_coarse_path = output_dir + "/" + label_dir + "/" + split_file + '_gray_coarse' + "/" + os.path.splitext(image_file)[0] + '_gray_coarse' + '.png'
                gray_fine_path = output_dir + "/" + label_dir + "/" + split_file + '_gray_fine' + "/" + os.path.splitext(image_file)[0] + '_gray_fine' + '.png'
                # 如果此路径已经存在，则说明之前已经处理过这张语义图像
                if os.path.exists(rgb_landuse_path):
                    print("已经存在：" + rgb_landuse_path)
                else:
                    rgb_landuse_image.save(rgb_landuse_path)
                    print("正在保存：" + rgb_landuse_path)
                if os.path.exists(rgb_coarse_path):
                    print("已经存在：" + rgb_coarse_path)
                else:
                    rgb_coarse_image.save(rgb_coarse_path)
                    print("正在保存：" + rgb_coarse_path)
                if os.path.exists(rgb_fine_path):
                    print("已经存在：" + rgb_fine_path)
                else:
                    rgb_fine_image.save(rgb_fine_path)
                    print("正在保存：" + rgb_fine_path)
                if os.path.exists(gray_landuse_path):
                    print("已经存在：" + gray_landuse_path)
                else:
                    gray_landuse_image.convert("L").save(gray_landuse_path)
                    print("正在保存：" + gray_landuse_path)
                if os.path.exists(gray_coarse_path):
                    print("已经存在：" + gray_coarse_path)
                else:
                    gray_coarse_image.convert("L").save(gray_coarse_path)
                    print("正在保存：" + gray_coarse_path)
                if os.path.exists(gray_fine_path):
                    print("已经存在：" + gray_fine_path)
                else:
                    gray_fine_image.convert("L").save(gray_fine_path)
                    print("正在保存：" + gray_fine_path)
        '''
        # 方法二：用cv2库进行处理，转换为向量以后在处理
        # 对每一张图片分别进行处理
        for image_file in os.listdir(image_path):
            image_file_path = image_path + "/" + image_file  # 每一张语义图片的完整路径
            print("正在处理：" + image_file_path)
            if not os.path.exists(image_file_path):
                print("不存在此图片：" + image_file_path)
            if os.path.isfile(image_file_path):  # 判断待处理的图像是否是文件而不是文件夹
                save_paths = get_unirgb_label(args['dataset'], image_file_path)
                if os.path.exists(save_paths['rgb_landuse']):
                    continue
                raw_image = read_image_file_path(args['dataset'], image_file_path)
                # # 以RGB图像的形式打开待处理文件，conver("RGB")的作用是将所有格式的图像转换为RGB三通道的
                # raw_image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
                # # 检查图像通道数
                # if len(raw_image.shape) == 2:  # 灰度图
                #     # 将灰度图转为RGB
                #     raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
                # raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                # arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
                rgb_landuse_image = np.zeros((raw_image.shape[0], raw_image.shape[1], raw_image.shape[2]),
                                             dtype=np.uint8)  # ①处理转换rgb的landuse分类
                rgb_coarse_image = np.zeros((raw_image.shape[0], raw_image.shape[1], raw_image.shape[2]),
                                            dtype=np.uint8)  # ②处理转换rgb的landform_rough分类
                rgb_fine_image = np.zeros((raw_image.shape[0], raw_image.shape[1], raw_image.shape[2]),
                                          dtype=np.uint8)  # ③处理转换rgb的landform_detailed分类
                gray_landuse_image = np.zeros((raw_image.shape[0], raw_image.shape[1]),
                                              dtype=np.uint8)  # ④处理转换gray的landuse分类
                gray_coarse_image = np.zeros((raw_image.shape[0], raw_image.shape[1]),
                                             dtype=np.uint8)  # ⑤处理转换gray的landfrom_rough分类
                gray_fine_image = np.zeros((raw_image.shape[0], raw_image.shape[1]),
                                           dtype=np.uint8)  # ⑥处理转换gray的landform_detailed分类
                # ①处理转换rgb的landuse分类           ④处理转换gray的landuse分类
                landuse_label_mapping = labels_dict[args['dataset']]['mapping']['landuse']
                for rgb_color, label in landuse_label_mapping.items():
                    mask = np.all(raw_image == np.array(rgb_color).reshape(1, 1, 3), axis=2)
                    rgb_landuse_image[mask] = mapping2rgb[label]
                    gray_landuse_image[mask] = mapping2gray[label]
                # ②处理转换rgb的landform_rough分类
                coarse_label_mapping = labels_dict[args['dataset']]['mapping']['coarse']
                for rgb_color, label in coarse_label_mapping.items():
                    mask = np.all(raw_image == np.array(rgb_color).reshape(1, 1, 3), axis=2)
                    rgb_coarse_image[mask] = mapping2rgb[label]
                    gray_coarse_image[mask] = mapping2gray[label]
                # ③处理转换rgb的landform_detailed分类
                fine_label_mapping = labels_dict[args['dataset']]['mapping']['fine']
                for rgb_color, label in fine_label_mapping.items():
                    mask = np.all(raw_image == np.array(rgb_color).reshape(1, 1, 3), axis=2)
                    rgb_fine_image[mask] = mapping2rgb[label]
                    gray_fine_image[mask] = mapping2gray[label]
                # 数据保存
                # 将像素数组转换为图像对象
                rgb_landuse_image = Image.fromarray(rgb_landuse_image.astype(np.uint8))
                gray_landuse_image = Image.fromarray(gray_landuse_image.astype(np.uint8))
                rgb_coarse_image = Image.fromarray(rgb_coarse_image.astype(np.uint8))
                gray_coarse_image = Image.fromarray(gray_coarse_image.astype(np.uint8))
                rgb_fine_image = Image.fromarray(rgb_fine_image.astype(np.uint8))
                gray_fine_image = Image.fromarray(gray_fine_image.astype(np.uint8))
                # 设计六个处理后的图像保存路径
                save_paths = get_unirgb_label(args['dataset'], image_file_path)
                if os.path.exists(save_paths['rgb_landuse']):
                    print("已经存在：" + save_paths['rgb_landuse'])
                else:
                    rgb_landuse_image.save(save_paths['rgb_landuse'])
                    print("正在保存：" + save_paths['rgb_landuse'])
                if os.path.exists(save_paths['rgb_coarse']):
                    print("已经存在：" + save_paths['rgb_coarse'])
                else:
                    rgb_coarse_image.save(save_paths['rgb_coarse'])
                    print("正在保存：" + save_paths['rgb_coarse'])
                if os.path.exists(save_paths['rgb_fine']):
                    print("已经存在：" + save_paths['rgb_fine'])
                else:
                    rgb_fine_image.save(save_paths['rgb_fine'])
                    print("正在保存：" + save_paths['rgb_fine'])
                if os.path.exists(save_paths['gray_landuse']):
                    print("已经存在：" + save_paths['gray_landuse'])
                else:
                    gray_landuse_image.convert("L").save(save_paths['gray_landuse'])
                    print("正在保存：" + save_paths['gray_landuse'])
                if os.path.exists(save_paths['gray_coarse']):
                    print("已经存在：" + save_paths['gray_coarse'])
                else:
                    gray_coarse_image.convert("L").save(save_paths['gray_coarse'])
                    print("正在保存：" + save_paths['gray_coarse'])
                if os.path.exists(save_paths['gray_fine']):
                    print("已经存在：" + save_paths['gray_fine'])
                else:
                    gray_fine_image.convert("L").save(save_paths['gray_fine'])
                    print("正在保存：" + save_paths['gray_fine'])
    print("语义图像颜色统一处理完成")
    # ---------------------------三、处理工作（遥感图像完全复制粘贴）-------------------------------------------
    image_dir = labels_dict[args['dataset']]['image_dir']
    image_files = os.path.join(input_dir, image_dir)
    for split_file in labels_dict[args['dataset']]['split_names']:  # ['train','val','test']:
        image_path = os.path.join(image_files, split_file)
        for image_file in os.listdir(image_path):
            image_file_path = image_path + "/" + image_file  # 每一张遥感图片的完整路径
            save_path = get_unirgb_image(args['dataset'], image_file_path)  # 对应将要保存的位置
            print("正在复制：" + image_file_path + " 到 " + save_path)

            if args['dataset'] == 'FLAIR':
                tifdataset = gdal.Open(image_file_path)
                image = tifdataset.ReadAsArray()
                image = image.transpose(1, 2, 0)
                image = image.astype('uint8')
                image=image[:,:,:3]
                image = Image.fromarray(image)
                image.save(save_path)
            else:
                raw_image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                raw_image = Image.fromarray(raw_image.astype(np.uint8))
                raw_image.save(save_path)

    print("遥感图像复制完成")


# 设置参数信息
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, default='/home/wyd/dl/data')  # 所有原始数据集文件夹的目录的路径
    parser.add_argument('--dataset', choices=['iSAID', 'loveDA', 'HRSCD', 'enviroatlas_landcover_nlcd',
                                              'enviroatlas_landcover_highres', 'Geonrw', 'DroneDeploy',
                                              'cvpr_chesapeake_landcover_nlcd', 'cvpr_chesapeake_landcover_lc',
                                              'MiniFrance','FLAIR'], default='FLAIR',
                        type=str)  # 选择哪一个数据集进行处理，两个参数组合在一起就是目标数据集的完整路径

    # turn the args into a dictionary
    args = vars(parser.parse_args())
    main(args)
