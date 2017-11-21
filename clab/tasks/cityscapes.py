# -*- coding: utf-8 -*-
"""
http://crcv.ucf.edu/projects/Geosemantic/
http://www.6d-vision.com/scene-labeling
https://www.cc.gatech.edu/cpl/projects/videogeometriccontext/
"""
from clab.tasks._sseg import SemanticSegmentationTask


class CityScapes(SemanticSegmentationTask):
    def __init__(task):
        classnames = [
            'road', 'sidewalk', 'parking', 'rail track',
            'person', 'rider',
            'car', 'truck', 'bus', 'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer',
            'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'pole group', 'traffic sign', 'traffic light',
            'vegetation', 'terrain',
            'sky',
            'ground', 'dynamic', 'static',
        ]
        null_classname = None
        as_diva = True
        if as_diva:
            alias = {
                'road': 'Street',
                'sidewalk': 'Sidewalk',
                'parking': 'Parking_Lot',
                'rail track': 'Other',

                'person': 'Other',
                'rider': 'Other',

                'car': 'Other',
                'truck': 'Other',
                'bus': 'Other',
                'on rails': 'Other',
                'motorcycle': 'Other',
                'bicycle': 'Other',
                'caravan': 'Other',
                'trailer': 'Other',

                'building': 'Building',
                'wall': 'Other',
                'fence': 'Other',
                'guard rail': 'Other',
                'bridge': 'Other',
                'tunnel': 'Other',

                'pole': 'Other',
                'pole group': 'Other',
                'traffic sign': 'Other',
                'traffic light': 'Other',

                'vegetation': 'Trees',
                'terrain': 'Grass',

                'sky': 'Sky',

                'ground': 'Ground',
                'dynamic': 'Other',
                'static': 'Background',
            }
        else:
            alias = {}
        super(CityScapes, task).__init__(classnames, null_classname, alias)
