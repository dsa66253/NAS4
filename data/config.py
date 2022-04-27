# config.py

cfg_alexnet = {
    'name': 'alexnet',
    'clip': False,
    'loc_weight': 1.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 4,
    'epoch': 45,
    'decay1': 70,
    'decay2': 90,
    'image_size': 128,
    'pretrain': False,
    'in_channel': 8,
    'out_channel': 64, 
    "cuddbenchMark": False
}
# 
cfg_nasmodel = {
    'name': 'NasModel',
    'clip': False,
    'loc_weight': 1.0,
    'gpu_train': True,
    'batch_size': 64,
    'start_train_nas_epoch': 4,
    'ngpu': 1,
    'epoch': 45,
    'decay1': 70,
    'decay2': 90,
    'image_size': 128,
    'pretrain': False,
    'in_channel': 8,
    'out_channel': 64,
    "numOfClasses": 10,
    "numOfLayers": 5,
    "numOfInnerCell": 2,
    "numOfOperations":6,
    "cuddbenchMark": False,
    
}

cfg_newnasmodel = {
    'name': 'NewNasModel',
    'clip': False,
    'loc_weight': 1.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 1,
    'epoch': 45,
    'decay1': 70,
    'decay2': 90,
    'image_size': 128,
    'pretrain': False,
    'in_channel': 8,
    'out_channel': 64,
    "numOfClasses": 10,
    "numOfLayers": 5,
    "numOfInnerCell": 2,
    "numOfOperations":5,
    "cuddbenchMark": False
}

epoch_to_drop = [10, 25, 35] #在第幾個epoch要使用剔除機制
dropNum = [1, 1, 1] #在特定epoch剔除1個最小alpha的操作

PRIMITIVES = [
    'conv_3x3',
    'conv_5x5',
    'conv_7x7',
    'conv_9x9',
    'conv_11x11',
]

PRIMITIVES_max = [
    'conv_1x1',
    'conv_1x1',
    'conv_1x1',
    'conv_1x1',
    'conv_1x1',
    'max_pool_3x3',
    'avg_pool_3x3'
]

PRIMITIVES_skip = [
    'conv_3x3',
    'conv_5x5',
    'conv_7x7',
    'conv_9x9',
    'conv_11x11',
    'skip_connect'
]
