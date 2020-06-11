'''
@Author: your name
@Date: 2020-06-10 13:12:47
@LastEditTime: 2020-06-10 13:25:06
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /DeepCTR/mydeepctr/models/__init__.py
'''
from .fm import FM, FMConfig
from .ffm import FFM, FFMConfig
from .fm import FM as DeepFM
from .fm import FMConfig as DeepFMConfig
from .xdeepfm import XDeepFM, XDeepFMConfig
from .dcn import DCN, DCNConfig
from .mlr import MLR, MLRConfig
from .pnn import PNN, PNNConfig
from .wdl import WDL, WDLConfig

__all__ = ["FM", "FFM", "DeepFM", "XDeepFM", "DCN", "MLR", "PNN", "WDL", ]


model_dict = {
    "fm":FM,
    "ffm":FFM, 
    "deepfm":DeepFM, 
    "xdeepfm":XDeepFM, 
    "dcn":DCN, 
    "mlr":MLR, 
    "pnn":PNN, 
    "wdl":WDL,
}

model_config_dict = {
    "fm":FMConfig,
    "ffm":FFMConfig, 
    "deepfm":DeepFMConfig, 
    "xdeepfm":XDeepFMConfig, 
    "dcn":DCNConfig, 
    "mlr":MLRConfig, 
    "pnn":PNNConfig, 
    "wdl":WDLConfig,    
}