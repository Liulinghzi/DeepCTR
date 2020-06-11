'''
@Author: your name
@Date: 2020-04-09 18:11:17
@LastEditTime: 2020-05-07 13:42:48
@LastEditors: your name
@Description: In User Settings Edit
@FilePath: /code learn/DeepCTR/deepctr/models/__init__.py
'''
from .afm import AFM
from .autoint import AutoInt
from .ccpm import CCPM
from .dcn import DCN
from .deepfm import DeepFM
from .multi_deepfm import MultiDeepFM
from .dien import DIEN
from .din import DIN
from .fnn import FNN
from .mlr import MLR
from .onn import ONN
from .onn import ONN as NFFM
from .nfm import NFM
from .pnn import PNN
from .wdl import WDL
from .xdeepfm import xDeepFM
from .fgcnn import FGCNN
from .dsin import DSIN
from .fibinet import FiBiNET
from .flen import FLEN

__all__ = ["AFM", "CCPM","DCN", "MLR",  "DeepFM",
           "MLR", "NFM", "DIN", "DIEN", "FNN", "PNN", "WDL", "xDeepFM", "AutoInt", "ONN", "FGCNN", "DSIN", "FiBiNET", 'FLEN']
