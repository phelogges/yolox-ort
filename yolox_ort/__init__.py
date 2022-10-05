# AUTHOR: raichu
# CONTACT: 1012415660@qq.com
# FILE: __init__.py
# DATE: 2022/10/5

__version__ = "0.1.0"

from yolox_ort.detector import Detector
from yolox_ort import utils

__all__ = ["utils", "Detector"]
