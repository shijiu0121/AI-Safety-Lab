"""兼容工作台入口；算法实现位于 aegis_lab。"""
import sys
from aegis_lab._core import training_config as implementation
sys.modules[__name__] = implementation
