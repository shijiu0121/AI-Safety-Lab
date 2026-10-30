"""Aegis Lab：直接在 PyTorch 项目中使用的算法库。"""
from .models import Classifier, create_model
from .attacks import attack, AttackConfig

__version__ = "0.4.0"
__all__ = ["Classifier", "create_model", "attack", "AttackConfig"]
