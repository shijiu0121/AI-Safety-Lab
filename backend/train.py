"""兼容工作台入口；算法实现位于 aegis_lab。"""
import sys
from aegis_lab._core import train as implementation
if __name__ == '__main__':
    implementation.main()
else:
    sys.modules[__name__] = implementation
