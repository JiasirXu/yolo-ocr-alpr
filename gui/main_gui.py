"""
ALPR 项目图形界面入口脚本
在本文件中初始化并启动 PyQt5 应用，展示 MainWindow。

usage:
    python gui/main_gui.py

确保已在虚拟环境 alpr-env 中安装所需依赖。
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication

# 将项目根目录加入 Python 路径，保证相对导入正常
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from gui.main_window import MainWindow


def main():
    """应用入口"""
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()