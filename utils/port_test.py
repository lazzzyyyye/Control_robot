import sys

from pymycobot import ElephantRobot

sys.path.append("./")

import numpy as np
from typing import Dict, Any, List

# 导入依赖（根据实际使用的 mycobot SDK 调整，这里以常见的 pymycobot 为例）
from pymycobot.mycobot import MyCobot
from pymycobot.genre import Angle, Coord

from controller.arm_controller import ArmController
from utils.data_handler import debug_print

robot = ElephantRobot("10.194.12.181", 5001)
robot.start_client()
while(True):
    print(1)










