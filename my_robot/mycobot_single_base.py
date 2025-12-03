import sys

sys.path.append("./")

import numpy as np

from my_robot.base_robot import Robot

from controller.mycobot_controller import MycobotController
from sensor.Realsense_sensor import RealsenseSensor

from utils.data_handler import is_enter_pressed,debug_print

from data.collect_any import CollectAny

# 填写相机序列号
CAMERA_SERIALS = {
    'head': '344422070499',  # Replace with actual serial number
    # 'wrist': '344422070499',  # Replace with actual serial number
}

# 初始关节
START_POSITION_ANGLE_LEFT_ARM = [
    0.0,  # Joint 1
    -140.0,  # Joint 2
    120.9,  # Joint 3
    -85.5,  # Joint 4
    -92.5,  # Joint 5
    -23.5,  # Joint 6
]

# 单臂不需要管
START_POSITION_ANGLE_RIGHT_ARM = [
    0,  # Joint 1
    0,  # Joint 2
    0,  # Joint 3
    0,  # Joint 4
    0,  # Joint 5
    0,  # Joint 6
]

condition = {
    "robot": "mycobot_single",
    "save_path": "./datasets/",
    "task_name": "test",
    "save_format": "hdf5",
    "save_freq": 10,
}


class MycobotSingle(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.condition = condition
        self.controllers = {
            "arm": {
                "left_arm": MycobotController("left_arm"),
            },
        }
        self.sensors = {
            "image": {
                "cam_head": RealsenseSensor("cam_head"),
                #"cam_wrist": RealsenseSensor("cam_head"),
                # 等第二个摄像头
                # "cam_wrist": RealsenseSensor("cam_wrist"),
            },
        }

    # ============== init ==============
    def reset(self):
        self.controllers["arm"]["left_arm"].reset(np.array(START_POSITION_ANGLE_LEFT_ARM))

    def set_up(self):
        super().set_up()
        # TODO can0是什么
        # self.controllers["arm"]["left_arm"].set_up("can0")
        self.controllers["arm"]["left_arm"].set_up()
        self.sensors["image"]["cam_head"].set_up(CAMERA_SERIALS["head"])
        # 等第二个摄像头
        #self.sensors["image"]["cam_wrist"].set_up(CAMERA_SERIALS["head"])
        # self.sensors["image"]["cam_wrist"].set_up(CAMERA_SERIALS["wrist"])

        # 需要收集的机械臂参数
        # self.set_collect_type({"arm": ["joint", "qpos", "gripper"],
        #                       "image": ["color"]
        #                       })

        self.set_collect_type({"arm": ["joint", "qpos", "gripper"],
                               "image": ["color"]
                               })

        print("set up success!")


if __name__ == "__main__":
    import time

    robot = MycobotSingle()
    robot.set_up()
    # 初始化
    robot.reset()
    # debug_print("main", "Press Enter to start...", "INFO")
    # while not robot.is_start() or not is_enter_pressed():
    #    time.sleep(1 / robot.condition["save_freq"])

    # debug_print("main", "Press Enter to finish...", "INFO")

    #data_list = []
    #for i in range(100):
    #    print(i)
    #    data = robot.get()
    #    robot.collect(data)
    #    time.sleep(0.1)
    #robot.finish()

    # moving test
    move_data = {
        "arm": {
            "left_arm": {
                # [51.3, 85.7, 472.1, 146.5, 15.6, 108.0]
                #"qpos": [100, 85.34972279, 472.2331878, 143.23518357, 13.15332814, 107.07964034],
                "qpos": [50.85943208, 85.19449816, 468.90841, 143.29187535, 13.267704, 107.25389589],
                #"qpos": [-130.824, 256.262, 321.533, 176.891, -0.774, -128.700],
                # "qpos": [-10, 85.34972279, 469.2331878, 143.23518357, 13.15332814, 107.07964034],
                # "gripper": 0.2,
            },
        },
    }
    # robot.move(move_data)