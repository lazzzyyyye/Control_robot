import sys

sys.path.append("./")

import numpy as np

from my_robot.base_robot import Robot

from controller.mycobot_controller import MycobotController
from sensor.Realsense_sensor import RealsenseSensor

from data.collect_any import CollectAny

# 填写相机序列号
CAMERA_SERIALS = {
    'head': '111',  # Replace with actual serial number
    # 'wrist': '111',  # Replace with actual serial number
}

# 初始关节
START_POSITION_ANGLE_LEFT_ARM = [
    71.5,  # Joint 1
    -148.5,  # Joint 2
    116.5,  # Joint 3
    -88.0,  # Joint 4
    -90.0,  # Joint 5
    0.0,  # Joint 6
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
                # "cam_wrist": RealsenseSensor("cam_wrist"),
            },
        }

    # ============== init ==============
    def reset(self):
        self.controllers["arm"]["left_arm"].reset(np.array(START_POSITION_ANGLE_LEFT_ARM))

    def set_up(self):
        super().set_up()
        # TODO can0是什么
        self.controllers["arm"]["left_arm"].set_up("can0")
        self.sensors["image"]["cam_head"].set_up(CAMERA_SERIALS["head"])
        # 等第二个摄像头
        # self.sensors["image"]["cam_wrist"].set_up(CAMERA_SERIALS["wrist"])

        # 需要收集的机械臂参数
        self.set_collect_type({"arm": ["joint", "qpos", "gripper"],
                               "image": ["color"]
                               })

        print("set up success!")


if __name__ == "__main__":
    import time

    robot = MycobotSingle()
    robot.set_up()
    # collection test
    robot.reset()
    data_list = []
    for i in range(100):
        print(i)
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)
    robot.finish()

    # moving test
    move_data = {
        "arm": {
            "left_arm": {
                "qpos": [0.057, 0.0, 0.216, 0.0, 0.085, 0.0],
                # "gripper": 0.2,
            },
        },
    }
    robot.move(move_data)
    time.sleep(1)
    move_data = {
        "arm": {
            "left_arm": {
                "joint": [0.00, 0.0, 0.0, 0.0, 0.0, 0.0],
                # "gripper": 0.2,
            },
        },
    }
    robot.move(move_data)