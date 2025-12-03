import sys
import time
from pymycobot import ElephantRobot
from pymycobot import MyCobot

from utils.elegripper import Gripper

sys.path.append("./")

import numpy as np
from typing import Dict, Any, List

# 导入依赖（根据实际使用的 mycobot SDK 调整，这里以常见的 pymycobot 为例）
from pymycobot.mycobot import MyCobot
from pymycobot.genre import Angle, Coord

from controller.arm_controller import ArmController
from utils.data_handler import debug_print

class MycobotController(ArmController):
    # TODO
    def __init__(self, port: str, baudrate: int = 115200):
        # 调用父类构造函数
        super().__init__()
        self.name = "mycobot_controller"  # 重写控制器名称
        self.port = port  # 机械臂连接端口（如 "/dev/ttyUSB0" 或 "COM3"）
        self.baudrate = baudrate  # 波特率（默认 115200，需与机械臂匹配）
        self.controller = None  # 存储 mycobot 底层 SDK 实例
        self.gripper = None
        self.joint_count = 6  # mycobot 通常为 6 关节机械臂
        self.gripper_open_value = 100  # 夹爪完全张开值（根据实际校准调整）
        self.gripper_close_value = 0  # 夹爪完全闭合值

    # 初始化
    def set_up(self):
        try:
            # 初始化实例
            self.controller = ElephantRobot("10.194.18.232", 5001)
            self.gripper = Gripper("com3", baudrate=115200, id=14)
            self.controller.start_client()
            time.sleep(3)
            self.controller.start_robot()
            time.sleep(3)

            # 检查连接状态
            if self.controller.is_client_started:
                print("初始化完毕")
                # 启动机械臂
                self.controller._power_on()
            else:
                raise ConnectionError("Failed to connect to mycobot")
        except Exception as e:
            debug_print(self.name, f"Set up failed: {e}", "ERROR")
            raise  # 抛出异常，提示初始化失败

    # 初始化机械臂位置
    def reset(self, start_state):
        try:

            self.set_joint(start_state)
            self.gripper.set_gripper_value(0, 100)
            print("初始化完毕")

        except :
            print(f"reset error")
        return

    # 获取机械臂状态
    def get_state(self) -> Dict[str, Any]:
        # TODO 可以加个判断是否连接

        state = {}
        try:
            # 1. 获取关节角度（单位：度）
            joint_angles = self.controller.get_angles()
            state["joint"] = np.array(joint_angles) if joint_angles else np.zeros(self.joint_count)

            # 2. 获取末端执行器位置（笛卡尔坐标：x, y, z, rx, ry, rz）
            end_pos = self.controller.get_coords()
            state["qpos"] = np.array(end_pos) if end_pos else np.zeros(6)

            # TODO 3. 获取夹爪状态（0-100：闭合-张开）
            gripper_state = 1
            # gripper_state = self.controller.get_gripper_value()
            state["gripper"] = np.array([gripper_state]) if gripper_state is not None else np.array([self.gripper_close_value])

            # 4. 获取关节速度（部分 mycobot 型号支持，若无则返回0）
            # 若SDK不支持，可注释或改为估算逻辑
            # state["velocity"] = np.zeros(self.joint_count)  # 示例：默认返回0向量

            # 5. 获取力传感器数据（若机械臂带力传感，需根据SDK调整）
            # state["force"] = np.zeros(self.joint_count)  # 示例：默认返回0向量

            # 6. 记录最近执行的动作（这里简化存储，实际可根据需求扩展）
            # state["action"] = np.array([])

        except Exception as e:
            debug_print(self.name, f"Failed to get state: {e}", "WARNING")
            # state = self._get_empty_state()

        return state

    # TODO
    def _get_empty_state(self) -> Dict[str, np.ndarray]:
        """返回空状态（连接失败时使用，避免空指针）"""
        return {
            "joint": np.zeros(self.joint_count),
            "qpos": np.zeros(6),
            "gripper": np.array([self.gripper_close_value]),
            "velocity": np.zeros(self.joint_count),
            "force": np.zeros(self.joint_count),
            "action": np.array([]),
        }
    # ------------------------------ 底层控制方法实现 ------------------------------
    # 设置关节角度
    def set_joint(self, joint_angles: np.ndarray):
        """控制关节角度（绝对角度，单位：度）"""
        if len(joint_angles) != self.joint_count:
            debug_print(self.name, f"Joint count mismatch: expected {self.joint_count}, got {len(joint_angles)}", "ERROR")
            return
        # 限制关节角度范围（根据 mycobot 实际量程调整，避免机械损坏）
        # joint1_angle: 关节1角度，范围 - 180.00 ~ 180.00
        # joint2_angle: 关节2角度，范围 - 270.00 ~ 90.00
        # joint3_angle: 关节3角度，范围 - 150.00 ~ 150.00
        # joint4_angle: 关节4角度，范围 - 260.00 ~ 80.00
        # joint5_angle: 关节5角度，范围 - 168.00 ~ 168.00
        # joint6_angle: 关节6角度，范围 - 174.00 ~ 174.00
        # speed: 表示机械臂运动的速度，取值范围是0 ~ 2000
        joint_angles[0] = np.clip(joint_angles[0], -170, 170)
        joint_angles[1] = np.clip(joint_angles[1], -260, 90)
        joint_angles[2] = np.clip(joint_angles[2], -140, 140)
        joint_angles[3] = np.clip(joint_angles[3], -250, 70)
        joint_angles[4] = np.clip(joint_angles[4], -160, 160)
        joint_angles[5] = np.clip(joint_angles[5], -170, 170)
        # 为什么文档API和具体库不一样？
        self.controller.write_angles(joint_angles.tolist(), 1000)

    # 设置绝对坐标
    def set_position(self, position: np.ndarray):
        if len(position) != 6:
            debug_print(self.name, f"Position dimension mismatch: expected 6, got {len(position)}", "ERROR")
            return
        print(position.tolist())
        self.controller.write_coords(position.tolist(), 2000)
        self.controller.command_wait_done()
        # self.controller.command_wait_done()


    # TODO
    def set_position_teleop(self, position: np.ndarray):
        """遥操作模式下的位置控制（复用绝对位置控制，可根据需求扩展）"""
        self.set_position(position)

    # TODO
    def set_gripper(self, gripper_value: np.ndarray):
        """控制夹爪（0-100：闭合-张开）"""
        gripper_val = int(np.clip(gripper_value[0], self.gripper_close_value, self.gripper_open_value))
        self.controller.set_gripper_value(gripper_val, speed=50)

    # TODO
    def set_velocity(self, velocity: np.ndarray):
        """控制关节速度（部分 mycobot 型号支持，需SDK适配）"""
        debug_print(self.name, "Velocity control not fully implemented (check mycobot SDK)", "INFO")
        # 若SDK支持速度控制，可添加：self.controller.send_velocity(velocity.tolist())

    # TODO
    def set_force(self, force: np.ndarray):
        """控制力限制（部分 mycobot 型号支持力控，需SDK适配）"""
        debug_print(self.name, "Force control not fully implemented (check mycobot SDK)", "INFO")
        # 若SDK支持力控，可添加：self.controller.set_force_limit(force.tolist())

    # TODO
    def set_action(self, action: np.ndarray):
        """执行预定义动作（示例：根据动作向量索引调用内置动作，可自定义扩展）"""
        debug_print(self.name, f"Executing action: {action}", "DEBUG")
        # 示例：action[0] 为动作索引，1=抓取，2=释放
        if action.size > 0:
            action_idx = int(action[0])
            if action_idx == 1:
                self.set_gripper(np.array([self.gripper_close_value]))  # 抓取
            elif action_idx == 2:
                self.set_gripper(np.array([self.gripper_open_value]))  # 释放

    # TODO
    #def __repr__(self):
    #    """重写打印信息"""
    #    base_info = super().__repr__()
    #    if self.controller is not None:
    #        return f"{self.name}: \n \
    #                port: {self.port} \n \
    #                baudrate: {self.baudrate} \n \
    #                connected: {self.controller.is_connected()}"
    #    return base_info