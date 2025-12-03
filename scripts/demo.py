import sys
sys.path.append('./')

import os
import time
from collections import deque
from dataclasses import dataclass
import numpy as np
from my_robot.mycobot_single_base import MycobotSingle
from utils.data_handler import is_enter_pressed

# --- 新增依赖 ---
import pyrealsense2 as rs

from utils.image_tools import convert_to_uint8, resize_with_pad
from utils.websocket_client_policy import WebsocketClientPolicy
from utils.misc import _quat2axisangle


@dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    max_steps: int = 100


class RealRobotEnv:
    def __init__(self):
        print("Initializing RealSense Cameras...")

        # =========================================================
        # TODO: 在这里填入你真实相机的序列号 (字符串格式)
        # =========================================================
        self.FRONT_CAM_SERIAL = "344422070499"  # 替换为前视相机序列号
        # self.WRIST_CAM_SERIAL = "123456789002"  # 替换为手腕相机序列号

        # 配置相机分辨率 (640x480 对于 Resize 到 224x224 足够清晰且节省USB带宽)
        self.img_width = 640
        self.img_height = 480
        self.fps = 30

        # TODO 初始化两个相机管道
        # self.pipe_front = self._init_realsense(self.FRONT_CAM_SERIAL)
        # self.pipe_wrist = self._init_realsense(self.WRIST_CAM_SERIAL)

        # 2. 初始化机械臂 (这里保留你的逻辑)
        # self.robot = RobotArm(ip="192.168.1.x")
        # self.robot.connect()
        # self.robot.move_to_home()

    def _init_realsense(self, serial_number):
        """辅助函数：初始化单个 RealSense 相机"""
        pipeline = rs.pipeline()
        config = rs.config()

        # 启用指定序列号的设备
        config.enable_device(serial_number)

        # 配置 RGB 流 (直接获取 RGB，省去 BGR转RGB 的步骤)
        config.enable_stream(rs.stream.color, self.img_width, self.img_height, rs.format.rgb8, self.fps)

        try:
            pipeline.start(config)
            print(f"Camera {serial_number} started successfully.")
            return pipeline
        except Exception as e:
            print(f"Error starting camera {serial_number}: {e}")
            raise e

    def _get_realsense_frame(self, pipeline):
        """辅助函数：从管道获取一帧 numpy 图像"""
        # 等待一帧数据 (阻塞模式，确保数据同步)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise Exception("RealSense frame drop")

        # 转换为 Numpy 数组
        # 注意：因为我们配置的是 rs.format.rgb8，所以这里拿到已经是 RGB 格式
        img = np.asanyarray(color_frame.get_data())
        return img

    def get_observation(self):
        """获取真机当前的图像和状态"""

        # --- 获取图像 (RealSense) ---
        # 直接获取 RGB 图像
        # img_front = self._get_realsense_frame(self.pipe_front)

        # --- 获取机械臂状态 ---
        # TODO 模拟数据 (替换为真实读取的代码)
        info = robot.get()
        img = info[1]["cam_head"]["color"]


        eef_pos = info[0]["left_arm"]["qpos"]
        eef_gripper = info[0]["left_arm"]["gripper"]
        gripper_2d = np.repeat(eef_gripper, 2)

        # 2. 拼接
        # 6维关节 + 2维夹爪 = 8维向量
        state_8_dim = np.concatenate([eef_pos, gripper_2d], axis=0)

        return {
            "frontview_image": img,
            # 如果你只有一个相机，这里暂时复用 img，或者填入全黑图像
            "robot0_eye_in_hand_image": img,
            "robot0_eef_pos": state_8_dim,
        }

    def step(self, action):
        """在真机执行动作"""
        # self.robot.move_delta(action)
        move_data = self.convert_action_to_move_data(action)
        robot.move(move_data)
        print(f"Executing Action on Real Robot: {action}")
        time.sleep(0.1)

    def cleanup(self):
        print("Stopping cameras...")
        try:
            self.pipe_front.stop()
        except Exception as e:
            print(f"Error stopping cameras: {e}")

    def convert_action_to_move_data(self, action):

        qpos = action[:6].tolist() if isinstance(action, np.ndarray) else list(action[:6])

        raw_gripper = float(action[6])

        gripper_cmd = raw_gripper  # 这里暂时保持原值

        # 3. 组装字典
        move_data = {
            "arm": {
                "left_arm": {
                    "qpos": qpos,
                    # "gripper": gripper_cmd,
                },
            },
        }

        return move_data


def eval_real_robot(args: Args):
    client = WebsocketClientPolicy(args.host, args.port)



    # 替换为真机环境类
    try:
        env = RealRobotEnv()
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    task = input("输入任务指令：")
    action_deque = deque()

    try:
        for step in range(args.max_steps):
            # 1. 获取观测
            obs = env.get_observation()

            # 2. 预处理图像
            img = obs["frontview_image"]
            wrist_img = obs["robot0_eye_in_hand_image"]

            # RealSense 获取的已经是 RGB，且方向通常是正的
            # 如果发现图像倒了，可以使用 img = cv2.flip(img, -1)
            img = convert_to_uint8(resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = convert_to_uint8(resize_with_pad(wrist_img, args.resize_size, args.resize_size))

            if len(action_deque) == 0:
                # 3. 组装数据包发送给 Server
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": obs["robot0_eef_pos"],
                    "prompt": str(task),
                }

                # 4. 推理
                action_chunk = client.infer(element)["actions"]
                action_deque.extend(action_chunk[: args.replan_steps])

            # 5. 执行动作
            action = action_deque.popleft()
            env.step(action)

    # except KeyboardInterrupt:
        #print("停止运行")
    # except Exception as e:
        #print(f"[error]: {e}")
    finally:
        env.cleanup()


if __name__ == "__main__":
    args = Args()
    robot = MycobotSingle()
    robot.set_up()

    robot.reset()
    print("按回车")
    while not is_enter_pressed():
        time.sleep(1 / robot.condition["save_freq"])

    eval_real_robot(args)