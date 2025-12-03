import sys

sys.path.append('./')

from pymycobot import ElephantRobot
import time
import pygame
import sys
import platform
import threading

from utils.elegripper import Gripper
from my_robot.mycobot_single_base import MycobotSingle

# 键盘映射配置（在pygame.init()后初始化）
KEY_MAP = {}

# init
init_angles = [0.0, -163.0, 122.0, -85.5, -92.5, -23.5]
go_home = [0, 0, 0, 0, 0, 0]

# 当前按下的键集合（用于跟踪持续按下的键）
pressed_keys = set()
last_action_time = {}  # 记录上次执行动作的时间，用于节流
action_interval = 0.05  # 动作执行间隔（秒），机械臂移动间隔
gripper_interval = 0.1  # 【新增】夹爪执行间隔，建议比机械臂稍慢，避免串口堵塞

# 夹爪初始化
current_gripper_value = 0  # 初始开度（0=完全闭合，100=完全张开）
max_value = 100  # 最大开度
step = 5  # 每次增加的开度步长
speed = 100  # 夹爪运动速度（0-100）

global_speed = 2000


def safe_stop():
    """安全停止机械臂运动"""
    try:
        ec.task_stop()
        time.sleep(0.02)
    except Exception as e:
        print("stop 出错：", e)


def gripper_open():
    """打开夹爪 - 增加开度值"""
    global current_gripper_value

    # 如果当前值还未达到最大值，则进行增加
    if current_gripper_value < max_value:
        current_gripper_value += step

        # 边界检查：确保不超过最大值
        if current_gripper_value > max_value:
            current_gripper_value = max_value

        # print(f"执行夹爪打开，当前值: {current_gripper_value}") # 注释掉避免刷屏
        try:
            # 调用机械臂接口设定夹爪数值
            g.set_gripper_value(current_gripper_value, speed)
        except Exception as e:
            print(f"夹爪指令发送失败: {e}")
    else:
        pass  # 到达最大值不打印，避免刷屏


def gripper_close():
    """关闭夹爪 - 减小开度值"""
    global current_gripper_value

    # 如果当前值大于0，则进行减小
    if current_gripper_value > 0:
        current_gripper_value -= step

        # 边界检查：确保不小于0
        if current_gripper_value < 0:
            current_gripper_value = 0

        # print(f"执行夹爪关闭，当前值: {current_gripper_value}") # 注释掉避免刷屏
        try:
            # 调用机械臂接口设定夹爪数值
            g.set_gripper_value(current_gripper_value, speed)
        except Exception as e:
            print(f"夹爪指令发送失败: {e}")
    else:
        pass  # 到达最小值不打印


def handle_keyboard_input():
    """处理键盘输入并执行相应的机械臂动作（包含夹爪）"""
    global current_gripper_value
    current_time = time.time()

    # --- 处理夹爪持续运动 (新增部分) ---
    # 检查打开键
    if KEY_MAP['gripper_open'] in pressed_keys:
        # 使用独立的夹爪间隔或复用action_interval
        if KEY_MAP['gripper_open'] not in last_action_time or (
                current_time - last_action_time[KEY_MAP['gripper_open']]) >= gripper_interval:
            print(f"打开夹爪... Val: {current_gripper_value}")
            gripper_open()
            last_action_time[KEY_MAP['gripper_open']] = current_time

    # 检查关闭键 (使用elif互斥，防止同时按)
    elif KEY_MAP['gripper_close'] in pressed_keys:
        if KEY_MAP['gripper_close'] not in last_action_time or (
                current_time - last_action_time[KEY_MAP['gripper_close']]) >= gripper_interval:
            print(f"关闭夹爪... Val: {current_gripper_value}")
            gripper_close()
            last_action_time[KEY_MAP['gripper_close']] = current_time

    # --- 下面是原有的机械臂移动逻辑 ---

    need_action = False
    # 简单的节流检查
    for key in pressed_keys:
        # 排除夹爪键，因为夹爪上面已经处理过了
        if key == KEY_MAP['gripper_open'] or key == KEY_MAP['gripper_close']:
            continue

        if key not in last_action_time or (current_time - last_action_time[key]) >= action_interval:
            need_action = True
            break

    if not need_action:
        return

    # 处理坐标轴移动
    if KEY_MAP['move_x_forward'] in pressed_keys:
        print("+X")
        ec.jog_coord('X', 1, global_speed)
        last_action_time[KEY_MAP['move_x_forward']] = current_time
    elif KEY_MAP['move_x_backward'] in pressed_keys:
        print("-X")
        ec.jog_coord('X', -1, global_speed)
        last_action_time[KEY_MAP['move_x_backward']] = current_time

    if KEY_MAP['move_y_right'] in pressed_keys:
        print("-Y")
        ec.jog_coord('Y', -1, global_speed)
        last_action_time[KEY_MAP['move_y_right']] = current_time
    elif KEY_MAP['move_y_left'] in pressed_keys:
        print("+Y")
        ec.jog_coord('Y', 1, global_speed)
        last_action_time[KEY_MAP['move_y_left']] = current_time

    if KEY_MAP['move_z_up'] in pressed_keys:
        print("+Z")
        ec.jog_coord('Z', 1, global_speed)
        last_action_time[KEY_MAP['move_z_up']] = current_time
    elif KEY_MAP['move_z_down'] in pressed_keys:
        print("-Z")
        ec.jog_coord('Z', -1, global_speed)
        last_action_time[KEY_MAP['move_z_down']] = current_time

    # 处理旋转
    if KEY_MAP['rotate_rx_positive'] in pressed_keys:
        print("+RX")
        ec.jog_coord('RX', 1, global_speed)
        last_action_time[KEY_MAP['rotate_rx_positive']] = current_time
    elif KEY_MAP['rotate_rx_negative'] in pressed_keys:
        print("-RX")
        ec.jog_coord('RX', -1, global_speed)
        last_action_time[KEY_MAP['rotate_rx_negative']] = current_time

    if KEY_MAP['rotate_ry_positive'] in pressed_keys:
        print("-RY")
        ec.jog_coord('RY', -1, global_speed)
        last_action_time[KEY_MAP['rotate_ry_positive']] = current_time
    elif KEY_MAP['rotate_ry_negative'] in pressed_keys:
        print("+RY")
        ec.jog_coord('RY', 1, global_speed)
        last_action_time[KEY_MAP['rotate_ry_negative']] = current_time

    if KEY_MAP['rotate_rz_positive'] in pressed_keys:
        print("+RZ")
        ec.jog_coord('RZ', 1, global_speed)
        last_action_time[KEY_MAP['rotate_rz_positive']] = current_time
    elif KEY_MAP['rotate_rz_negative'] in pressed_keys:
        print("-RZ")
        ec.jog_coord('RZ', -1, global_speed)
        last_action_time[KEY_MAP['rotate_rz_negative']] = current_time

    # 检查是否需要停止（所有移动键都释放时）
    movement_keys = [
        KEY_MAP['move_x_forward'], KEY_MAP['move_x_backward'],
        KEY_MAP['move_y_right'], KEY_MAP['move_y_left'],
        KEY_MAP['move_z_up'], KEY_MAP['move_z_down'],
        KEY_MAP['rotate_rx_positive'], KEY_MAP['rotate_rx_negative'],
        KEY_MAP['rotate_ry_positive'], KEY_MAP['rotate_ry_negative'],
        KEY_MAP['rotate_rz_positive'], KEY_MAP['rotate_rz_negative'],
    ]
    any_movement_pressed = any(key in pressed_keys for key in movement_keys)
    if not any_movement_pressed and any(key in last_action_time for key in movement_keys):
        # 如果有之前的移动动作但现在没有键按下，则停止
        threading.Thread(target=safe_stop).start()
        # 清理移动键的记录
        for key in movement_keys:
            last_action_time.pop(key, None)


if __name__ == '__main__':
    pygame.init()

    # 初始化键盘映射配置（必须在pygame.init()之后）
    KEY_MAP.update({
        'move_x_forward': pygame.K_w,
        'move_x_backward': pygame.K_s,
        'move_y_right': pygame.K_d,
        'move_y_left': pygame.K_a,
        'move_z_up': pygame.K_q,
        'move_z_down': pygame.K_e,
        'rotate_rx_positive': pygame.K_y,
        'rotate_rx_negative': pygame.K_h,
        'rotate_ry_positive': pygame.K_g,
        'rotate_ry_negative': pygame.K_j,
        'rotate_rz_positive': pygame.K_t,
        'rotate_rz_negative': pygame.K_u,
        'gripper_open': pygame.K_r,
        'gripper_close': pygame.K_f,
        'to_init': pygame.K_z,
        'stop': pygame.K_x,
    })

    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("机械臂键盘控制 - 按ESC退出")

    ec = None
    try:
        robot = MycobotSingle()
        robot.set_up()
        robot.reset()
        ec = robot.controllers["arm"]["left_arm"].controller
        g = robot.controllers["arm"]["left_arm"].gripper

        print("程序运行中，请使用键盘控制机械臂...")
        print("注意：此窗口需要保持激活状态才能接收键盘输入\n")

        running = True
        clock = pygame.time.Clock()

        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key = event.key

                    if key == pygame.K_ESCAPE:
                        print("退出程序...")
                        running = False
                        break

                    elif key == KEY_MAP['to_init']:
                        print("回到初始化位置")
                        ec.write_angles(init_angles, 1000)
                        time.sleep(0.1)

                    # 移除了这里的夹爪控制代码，只保留添加到 pressed_keys
                    # 让 handle_keyboard_input 在主循环中处理它

                    # 添加到按下的键集合
                    pressed_keys.add(key)

                elif event.type == pygame.KEYUP:
                    key = event.key
                    pressed_keys.discard(key)

                    # 如果松开的是夹爪键，从时间记录中移除，以便下次按下立即响应
                    if key == KEY_MAP['gripper_open'] or key == KEY_MAP['gripper_close']:
                        last_action_time.pop(key, None)

                    # 机械臂停止逻辑保持不变
                    movement_keys = [
                        KEY_MAP['move_x_forward'], KEY_MAP['move_x_backward'],
                        KEY_MAP['move_y_right'], KEY_MAP['move_y_left'],
                        KEY_MAP['move_z_up'], KEY_MAP['move_z_down'],
                        KEY_MAP['rotate_rx_positive'], KEY_MAP['rotate_rx_negative'],
                        KEY_MAP['rotate_ry_positive'], KEY_MAP['rotate_ry_negative'],
                        KEY_MAP['rotate_rz_positive'], KEY_MAP['rotate_rz_negative'],
                    ]
                    if key in movement_keys:
                        threading.Thread(target=safe_stop).start()

            # 持续监测按键状态并执行动作
            handle_keyboard_input()

            clock.tick(50)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if ec:
            try:
                safe_stop()
                time.sleep(0.1)
                ec._power_off()
                ec._state_off()
                ec.stop_client()
            except Exception as e:
                print(f"清理时出错: {e}")
        pygame.quit()
        sys.exit()