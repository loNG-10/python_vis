import tkinter as tk
from tkinter import ttk, messagebox
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
import matplotlib
from datetime import datetime
import time
import serial
import serial.tools.list_ports
import re
import threading
import queue

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataGloveVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("数据手套可视化系统")
        self.root.geometry("1200x800")
        
        # 初始化数据队列
        self.data_queue = queue.Queue()
        self.serial_thread = None
        self.thread_running = False
        
        # 初始化串口
        self.angle_serial = None
        self.pressure_serial = None
        self.available_ports = []  # 存储可用串口列表
        
        # 添加校准数据存储
        self.calibration_min = [float('inf')] * 14
        self.calibration_max = [float('-inf')] * 14
        self.is_calibrated = False
        
        # 初始化视角参数
        self.view_elev = 45
        self.view_azim = 45
        self.update_interval = 17  # 默认60FPS (1000ms/60 ≈ 16.67ms)
        
        # 初始化颜色方案
        self.finger_colors = {
            'finger_0': '#FF99CC',  # 大拇指
            'finger_1': '#FF9999',  # 食指
            'finger_2': '#99FF99',  # 中指
            'finger_3': '#9999FF',  # 无名指
            'finger_4': '#FFFF99'   # 小指
        }
        
        # 初始化角度和压力数据
        self.angle_data = {}
        self.angle_data["finger_1"] = [30, 40, 50]  # 食指
        self.angle_data["finger_2"] = [35, 45, 55]  # 中指
        self.angle_data["finger_3"] = [40, 50, 60]  # 无名指
        self.angle_data["finger_4"] = [45, 55, 65]  # 小指
        self.angle_data["finger_0"] = [180]*2  # 大拇指只有两个值
        self.pressure_data = [0]*5
        
        # 创建主分割窗口
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建上部控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 添加更新锁定变量
        self.updating = False
        
        # 添加控制按钮
        self.init_control_panel()
        
        # 创建左侧和右侧分割窗口
        self.left_frame = ttk.Frame(self.main_frame)
        self.right_frame = ttk.Frame(self.main_frame)
        
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 3D可视化区域
        self.frame_3d = ttk.LabelFrame(self.left_frame, text="3D手部模型")
        self.frame_3d.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 数据监控区域
        self.frame_data = ttk.Notebook(self.right_frame)
        self.frame_data.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 初始化各个组件
        self.init_3d_view()
        self.init_data_tab()
        
        logging.info("应用程序初始化完成")

    def create_hand_model(self, angles=None, pressures=None):
        """使用 Matplotlib 创建手部模型"""
        if angles is None:
            angles = self.angle_data
        if pressures is None:
            pressures = self.pressure_data
            
        # 清除当前图形
        self.ax.cla()
        
        # 更新视图范围
        self.update_view_limits()
        
        # 手掌参数
        palm_width = 0.3
        palm_height = 0.4
        
        # 绘制手掌轮廓
        palm_points = np.array([
            [-palm_width/2, 0, 0],  # 左下
            [palm_width/2, 0, 0],   # 右下
            [palm_width/2, palm_height, 0],  # 右上
            [-palm_width/2, palm_height, 0]  # 左上
        ])
        
        # 绘制手掌
        self.ax.plot(palm_points[[0,1,2,3,0], 0], 
                    palm_points[[0,1,2,3,0], 1], 
                    palm_points[[0,1,2,3,0], 2], 
                    color='gray', alpha=0.5)
        
        # 手指基础参数
        finger_positions = [
            (-0.12, palm_height, 0),    # 食指
            (-0.04, palm_height, 0),    # 中指
            (0.04, palm_height, 0),     # 无名指
            (0.12, palm_height, 0)      # 小指
        ]
        
        # 绘制手指
        for i, (x, y, z) in enumerate(finger_positions):
            finger_key = f'finger_{i+1}'
            finger_angles = angles[finger_key]
            finger_color = self.finger_colors[finger_key]
            
            # 获取压力值并计算颜色
            pressure = min(max(pressures[i+1] / 1024.0, 0), 1)  # 归一化压力值
            base_color = np.array(matplotlib.colors.to_rgb(finger_color))
            highlight_color = np.array([1, 0, 0])  # 红色
            color = tuple(base_color * (1-pressure) + highlight_color * pressure)
            
            # 计算累积角度和位置
            current_x, current_y, current_z = x, y, z
            cumulative_angle = 0  # 累积角度
            joint_lengths = [0.12, 0.10, 0.08]  # 关节长度
            
            # 绘制起始关节
            self.ax.scatter([current_x], [current_y], [current_z], 
                          c=[color], s=60, alpha=0.8)
            
            # 绘制每个关节
            for j, (angle, length) in enumerate(zip(finger_angles, joint_lengths)):
                # 累积角度（相对于上一个关节）
                cumulative_angle += angle
                
                # 计算新的位置
                radians = np.deg2rad(cumulative_angle)
                next_x = current_x
                next_y = current_y + length * np.cos(radians)
                next_z = current_z + length * np.sin(radians)
                
                # 绘制骨节
                self.ax.plot([current_x, next_x], 
                           [current_y, next_y], 
                           [current_z, next_z],
                           color=color, 
                           linewidth=3*(1-j*0.2),
                           alpha=0.8)
                
                # 绘制关节球体
                joint_size = 60 * (1-(j+1)*0.2)
                self.ax.scatter([next_x], [next_y], [next_z], 
                              c=[color], s=joint_size, alpha=0.8)
                
                # 更新当前位置
                current_x, current_y, current_z = next_x, next_y, next_z
        
        # 特殊处理大拇指
        thumb_pos = (-palm_width/2 - 0.02, palm_height*0.3, 0)
        thumb_angles = angles['finger_0']
        thumb_color = self.finger_colors['finger_0']
        
        # 获取大拇指压力值并计算颜色
        thumb_pressure = min(max(pressures[0] / 1024.0, 0), 1)  # 归一化压力值
        base_color = np.array(matplotlib.colors.to_rgb(thumb_color))
        highlight_color = np.array([1, 0, 0])  # 红色
        thumb_color = tuple(base_color * (1-thumb_pressure) + highlight_color * thumb_pressure)
        
        current_x, current_y, current_z = thumb_pos
        cumulative_angle = -30  # 大拇指基础角度
        
        # 绘制大拇指起始关节
        self.ax.scatter([current_x], [current_y], [current_z], 
                      c=[thumb_color], s=60, alpha=0.8)
        
        # 大拇指关节
        joint_lengths = [0.12, 0.10]
        for j, (angle, length) in enumerate(zip(thumb_angles, joint_lengths)):
            # 累积角度
            cumulative_angle += angle
            
            # 计算新位置
            radians = np.deg2rad(cumulative_angle)
            next_x = current_x - length * np.cos(radians) * 0.5
            next_y = current_y + length * np.cos(radians)
            next_z = current_z + length * np.sin(radians)
            
            # 绘制骨节
            self.ax.plot([current_x, next_x], 
                       [current_y, next_y], 
                       [current_z, next_z],
                       color=thumb_color, 
                       linewidth=3*(1-j*0.2),
                       alpha=0.8)
            
            # 绘制关节球体
            joint_size = 60 * (1-j*0.2)
            self.ax.scatter([next_x], [next_y], [next_z], 
                          c=[thumb_color], s=joint_size, alpha=0.8)
            
            current_x, current_y, current_z = next_x, next_y, next_z
        
        # 更新画布
        self.canvas.draw()

    def init_3d_view(self):
        """初始化3D视图"""
        # 创建图形和轴对象
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 初始化视图参数
        self.view_azim = 0
        self.view_elev = 0
        self.zoom_scale = 0.5
        
        # 设置固定的显示范围
        self.update_view_limits()
        
        # # 设置标签
        # self.ax.set_xlabel('X轴')
        # self.ax.set_ylabel('Y轴')
        # self.ax.set_zlabel('Z轴')
        
        # # 添加标题
        # self.ax.set_title('手部模型可视化')
        
        # 优化显示效果
        self.ax.grid(False)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # 在Tkinter中嵌入Matplotlib图形
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_3d)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加鼠标事件处理
        self._init_mouse_control()
        
        # 创建初始手部模型
        self.create_hand_model()
        
        logging.info("3D视图初始化完成")

    def update_view_limits(self):
        """更新视图范围"""
        self.ax.set_xlim(-0.8 * self.zoom_scale, 0.8 * self.zoom_scale)
        self.ax.set_ylim(-0.3 * self.zoom_scale, 1.0 * self.zoom_scale)
        self.ax.set_zlim(-0.5 * self.zoom_scale, 0.8 * self.zoom_scale)

    def _init_mouse_control(self):
        """初始化鼠标控制"""
        self.last_x = 0
        self.last_y = 0
        self.is_rotating = False

        def on_mouse_press(event):
            if event.inaxes == self.ax:
                self.is_rotating = True
                self.last_x = event.xdata
                self.last_y = event.ydata

        def on_mouse_release(event):
            self.is_rotating = False

        def on_mouse_move(event):
            if self.is_rotating and event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                dx = event.xdata - self.last_x
                dy = event.ydata - self.last_y
                
                # 更新视角
                self.view_azim = (self.view_azim - dx * 150) % 360
                self.view_elev = np.clip(self.view_elev + dy * 150, -90, 90)
                
                self.ax.view_init(self.view_elev, self.view_azim)
                self.canvas.draw()
                
                self.last_x = event.xdata
                self.last_y = event.ydata

        def on_scroll(event):
            # 根据滚轮方向调整缩放比例
            if event.button == 'up':
                self.zoom_scale *= 0.9  # 放大（缩小范围）
            else:
                self.zoom_scale *= 1.1  # 缩小（增大范围）
            
            # 更新视图范围
            self.update_view_limits()
            self.canvas.draw()

        # 绑定鼠标事件
        self.canvas.mpl_connect('button_press_event', on_mouse_press)
        self.canvas.mpl_connect('button_release_event', on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        self.canvas.mpl_connect('scroll_event', on_scroll)

    def update_hand_model(self, angles=None, pressures=None):
        """更新手部模型"""
        logging.debug(f"update_hand_model called with angles={angles}, pressures={pressures}")
        logging.debug(f"current self.angle_data={self.angle_data}")
        logging.debug(f"current self.pressure_data={self.pressure_data}")
        
        # 保存当前视角
        current_elev = self.ax.elev
        current_azim = self.ax.azim
        
        # 清除当前图形内容，但保留坐标轴
        self.ax.clear()
        
        # 重新设置视图属性
        self.update_view_limits()
        self.ax.grid(False)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.set_xlabel('X轴')
        self.ax.set_ylabel('Y轴')
        self.ax.set_zlabel('Z轴')
        self.ax.set_title('手部模型可视化')
        
        # 恢复之前的视角
        self.ax.view_init(elev=current_elev, azim=current_azim)
        
        # 创建新的手部模型
        self.create_hand_model(angles, pressures)
        
        # 更新画布
        self.canvas.draw()

    def serial_read_thread(self):
        """串口数据读取线程"""
        print("\n=== 串口读取线程启动 ===")
        buffer = ""
        
        while self.thread_running:
            try:
                if self.angle_serial and self.angle_serial.is_open:
                    if self.angle_serial.in_waiting:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"\n[{timestamp}] ===== 角度传感器数据 =====")
                        data = self.angle_serial.readline().decode('utf-8').strip()
                        print(f"► 原始字节: {data}")
                        try:
                            decoded = data.strip()
                            print(f"► 解码数据: {decoded}")
                            if decoded:
                                success = self.parse_serial_data(decoded)
                                print(f"► 数据解析{'成功' if success else '失败'}")
                                if success:
                                    self.data_queue.put(True)
                        except Exception as e:
                            print(f"[错误] 角度传感器数据解码失败: {e}")
                            continue
                
                if self.pressure_serial and self.pressure_serial.is_open:
                    if self.pressure_serial.in_waiting:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"\n[{timestamp}] ===== 压力传感器数据 =====")
                        data = self.pressure_serial.readline().decode('utf-8').strip()
                        print(f"► 原始字节: {data}")
                        try:
                            decoded = data.strip()
                            print(f"► 解码数据: {decoded}")
                            if decoded:
                                buffer += decoded
                                while '\n' in buffer:
                                    line, buffer = buffer.split('\n', 1)
                                    line = line.strip()
                                    print(f"► 处理行: {line}")
                                    if line.startswith('ch0:'):
                                        values = re.findall(r'ch\d:(\d+)', line)
                                        if len(values) == 5:
                                            pressure_data = [int(v) for v in values]
                                            print(f"[压力传感器] 解析出的数据: {pressure_data}")
                                            self.data_queue.put(pressure_data)
                        except Exception as e:
                            print(f"[错误] 压力传感器数据解码失败: {e}")
                            buffer = ""
                            continue
                
                time.sleep(0.1)  # 短暂休眠，避免CPU占用过高
                    
            except Exception as e:
                print(f"串口读取错误: {e}")
                break
        
        print("=== 串口读取线程结束 ===\n")


    def stop_serial_thread(self):
        """停止串口数据读取线程"""
        self.thread_running = False
        if self.serial_thread:
            self.serial_thread.join(timeout=1.0)

    def toggle_connection(self):
        """切换串口连接状态"""
        if (self.angle_serial and self.angle_serial.is_open) or (self.pressure_serial and self.pressure_serial.is_open):
            try:
                self.stop_serial_thread()  # 停止数据读取线程
                if self.angle_serial and self.angle_serial.is_open:
                    self.angle_serial.close()
                if self.pressure_serial and self.pressure_serial.is_open:
                    self.pressure_serial.close()
                self.connect_btn.configure(text="连接")
                logging.info("串口已断开")
            except Exception as e:
                messagebox.showerror("错误", f"关闭串口时出错: {e}")
        else:
            try:
                angle_port = self.angle_port_var.get()
                pressure_port = self.pressure_port_var.get()
                
                if not angle_port and not pressure_port:
                    messagebox.showwarning("警告", "请至少选择一个串口")
                    return
                
                if angle_port:
                    self.angle_serial = serial.Serial(angle_port, 115200, timeout=0.1)
                    logging.info(f"角度数据串口{angle_port}已连接")
                
                if pressure_port:
                    self.pressure_serial = serial.Serial(pressure_port, 115200, timeout=0.1)
                    logging.info(f"压力数据串口{pressure_port}已连接")
                
                self.connect_btn.configure(text="断开")
                self.start_serial_thread()  # 启动数据读取线程
                
            except Exception as e:
                messagebox.showerror("错误", f"连接串口时出错: {e}")
                if self.angle_serial and self.angle_serial.is_open:
                    self.angle_serial.close()
                if self.pressure_serial and self.pressure_serial.is_open:
                    self.pressure_serial.close()
                self.connect_btn.configure(text="连接")

    def update(self):
        """更新函数"""
        if not self.updating:
            try:
                self.updating = True
                
                # 检查数据队列
                while not self.data_queue.empty():
                    try:
                        # 取出数据但不使用，因为实际数据已经在parse_serial_data中更新
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # 更新显示
                self.update_data_display()
                self.update_hand_model()
                
            except Exception as e:
                logging.error(f"更新时发生错误: {e}")
            finally:
                self.updating = False
                
        # 安排下一次更新
        self.root.after(self.update_interval, self.update)

    def update_data_display(self):
        """更新数据显示"""
        try:
            # 更新压力数据显示
            if hasattr(self, 'pressure_data') and isinstance(self.pressure_data, list):
                for i, (var, label) in enumerate(zip(self.pressure_vars, self.pressure_labels)):
                    if i < len(self.pressure_data):
                        value = self.pressure_data[i]
                        var.set(value)
                        label.config(text=str(value))
        
            # 更新角度数据显示
            if hasattr(self, 'angle_data') and isinstance(self.angle_data, dict):
                for finger_id, labels in self.angle_labels.items():
                    if finger_id in self.angle_data:
                        angles = self.angle_data[finger_id]
                        for i, label in enumerate(labels):
                            if i < len(angles):
                                label.config(text=f"{angles[i]:.1f}°")
        except Exception as e:
            logging.error(f"更新数据显示时出错: {e}")

    def init_data_tab(self):
        """初始化数据监控标签页"""
        # 创建数据显示标签页
        self.data_frame = ttk.Frame(self.frame_data)
        self.frame_data.add(self.data_frame, text="数据监控")
        
        # 创建左右分栏
        left_frame = ttk.Frame(self.data_frame)
        right_frame = ttk.Frame(self.data_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 创建压力数据显示区域
        pressure_frame = ttk.LabelFrame(left_frame, text="压力数据")
        pressure_frame.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)
        
        # 为每个手指创建滑动条和标签
        self.pressure_vars = []
        self.pressure_labels = []
        finger_names = ["大拇指", "食指", "中指", "无名指", "小指"]
        
        for i, name in enumerate(finger_names):
            # 创建行框架
            row_frame = ttk.Frame(pressure_frame)
            row_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # 添加手指名称标签
            ttk.Label(row_frame, text=f"{name}:", width=8).pack(side=tk.LEFT)
            
            # 创建压力值变量和标签
            pressure_var = tk.IntVar(value=0)
            self.pressure_vars.append(pressure_var)
            
            # 创建滑动条
            scale = ttk.Scale(row_frame, from_=0, to=1024, orient=tk.HORIZONTAL,
                            variable=pressure_var, state='readonly')
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # 创建数值标签
            value_label = ttk.Label(row_frame, text="0", width=6)
            value_label.pack(side=tk.LEFT)
            self.pressure_labels.append(value_label)
        
        # 创建角度数据显示区域
        angle_frame = ttk.LabelFrame(right_frame, text="角度数据")
        angle_frame.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)
        
        # 为每个手指创��角度显示
        self.angle_labels = {}
        
        for i, name in enumerate(finger_names):
            # 创建手指框架
            finger_frame = ttk.LabelFrame(angle_frame, text=name)
            finger_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # 创建关节角度标签
            labels = []
            num_joints = 2 if i == 0 else 3  # 大拇指2个关节，其他手指3个关节
            
            for j in range(num_joints):
                label_frame = ttk.Frame(finger_frame)
                label_frame.pack(fill=tk.X, padx=2, pady=1)
                
                ttk.Label(label_frame, text=f"关节{j+1}:").pack(side=tk.LEFT, padx=2)
                value_label = ttk.Label(label_frame, text="0°", width=8)
                value_label.pack(side=tk.LEFT)
                labels.append(value_label)
            
            self.angle_labels[f"finger_{i}"] = labels
            
    def get_available_ports(self):
        """获取可用的串口列表"""
        available_ports = []
        for port in serial.tools.list_ports.comports():
            # 只检查是否存在，不尝试打开
            available_ports.append(port.device)
            logging.debug(f"发现串口: {port.device}")
        return available_ports

    def refresh_ports(self):
        """刷新串口列表"""
        # 获取当前可用串口
        self.available_ports = self.get_available_ports()
        
        # 更新下拉框的值
        self.angle_port_combo['values'] = self.available_ports
        self.pressure_port_combo['values'] = self.available_ports
        
        # 如果当前选中的串口不在可用列表中，选择第一个可用串口
        if self.angle_port_var.get() not in self.available_ports and self.available_ports:
            self.angle_port_var.set(self.available_ports[0])
        if self.pressure_port_var.get() not in self.available_ports and self.available_ports:
            self.pressure_port_var.set(self.available_ports[0])
        
        # 如果没有可用串口，禁用连接按钮
        if not self.available_ports:
            self.connect_btn.configure(state='disabled')
            self.angle_port_combo.configure(state='disabled')
            self.pressure_port_combo.configure(state='disabled')
        else:
            self.connect_btn.configure(state='normal')
            self.angle_port_combo.configure(state='normal')
            self.pressure_port_combo.configure(state='normal')

    def init_control_panel(self):
        """初始化控制面板"""
        # 创建按钮框架
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 角度数据串口选择框架
        angle_port_frame = ttk.LabelFrame(btn_frame, text="角度数据串口")
        angle_port_frame.pack(side=tk.LEFT, padx=5)
        
        self.angle_port_var = tk.StringVar()
        self.angle_port_combo = ttk.Combobox(angle_port_frame, textvariable=self.angle_port_var, width=10)
        self.angle_port_combo.pack(side=tk.LEFT, padx=2)
        
        # 压力数据串口选择框架
        pressure_port_frame = ttk.LabelFrame(btn_frame, text="压力数据串口")
        pressure_port_frame.pack(side=tk.LEFT, padx=5)
        
        self.pressure_port_var = tk.StringVar()
        self.pressure_port_combo = ttk.Combobox(pressure_port_frame, textvariable=self.pressure_port_var, width=10)
        self.pressure_port_combo.pack(side=tk.LEFT, padx=2)
        
        # 刷新串口按钮
        self.refresh_btn = ttk.Button(btn_frame, text="刷新", command=self.refresh_ports)
        self.refresh_btn.pack(side=tk.LEFT, padx=2)
        
        # 连接按钮
        self.connect_btn = ttk.Button(btn_frame, text="连接", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=2)
        
        # 添加校准按钮
        self.calib_frame = ttk.LabelFrame(self.control_frame, text="角度校准")
        self.calib_frame.pack(side=tk.LEFT, padx=5)
        
        self.min_btn = ttk.Button(self.calib_frame, text="设置最小值", command=self.set_min_values)
        self.min_btn.pack(side=tk.LEFT, padx=5)
        
        self.max_btn = ttk.Button(self.calib_frame, text="设置最大值", command=self.set_max_values)
        self.max_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_calib_btn = ttk.Button(self.calib_frame, text="重置校准", command=self.reset_calibration)
        self.reset_calib_btn.pack(side=tk.LEFT, padx=5)
        
        # 更新频率选择
        freq_frame = ttk.LabelFrame(btn_frame, text="更新频率(FPS)")
        freq_frame.pack(side=tk.LEFT, padx=5)
        
        self.fps_var = tk.StringVar(value="60")
        fps_combo = ttk.Combobox(freq_frame, textvariable=self.fps_var, values=["1", "5", "10", "30", "60"], width=5)
        fps_combo.pack(side=tk.LEFT, padx=2)
        fps_combo.bind("<<ComboboxSelected>>", self.update_frequency)
        
        # 初始化时刷新串口列表
        self.refresh_ports()
        
    def update_frequency(self, event=None):
        """更新刷新频率"""
        if not hasattr(self, 'update_locked') or not self.update_locked:
            self.update_locked = True
            try:
                fps = int(self.fps_var.get())
                if fps > 0:
                    self.update_interval = int(1000 / fps)
                self.root.after(100, self.release_update_lock)  # 100ms后释放锁定
            except ValueError:
                logging.error("无效的FPS值")
                self.fps_var.set("60")
                self.root.after(100, self.release_update_lock)

    def release_update_lock(self):
        """释放更新锁定"""
        self.update_locked = False

    def set_min_values(self):
        """设置当前角度值为最小值"""
        if not self.angle_serial or not self.angle_serial.is_open:
            messagebox.showwarning("警告", "请先连接串口")
            return
        self.calibration_min = self.current_raw_angles if hasattr(self, 'current_raw_angles') else [float('inf')] * 14
        logging.info("已设置最小值校准点")
        messagebox.showinfo("成功", "已设置最小值校准点")

    def set_max_values(self):
        """设置当前角度值为最大值"""
        if not self.angle_serial or not self.angle_serial.is_open:
            messagebox.showwarning("警告", "请先连接串口")
            return
        self.calibration_max = self.current_raw_angles if hasattr(self, 'current_raw_angles') else [float('-inf')] * 14
        self.is_calibrated = True
        logging.info("已设置最大值校准点")
        messagebox.showinfo("成功", "已设置最大值校准点")

    def reset_calibration(self):
        """重置校准数据"""
        self.calibration_min = [float('inf')] * 14
        self.calibration_max = [float('-inf')] * 14
        self.is_calibrated = False
        logging.info("已重置校准数据")
        messagebox.showinfo("成功", "已重置校准数据")

    def map_angle(self, value, idx):
        """将原始角度映射到0-90度范围"""
        if not self.is_calibrated:
            logging.debug(f"未校准，返回原始值: idx={idx}, value={value}")
            return value
        
        min_val = self.calibration_min[idx]
        max_val = self.calibration_max[idx]
        
        if min_val == float('inf') or max_val == float('-inf'):
            logging.debug(f"校准值无效: idx={idx}, min={min_val}, max={max_val}")
            return value
        
        # 确保不会出现除以零的情况
        if max_val == min_val:
            logging.debug(f"最大最小值相等: idx={idx}, value={max_val}")
            return 45  # 返回中间值
        
        mapped = (value - min_val) / (max_val - min_val) * 90
        mapped = max(0, min(90, mapped))
        logging.debug(f"角度映射: idx={idx}, raw={value}, min={min_val}, max={max_val}, mapped={mapped}")
        return mapped

    def parse_serial_data(self, data):
        """解析串口数据"""
        try:
            # 解析形如 C1=115.412,C2=34.382,...
            parts = data.strip().split(',')
            if len(parts) != 14:
                return None
                
            angles = []
            for part in parts:
                try:
                    value = float(part.split('=')[1])
                    angles.append(value)
                except (ValueError, IndexError):
                    return None
                    
            self.current_raw_angles = angles
            
            # 映射角度到0-90度范围
            mapped_angles = [self.map_angle(angle, i) for i, angle in enumerate(angles)]
            
            # 将映射后的角度分配给手指关节
            self.angle_data["finger_1"] = mapped_angles[0:3]   # 食指 (C1-C3)
            self.angle_data["finger_2"] = mapped_angles[3:6]   # 中指 (C4-C6)
            self.angle_data["finger_3"] = mapped_angles[6:9]   # 无名指 (C7-C9)
            self.angle_data["finger_4"] = mapped_angles[9:12]  # 小指 (C10-C12)
            self.angle_data["finger_0"] = mapped_angles[12:14] # 大拇指 (C13-C14)
            
            logging.info(f"原始角度数据: {angles}")
            logging.info(f"映射后角度数据: {mapped_angles}")
            logging.info(f"更新后的手指角度: {self.angle_data}")
            
            return True
        except Exception as e:
            logging.error(f"数据解析错误: {e}")
            return None

    def serial_read_thread(self):
        """串口数据读取线程"""
        while self.thread_running:
            if self.angle_serial and self.angle_serial.is_open:
                try:
                    if self.angle_serial.in_waiting:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"\n[{timestamp}] ===== 角度传感器数据 =====")
                        data = self.angle_serial.readline().decode('utf-8').strip()
                        print(f"► 原始字节: {data}")
                        if data:
                            if self.parse_serial_data(data):
                                self.data_queue.put(True)
                except Exception as e:
                    print(f"[错误] 角度传感器数据读取失败: {e}")
                    continue

    def update_visualization(self):
        """更新可视化"""
        try:
            # 检查队列中是否有新数据
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                if isinstance(data, list) and len(data) == 5:  # 压力数据
                    self.pressure_data = data
                elif data is True:  # 角度数据更新标志
                    pass  # 角度数据已经在parse_serial_data中更新
            
            # 使用最新的数据更新模型
            self.create_hand_model(angles=self.angle_data, pressures=self.pressure_data)
            
        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"更新可视化时出错: {e}")
        finally:
            # 继续更新
            if self.thread_running:
                self.root.after(16, self.update_visualization)  # 约60FPS

    def run(self):
        """运行应用程序"""
        self.update_visualization()  # 启动更新循环
        self.root.mainloop()
        
        # 程序结束时清理
        self.stop_serial_thread()  # 停止数据读取线程
        if hasattr(self, 'angle_serial') and self.angle_serial and self.angle_serial.is_open:
            self.angle_serial.close()

if __name__ == "__main__":
    app = DataGloveVisualizer()
    app.run()
