import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point
import numpy as np
import cv2
import os
def vis_line(save_path, *geometries):
    # 创建图形，设置图像大小
    plt.figure(figsize=(12*2, 8*2))  # 你可以根据需要调整大小

    # 遍历所有的几何对象
    point_index = 0
    for i, geom in enumerate(geometries):
        if isinstance(geom, LineString):
            # 提取 LineString 的坐标
            x, y = geom.xy
            # 绘制线段
            plt.plot(x, y, label=f'Line {i+1}')
            
            # 标注开头的点
            plt.scatter(x[0], y[0], color='red', marker='o', label=f'Start {i+1}' if i == 0 else "")
            
            # 标注结尾的点
            plt.scatter(x[-1], y[-1], color='blue', marker='x', label=f'End {i+1}' if i == 0 else "")
        
        elif isinstance(geom, Polygon):
            # 提取 Polygon 的外环坐标
            x, y = geom.exterior.xy
            # 绘制多边形
            plt.plot(x, y, label=f'Polygon {i+1}')
            
            # 标注多边形的第一个点
            plt.scatter(x[0], y[0], color='black', marker='o', label=f'Polygon Start {i+1}' if i == 0 else "")
        
        elif isinstance(geom, np.ndarray):
            x, y = geom[:, 0], geom[:, 1]
            plt.plot(x, y, label=f'Vector {i+1}')
            plt.scatter(x[0], y[0], color='red', marker='o', label=f'Start {i+1}' if i == 0 else "")
            plt.scatter(x[-1], y[-1], color='blue', marker='x', label=f'End {i+1}' if i == 0 else "")
        elif isinstance(geom, torch.Tensor):
            x, y = geom[:, 0].numpy(), geom[:, 1].numpy()
            plt.plot(x, y, label=f'Vector {i+1}')
            plt.scatter(x[0], y[0], color='red', marker='o', label=f'Start {i+1}' if i == 0 else "")
            plt.scatter(x[-1], y[-1], color='blue', marker='x', label=f'End {i+1}' if i == 0 else "")
        elif isinstance(geom, Point):
            plt.scatter(geom.x, geom.y, color='black', marker=f'${point_index+1}$', label=f'Point {point_index+1}')
            # plt.scatter(geom.x, geom.y, color='black', marker='o')
            # plt.text(geom.x, geom.y, f'{point_index+1}', fontsize=8, color='black', ha='center', va='center', label=f'Point {point_index+1}')
            point_index += 1

    # 添加图例
    plt.legend()

    # 设置标题和标签
    plt.title('Geometry Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 保存图像到本地
    plt.savefig(save_path, dpi=300)  # 设置 dpi 增加分辨率
    print('save to :', save_path)
    # 清除画布
    plt.clf()

def vis_mask(save_dir, masks):
    for label,results in masks.items():
        for i,result in enumerate(results):
            # 保存结果到本地图片
            save_path = os.path.join(save_dir, f'{label}_result_{i+1}.png')
            cv2.imwrite(save_path, result[:]*255)
            print(f'Saved result {i+1} to {save_path}')


import torch
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr