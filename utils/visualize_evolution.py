import json
import os
import imageio
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pygmo as pg
import logging

log = logging.getLogger(__name__)


def plot_generation_3d(gen_data, gen_num, output_dir):
    """为单代数据创建一个3D散点图并保存为图片。"""
    if not gen_data:
        return

    # 准备数据
    df = pd.DataFrame([ind['kpi'] for ind in gen_data])
    # 将负数KPI恢复为正数以便于观察
    df['safety'] = -df['safety']
    df['creativity'] = -df['creativity']
    df['satisfaction'] = -df['satisfaction']
    df['rank'] = [ind['rank'] for ind in gen_data]

    # 找出帕累托前沿 (rank=1)
    front = df[df['rank'] == 1]

    # 创建3D图像
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制所有点
    ax.scatter(df['safety'], df['creativity'], df['satisfaction'], c='gray', alpha=0.5, label='Population')
    # 高亮帕累托前沿
    ax.scatter(front['safety'], front['creativity'], front['satisfaction'], c='red', s=60, edgecolor='black',
               label='Pareto Front (Rank 1)')

    # 设置图像属性
    ax.set_xlabel('Safety (H)', fontweight='bold')
    ax.set_ylabel('Creativity (E)', fontweight='bold')
    ax.set_zlabel('Satisfaction (S)', fontweight='bold')
    ax.set_title(f'Generation {gen_num}\nElite Solutions: {len(front)}', fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.legend()
    ax.view_init(elev=20, azim=135)  # 调整视角
    plt.tight_layout()

    # 保存图像
    filepath = os.path.join(output_dir, f'generation_{gen_num:03d}.png')
    plt.savefig(filepath)
    plt.close(fig)
    log.info(f"已保存图像: {filepath}")


def create_gif(image_folder, output_gif_path, duration=0.5):
    """将文件夹中的图片合成为GIF。"""
    images = []
    filenames = sorted([fn for fn in os.listdir(image_folder) if fn.endswith('.png')])
    for filename in filenames:
        images.append(imageio.imread(os.path.join(image_folder, filename)))

    imageio.mimsave(output_gif_path, images, duration=duration, loop=0)
    log.info(f"GIF 已创建: {output_gif_path}")


def plot_performance_metrics(history_data, output_dir):
    """绘制HV和精英解数量随代数变化的曲线图。"""
    generations = list(range(len(history_data)))
    front_sizes = []
    hvs = []

    for gen_data in history_data:
        front = [ind for ind in gen_data if ind['rank'] == 1]
        front_sizes.append(len(front))
        # 注意：HV是在负数KPI上计算的
        kpi_vectors = [list(ind['kpi'].values()) for ind in front]
        if kpi_vectors:
            hv_calculator = pg.hypervolume(kpi_vectors)
            hvs.append(hv_calculator.compute([0, 0, 0]))
        else:
            hvs.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制精英解数量
    ax1.plot(generations, front_sizes, marker='o', linestyle='-', color='b')
    ax1.set_ylabel('Pareto Front Size', fontweight='bold')
    ax1.set_title('Evolution Performance Metrics', fontsize=16)
    ax1.grid(True)

    # 绘制超体积
    ax2.plot(generations, hvs, marker='s', linestyle='-', color='r')
    ax2.set_xlabel('Generation', fontweight='bold')
    ax2.set_ylabel('Hypervolume (HV)', fontweight='bold')
    ax2.grid(True)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'performance_metrics.png')
    plt.savefig(filepath)
    plt.close(fig)
    log.info(f"已保存性能曲线图: {filepath}")


def nsga_evaluation_data_draw_main(json_filepath, output_dir):
    """
    主函数，加载数据并生成所有可视化。
    json_filepath: 主函数导出的json文件地址
    output_dir: 输出目录
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        log.error(f"错误: 找不到文件 {json_filepath}")
        return

    history = data.get("evolution_history")
    if not history:
        log.error("错误: JSON文件中未找到 'evolution_history' 数据。")
        return

    # 创建输出目录
    images_dir = os.path.join(output_dir, "generations")
    os.makedirs(images_dir, exist_ok=True)

    # 1. 生成每一代的静态图片
    for i, gen_data in enumerate(history):
        plot_generation_3d(gen_data, i, images_dir)

    # 2. 生成性能曲线图
    plot_performance_metrics(history, output_dir)

    # 3. 将静态图片合成为GIF
    gif_path = os.path.join(output_dir, 'evolution_animation.gif')
    create_gif(images_dir, gif_path, duration=0.5)


