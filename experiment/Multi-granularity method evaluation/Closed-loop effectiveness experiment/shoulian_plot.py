import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio

# 尝试导入 pygmo
try:
    import pygmo as pg

    HAS_PYGMO = True
except ImportError:
    HAS_PYGMO = False
    print("⚠️ 未检测到 pygmo 库，将跳过超体积 (HV) 计算。")

# ================= KDD 绘图标准设置 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ================= 配色方案 (加深版) =================
COLOR_PARETO = '#E67E22'  # 橙色
COLOR_POP = '#4682B4'  # 蓝色

# ================= 配置区域 =================
INPUT_JSON_PATH = r"experiment\多粒度方法评估\闭环有效性实验\验证通过\收敛性验证通过\低逆反\result\进化\实验数据\experiment_results.json"
OUTPUT_DIR = r"experiment\多粒度方法评估\闭环有效性实验\output"

GIF_DURATION = 0.5
KPI_KEYS = ['safety', 'creativity', 'satisfaction']


class EvolutionVisualizer:
    def __init__(self, json_path, output_dir=""):
        self.json_path = json_path
        if not output_dir:
            self.output_dir = os.path.join(os.path.dirname(json_path), "re_draw_output")
        else:
            self.output_dir = output_dir

        self.img_dir = os.path.join(self.output_dir, "3d_frames")
        self.data_dir = os.path.join(self.output_dir, "exported_data")
        self.data = None
        self.history = []

        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def load_data(self):
        print(f"📂 正在读取数据: {self.json_path}")
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.history = self.data.get("evolution_history", [])
            if not self.history:
                raise ValueError("JSON 文件中未找到 'evolution_history' 字段")
            print(f"✅ 数据加载成功，共 {len(self.history)} 代。")
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            exit(1)

    def _process_generation_data(self, gen_data):
        processed = []
        for ind in gen_data:
            kpi = ind.get('kpi', {})
            entry = {
                'safety': -kpi.get('safety', 0),
                'creativity': -kpi.get('creativity', 0),
                'satisfaction': -kpi.get('satisfaction', 0),
                'rank': ind.get('rank', 999)
            }
            processed.append(entry)
        return pd.DataFrame(processed)

    def export_detailed_history(self):
        """将每一代所有个体的详细数据导出为 CSV"""
        print("💾 正在导出全量历史数据...")
        all_frames = []
        for i, gen_data in enumerate(self.history):
            df = self._process_generation_data(gen_data)
            df['generation'] = i
            all_frames.append(df)

        full_df = pd.concat(all_frames, ignore_index=True)
        # 调整列顺序
        cols = ['generation', 'rank', 'safety', 'creativity', 'satisfaction']
        full_df = full_df[cols]

        save_path = os.path.join(self.data_dir, "evolution_detailed_history.csv")
        full_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ 详细历史数据已保存至: {save_path}")

    def _setup_3d_axis(self, ax, title_text, subplot_label):
        ax.set_xlabel('Safety', labelpad=5, fontweight='bold')
        ax.set_ylabel('Creativity', labelpad=5, fontweight='bold')
        ax.set_zlabel('Satisfaction', labelpad=5, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_zlim(0, 1.05)
        ax.view_init(elev=25, azim=130)
        ax.text2D(0.5, -0.15, f"{subplot_label} {title_text}",
                  transform=ax.transAxes, ha='center', va='top',
                  fontsize=11, fontweight='bold', color='black')

    def draw_3d_scatter_frames(self):
        print("🎨 正在生成 GIF 帧...")
        for i, gen_data in enumerate(self.history):
            df = self._process_generation_data(gen_data)
            elite = df[df['rank'] == 1]
            others = df[df['rank'] > 1]
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(others['safety'], others['creativity'], others['satisfaction'],
                       c=COLOR_POP, alpha=0.4, s=25, edgecolors='none')
            ax.scatter(elite['safety'], elite['creativity'], elite['satisfaction'],
                       c=COLOR_PARETO, alpha=1.0, s=60, edgecolor='white', linewidth=0.8)
            ax.set_title(f'Generation {i}', fontsize=12)
            ax.set_xlim(0, 1.05);
            ax.set_ylim(0, 1.05);
            ax.set_zlim(0, 1.05)
            ax.set_xlabel('Safety');
            ax.set_ylabel('Creativity');
            ax.set_zlabel('Satisfaction')
            ax.view_init(elev=25, azim=130)
            plt.savefig(os.path.join(self.img_dir, f"gen_{i:03d}.png"), dpi=100)
            plt.close(fig)

    def draw_static_comparison(self):
        print("📊 正在绘制首末代对比图...")
        gen_indices = [0, len(self.history) - 1]
        titles = ["Initial Generation", "Converged Generation"]
        labels = ["(a)", "(b)"]
        fig = plt.figure(figsize=(7.2, 3.6), dpi=300)
        for idx, gen_idx in enumerate(gen_indices):
            ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
            df = self._process_generation_data(self.history[gen_idx])
            elite = df[df['rank'] == 1]
            others = df[df['rank'] > 1]
            ax.scatter(others['safety'], others['creativity'], others['satisfaction'],
                       c=COLOR_POP, alpha=0.6, s=25, edgecolors='none', depthshade=True,
                       label='Dominated Solutions' if idx == 0 else "")
            ax.scatter(elite['safety'], elite['creativity'], elite['satisfaction'],
                       c=COLOR_PARETO, alpha=1.0, s=60, edgecolor='white', linewidth=0.4,
                       depthshade=False, zorder=200, label='Non-dominated (Pareto)' if idx == 0 else "")
            self._setup_3d_axis(ax, titles[idx], labels[idx])
            ax.xaxis.pane.fill = False;
            ax.yaxis.pane.fill = False;
            ax.zaxis.pane.fill = False
            ax.grid(True, linestyle='--', alpha=0.4, color='gray')
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=9, handletextpad=0.1)
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.15, wspace=0.05, top=0.9)
        plt.savefig(os.path.join(self.output_dir, "kdd_3d_evolution_compare.pdf"), format='pdf', dpi=600)
        plt.savefig(os.path.join(self.output_dir, "kdd_3d_evolution_compare.png"), format='png', dpi=600)

    def create_gif(self):
        print("🎞️ 正在合成 GIF...")
        gif_path = os.path.join(self.output_dir, "evolution_process.gif")
        images = []
        files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        if not files: return
        for filename in files:
            images.append(imageio.imread(filename))
        imageio.mimsave(gif_path, images, duration=GIF_DURATION, loop=0)

    def draw_performance_metrics(self):
        """绘制性能指标图并导出数据"""
        print("📈 正在计算性能指标...")
        generations = range(len(self.history))
        elite_counts = []
        hypervolumes = []
        ref_point = [0.0, 0.0, 0.0]

        for gen_data in self.history:
            ranks = [ind.get('rank', 999) for ind in gen_data]
            elite_counts.append(ranks.count(1))
            if HAS_PYGMO:
                elite_inds = [ind for ind in gen_data if ind.get('rank') == 1]
                if not elite_inds:
                    hypervolumes.append(0.0)
                    continue
                points = [[ind.get('kpi', {}).get(k, 0) for k in KPI_KEYS] for ind in elite_inds]
                try:
                    hv_algo = pg.hypervolume(points)
                    hypervolumes.append(hv_algo.compute(ref_point))
                except:
                    hypervolumes.append(0.0)
            else:
                hypervolumes.append(0.0)

        # 导出指标数据
        metrics_df = pd.DataFrame({
            'generation': generations,
            'pareto_front_size': elite_counts,
            'hypervolume': hypervolumes
        })
        metrics_path = os.path.join(self.data_dir, "evolution_performance_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"✅ 性能指标数据已保存至: {metrics_path}")

        # 绘图逻辑保持不变
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=300, sharex=True)
        ax1.plot(generations, elite_counts, 'o-', color=COLOR_PARETO, markersize=4, linewidth=1.8,
                 label='Pareto Front Size')
        ax1.set_ylabel('Pareto Size');
        ax1.grid(True, linestyle=':', alpha=0.5);
        ax1.legend(loc='lower right', frameon=True, fontsize=9)
        ax1.spines['top'].set_visible(False);
        ax1.spines['right'].set_visible(False)
        if HAS_PYGMO:
            ax2.plot(generations, hypervolumes, 's-', color=COLOR_POP, markersize=4, linewidth=1.8,
                     label='Hypervolume (HV)')
            ax2.set_ylabel('Hypervolume');
            ax2.legend(loc='lower right', frameon=True, fontsize=9)
        else:
            ax2.text(0.5, 0.5, "HV Skipped", ha='center', va='center')
        ax2.set_xlabel('Generation');
        ax2.grid(True, linestyle=':', alpha=0.5);
        ax2.spines['top'].set_visible(False);
        ax2.spines['right'].set_visible(False)
        fig.text(0.5, 0.02, "(c) Performance Metrics over Generations", ha='center', fontsize=11, fontweight='bold')
        plt.tight_layout();
        plt.subplots_adjust(bottom=0.12)
        plt.savefig(os.path.join(self.output_dir, "kdd_performance_metrics.pdf"), dpi=600)
        plt.savefig(os.path.join(self.output_dir, "kdd_performance_metrics.png"), dpi=600)

    def run(self):
        self.load_data()
        self.export_detailed_history()  # 导出每一代详细数据
        self.draw_3d_scatter_frames()
        self.create_gif()
        self.draw_static_comparison()
        self.draw_performance_metrics()  # 内部包含指标数据导出
        print("\n🎉 全部任务完成！数据已导出至 exported_data 文件夹。")


if __name__ == "__main__":
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"❌ 文件不存在: {INPUT_JSON_PATH}")
    else:
        viz = EvolutionVisualizer(INPUT_JSON_PATH, OUTPUT_DIR)
        viz.run()