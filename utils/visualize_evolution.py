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
    """Create a 3D scatter plot for single generation data and save it as an image."""
    if not gen_data:
        return

    # Prepare data
    df = pd.DataFrame([ind['kpi'] for ind in gen_data])
    # Restore negative KPIs to positive numbers for easier observation
    df['safety'] = -df['safety']
    df['creativity'] = -df['creativity']
    df['satisfaction'] = -df['satisfaction']
    df['rank'] = [ind['rank'] for ind in gen_data]

    # Find the Pareto front (rank=1)
    front = df[df['rank'] == 1]

    # Create 3D image
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    ax.scatter(df['safety'], df['creativity'], df['satisfaction'], c='gray', alpha=0.5, label='Population')
    # Highlight the Pareto front
    ax.scatter(front['safety'], front['creativity'], front['satisfaction'], c='red', s=60, edgecolor='black',
               label='Pareto Front (Rank 1)')

    # Set image attributes
    ax.set_xlabel('Safety (H)', fontweight='bold')
    ax.set_ylabel('Creativity (E)', fontweight='bold')
    ax.set_zlabel('Satisfaction (S)', fontweight='bold')
    ax.set_title(f'Generation {gen_num}\nElite Solutions: {len(front)}', fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.legend()
    ax.view_init(elev=20, azim=135)  # Adjust perspective
    plt.tight_layout()

    # Save image
    filepath = os.path.join(output_dir, f'generation_{gen_num:03d}.png')
    plt.savefig(filepath)
    plt.close(fig)
    log.info(f"Saved image: {filepath}")


def create_gif(image_folder, output_gif_path, duration=0.5):
    """Synthesize images in a folder into a GIF."""
    images = []
    filenames = sorted([fn for fn in os.listdir(image_folder) if fn.endswith('.png')])
    for filename in filenames:
        images.append(imageio.imread(os.path.join(image_folder, filename)))

    imageio.mimsave(output_gif_path, images, duration=duration, loop=0)
    log.info(f"GIF created: {output_gif_path}")


def plot_performance_metrics(history_data, output_dir):
    """Plot the curve of HV and the number of elite solutions changing with the number of generations."""
    generations = list(range(len(history_data)))
    front_sizes = []
    hvs = []

    for gen_data in history_data:
        front = [ind for ind in gen_data if ind['rank'] == 1]
        front_sizes.append(len(front))
        # Note: HV is calculated on negative KPIs
        kpi_vectors = [list(ind['kpi'].values()) for ind in front]
        if kpi_vectors:
            hv_calculator = pg.hypervolume(kpi_vectors)
            hvs.append(hv_calculator.compute([0, 0, 0]))
        else:
            hvs.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot the number of elite solutions
    ax1.plot(generations, front_sizes, marker='o', linestyle='-', color='b')
    ax1.set_ylabel('Pareto Front Size', fontweight='bold')
    ax1.set_title('Evolution Performance Metrics', fontsize=16)
    ax1.grid(True)

    # Plot hypervolume
    ax2.plot(generations, hvs, marker='s', linestyle='-', color='r')
    ax2.set_xlabel('Generation', fontweight='bold')
    ax2.set_ylabel('Hypervolume (HV)', fontweight='bold')
    ax2.grid(True)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'performance_metrics.png')
    plt.savefig(filepath)
    plt.close(fig)
    log.info(f"Saved performance curve graph: {filepath}")


def nsga_evaluation_data_draw_main(json_filepath, output_dir):
    """
    Main function, load data and generate all visualizations.
    json_filepath: address of the json file exported by the main function
    output_dir: output directory
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        log.error(f"Error: Cannot find file {json_filepath}")
        return

    history = data.get("evolution_history")
    if not history:
        log.error("Error: 'evolution_history' data not found in the JSON file.")
        return

    # Create output directory
    images_dir = os.path.join(output_dir, "generations")
    os.makedirs(images_dir, exist_ok=True)

    # 1. Generate static images for each generation
    for i, gen_data in enumerate(history):
        plot_generation_3d(gen_data, i, images_dir)

    # 2. Generate performance curve graphs
    plot_performance_metrics(history, output_dir)

    # 3. Synthesize static images into a GIF
    gif_path = os.path.join(output_dir, 'evolution_animation.gif')
    create_gif(images_dir, gif_path, duration=0.5)
