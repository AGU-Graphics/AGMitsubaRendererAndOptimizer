import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gc
import time
import drjit as dr
from tqdm import tqdm

from src.loader import load_true_image_and_values

def render_and_compare(scene, params, param_key, true_image_np, sigma_t_value, spp):
    # Set the current parameter value
    params[param_key] = mi.Color3f(sigma_t_value)
    params.update()

    # Render the scene with the new parameter value
    dr.sync_thread()
    rendered_image = mi.render(scene, params=params, spp=spp)
    dr.sync_thread()

    rendered_image_np = np.array(rendered_image)

    # Compute the Mean Squared Error (MSE) between the rendered and true images
    diff = np.mean((rendered_image_np - true_image_np) ** 2)

    del rendered_image, rendered_image_np
    gc.collect()
    torch.cuda.empty_cache()

    return diff

def grid_search_rgb_optimizer(scene, params, param_key, true_image_np, output_dir, grid_steps=10, spp=16):
    # Prepare the RGB grid search range for sigma_t
    grid_r = np.linspace(0.0, 1.0, grid_steps)
    grid_g = np.linspace(0.0, 1.0, grid_steps)
    grid_b = np.linspace(0.0, 1.0, grid_steps)

    sigma_t_values = []
    diffs = []

    # Perform grid search over RGB values with tqdm for progress
    for r in tqdm(grid_r, desc="Grid Search Progress (R)"):
        for g in tqdm(grid_g, desc="Grid Search Progress (G)", leave=False):
            for b in tqdm(grid_b, desc="Grid Search Progress (B)", leave=False):
                sigma_t_value = [r, g, b]
                diff = render_and_compare(scene, params, param_key, true_image_np, sigma_t_value, spp)
                sigma_t_values.append(sigma_t_value)
                diffs.append(diff)

    sigma_t_values = np.array(sigma_t_values)
    diffs = np.array(diffs)

    # Save the results
    # add the header to the csv file
    header = 'R Value, G Value, B Value, Difference'
    np.savetxt(f'{output_dir}/grid_search_rgb_results.csv', np.c_[sigma_t_values, diffs], delimiter=',', header=header, fmt='%.4f')

    return sigma_t_values, diffs

def grid_search_single_optimizer(scene, params, param_key, true_image_np, output_dir, grid_steps=10, spp=16):
    # Prepare the grid search range for a single sigma_t value
    grid_values = np.linspace(0.0, 1.0, grid_steps)

    sigma_t_values = []
    diffs = []

    # Perform grid search for a single value applied to all RGB channels
    for v in tqdm(grid_values, desc="Grid Search Progress (Single Value)"):
        sigma_t_value = [v, v, v]  # Apply the same value to R, G, and B
        diff = render_and_compare(scene, params, param_key, true_image_np, sigma_t_value, spp)
        sigma_t_values.append(sigma_t_value)
        diffs.append(diff)

    sigma_t_values = np.array(sigma_t_values)
    diffs = np.array(diffs)

    # Save the results
    header = 'R Value, G Value, B Value, Difference'
    np.savetxt(f'{output_dir}/grid_search_single_results.csv', np.c_[sigma_t_values, diffs], delimiter=',', header=header, fmt='%.4f')

    return sigma_t_values, diffs

def get_user_input(prompt, default_value):
    """ Helper function to get input from user with default value """
    user_input = input(f'{prompt} [Default: {default_value}]: ')
    if user_input.strip() == "":
        return default_value
    return user_input


def plot_2d_results(sigma_t_values, diffs, spp, grid, output_dir, single_value=True):
    """ Plot 2D results with Sigma_t values on the x-axis and loss (MSE) on the y-axis. """
    fig, ax = plt.subplots(figsize=(10, 6))

    if single_value:
        # For single value optimization, plot the single sigma_t value vs. difference
        ax.plot(sigma_t_values[:, 0], diffs, marker='o', linestyle='-', color='b')
        ax.set_xlabel('Sigma_t Value (R=G=B)', fontsize=12)
    else:
        # For RGB grid search, plot the average sigma_t value across RGB vs. difference
        avg_sigma_t_values = np.mean(sigma_t_values, axis=1)
        ax.plot(avg_sigma_t_values, diffs, marker='o', linestyle='-', color='b')
        ax.set_xlabel('Average Sigma_t Value (RGB)', fontsize=12)

    # Plot the minimum loss point in red
    min_diff_idx = np.argmin(diffs)
    min_sigma_t_value = sigma_t_values[min_diff_idx, 0] if single_value else np.mean(sigma_t_values[min_diff_idx])
    
    ax.plot(min_sigma_t_value, diffs[min_diff_idx], 'ro')  # Red point for the min diff
    ax.text(min_sigma_t_value, diffs[min_diff_idx], f'{min_sigma_t_value:.4f}', color='red', fontsize=10)

    # Update the title with the min loss and corresponding sigma value
    ax.set_title(f'Difference vs Sigma_t Values (SPP: {spp}, Grid: {grid})\nMin Diff: {diffs[min_diff_idx]:.4f} at Sigma_t: {min_sigma_t_value:.4f}', fontsize=14)

    ax.set_ylabel('Loss (MSE)', fontsize=12)

    # Save the plot
    plt.savefig(f'{output_dir}/spp_{spp}_grid_{grid}_2d_results.png')
    plt.show()


def plot_3d_results(sigma_t_values, diffs, spp, grid, output_dir):
    """ Plot 3D results with R, G, B values on the axes and loss (MSE) represented as color. """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(sigma_t_values[:, 0], sigma_t_values[:, 1], sigma_t_values[:, 2], c=diffs, cmap='viridis', alpha=0.3)

    # Set the labels for RGB values
    ax.set_xlabel('R Value')
    ax.set_ylabel('G Value')
    ax.set_zlabel('B Value')

    # Plot the min diff point
    min_diff_idx = np.argmin(diffs)
    ax.scatter(sigma_t_values[min_diff_idx, 0], sigma_t_values[min_diff_idx, 1], sigma_t_values[min_diff_idx, 2], 
               c='red', s=100, label='Min Diff Point', alpha=1.0)

    # Get the RGB values for the min diff point
    r_value = sigma_t_values[min_diff_idx, 0]
    g_value = sigma_t_values[min_diff_idx, 1]
    b_value = sigma_t_values[min_diff_idx, 2]

    # Set the title with Min Diff and RGB values
    ax.set_title(f'Difference between Rendered and True Images\nMin Diff: {diffs[min_diff_idx]:.4f} at (R: {r_value:.4f}, G: {g_value:.4f}, B: {b_value:.4f})', fontsize=14)

    # Add the color bar
    plt.colorbar(scatter, ax=ax)

    # Add the legend
    ax.legend()

    # Save the plot
    plt.savefig(f'{output_dir}/spp_{spp}_grid_{grid}_3d_results.png')
    plt.show()


def main():
    # Set the variant to use
    mi.set_variant('cuda_ad_rgb')

    # Parameter key to optimize
    param_key = 'Cube.interior_medium.sigma_t.value.value'

    # User inputs with default values
    spp = int(input('Enter Samples Per Pixel (SPP): '))
    grid_steps = int(input('Enter Grid Steps: '))
    
    # Default file paths
    default_true_scene_path = 'scenes/true_scene.xml'
    default_true_image_path = 'data/true_image.csv'
    
    true_scene_path = get_user_input('Enter True Scene Path', default_true_scene_path)
    true_image_path = get_user_input('Enter True Image CSV Path', default_true_image_path)

    # Scene to optimize
    scene_path = 'scenes/scene.xml'

    try:
        # Load the scene
        scene = mi.load_file(scene_path)
    except Exception as e:
        print(f"Error loading scene: {e}")
        return

    # Traverse the scene to find the parameters
    params = mi.traverse(scene)

    if param_key not in params:
        print(f"Error: Parameter key '{param_key}' not found in the scene parameters.")
        return

    try:
        # Load the true scene
        true_scene = mi.load_file(true_scene_path)
    except Exception as e:
        print(f"Error loading true scene: {e}")
        return

    # Load the true image and the true parameter values
    try:
        true_image_np, _ = load_true_image_and_values(true_image_path, true_scene, param_key)
    except Exception as e:
        print(f"Error loading true image and parameters: {e}")
        return

    # Set the timestamp for the run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f'output/GridSearch/spp_{spp}_grid_{grid_steps}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Reconstruct the true image from numpy array
    true_image = mi.TensorXf(true_image_np)
    mi.util.write_bitmap(f'{output_dir}/true_image.png', true_image, write_async=True)

    # Ask the user which optimizer to use
    optimizer_choice = input("Choose optimizer type (1: RGB Grid Search, 2: Single Value Grid Search): ").strip()

    if optimizer_choice == '1':
        # Run RGB grid search optimizer and use 3D plot
        sigma_t_values, diffs = grid_search_rgb_optimizer(scene, params, param_key, true_image_np, output_dir, grid_steps, spp)
        plot_3d_results(sigma_t_values, diffs, spp, grid_steps, output_dir)
    elif optimizer_choice == '2':
        # Run Single Value grid search optimizer and use 2D plot
        sigma_t_values, diffs = grid_search_single_optimizer(scene, params, param_key, true_image_np, output_dir, grid_steps, spp)
        plot_2d_results(sigma_t_values, diffs, spp, grid_steps, output_dir, single_value=True)
    else:
        print("Invalid optimizer choice. Exiting.")
        return


if __name__ == "__main__":
    main()
