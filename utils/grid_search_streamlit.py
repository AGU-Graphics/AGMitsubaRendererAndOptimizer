import streamlit as st
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

import os
import time
from stqdm import stqdm

from src.loader import load_true_image_and_values

def render_and_compare(scene, params, param_key, true_image_np, sigma_t_value, spp):
    # Set the current parameter value
    params[param_key] = sigma_t_value
    params.update()

    # Render the scene with the new parameter value
    rendered_image = mi.render(scene, spp=spp)
    rendered_image_np = np.array(rendered_image)

    # Compute the difference between the rendered and true images
    diff = np.mean(np.abs(rendered_image_np - true_image_np))

    return diff, rendered_image_np

def grid_search_optimizer(scene, params, param_key, true_image_np, true_params_np, output_dir, grid_steps=10, spp=16):
    # Prepare the RGB grid search range for sigma_t
    grid_r = np.linspace(true_params_np[0] * 0.5, true_params_np[0] * 1.5, grid_steps)
    grid_g = np.linspace(true_params_np[1] * 0.5, true_params_np[1] * 1.5, grid_steps)
    grid_b = np.linspace(true_params_np[2] * 0.5, true_params_np[2] * 1.5, grid_steps)

    sigma_t_values = []
    diffs = []

    total_steps = grid_steps ** 3
    # Perform grid search over RGB values and display progress
    for r in stqdm(grid_r, desc="Grid Search Progress"):
        for g in grid_g:
            for b in grid_b:
                sigma_t_value = [r, g, b]
                diff, _ = render_and_compare(scene, params, param_key, true_image_np, sigma_t_value, spp)
                sigma_t_values.append(sigma_t_value)
                diffs.append(diff)

    sigma_t_values = np.array(sigma_t_values)
    diffs = np.array(diffs)

    # Save the results
    np.savetxt(f'{output_dir}/grid_search_results.csv', np.c_[sigma_t_values, diffs], delimiter=',')

    return sigma_t_values, diffs

def plot_results(sigma_t_values, diffs):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for RGB values vs. differences
    scatter = ax.scatter(sigma_t_values[:, 0], sigma_t_values[:, 1], sigma_t_values[:, 2], c=diffs, cmap='viridis')

    ax.set_xlabel('R Value')
    ax.set_ylabel('G Value')
    ax.set_zlabel('B Value')
    ax.set_title('Difference between Rendered and True Images')

    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)

def main():
    # Set the variant to use
    mi.set_variant('cuda_ad_rgb')

    # Parameter key to optimize
    param_key = 'Cube.interior_medium.sigma_t.value.value'

    # Sidebar inputs for Streamlit
    st.sidebar.title("Optimization Parameters")
    spp = st.sidebar.slider('Samples Per Pixel (SPP)', min_value=1, max_value=64, value=16)
    grid_steps = st.sidebar.slider('Grid Steps', min_value=3, max_value=20, value=10)

    # True Scene
    true_scene_path = st.sidebar.text_input(label='True Scene Path', value='scenes/true_scene.xml')
    true_image_path = st.sidebar.text_input(label='True Image CSV Path', value='inputs/true_image.csv')

    # Scene to optimize
    scene_path = 'scenes/scene.xml'

    st.title("Mitsuba Grid Search Optimization")

    if st.button('Run Grid Search'):
        # Load the scene
        try:
            scene = mi.load_file(scene_path)
        except Exception as e:
            st.error(f"Error loading scene: {e}")
            return

        # Traverse the scene to find the parameters
        params = mi.traverse(scene)

        if param_key not in params:
            st.error(f"Error: Parameter key '{param_key}' not found in the scene parameters.")
            return

        try:
            true_scene = mi.load_file(true_scene_path)
        except Exception as e:
            st.error(f"Error loading true scene: {e}")
            return

        # Load the true image and the true parameter values
        try:
            true_image_np, true_params_np = load_true_image_and_values(true_image_path, true_scene, param_key)
        except Exception as e:
            st.error(f"Error loading true image and parameters: {e}")
            return

        # Set the timestamp for the run
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f'outputs/GridSearch/spp_{spp}_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)

        # Render the true image
        true_image = mi.render(true_scene, spp=spp)
        mi.util.write_bitmap(f'{output_dir}/true_image.png', true_image, write_async=True)
        st.image(true_image, caption='True Image')

        # # Run grid search optimizer
        # sigma_t_values, diffs = grid_search_optimizer(scene, params, param_key, true_image_np, true_params_np, output_dir, grid_steps, spp)

        # # Plot the results in 3D space
        # plot_results(sigma_t_values, diffs)

if __name__ == "__main__":
    main()
