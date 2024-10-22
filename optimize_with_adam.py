import mitsuba as mi
import wandb
import os
import gc
import torch
import time

from src.optimizer import adam_optimizer
from src.loader import load_true_image_and_values

def main():
    # Set the variant to use
    mi.set_variant('cuda_ad_rgb')

    # Parameter key to optimize
    param_key = 'Cube.interior_medium.sigma_t.value.value'

    # Rendering parameters
    spp = 462

    # Adam optimizer parameters
    learning_rate = 0.02
    fd_epsilon = 0.004
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    iterations = 5000
    convergence_threshold = 1e-4
    
    # for the regularizer
    lambda_tv = 0.0

    # True Scene
    true_scene_path = 'scenes/true_scene.xml'
    true_image_path = 'data/true_image.csv'

    # Scene to optimize
    scene_path = 'scenes/scene.xml'

    opt_config = {
        "optimizer": "Adam with finite differences",
        "learning_rate": learning_rate,
        "finite_difference_epsilon": fd_epsilon,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "iterations": iterations,
        "spp": spp,
        "parameter_key": param_key,
        "convergence_threshold": convergence_threshold,
        "lambda_tv": lambda_tv
    }
    
    # Initialize wandb
    wandb.init(
        project="mitsuba_optimization",
        config=opt_config,
        name=f"adam_spp_{spp}_lr_{learning_rate}_fd_{fd_epsilon}",
    )

    # Load the scene
    try:
        scene = mi.load_file(scene_path)
    except Exception as e:
        print(f"Error loading scene: {e}")
        wandb.finish()
        return

    # Traverse the scene to find the parameters
    params = mi.traverse(scene)

    print(f"\n[Main] Parameter key: {param_key}")
    if param_key not in params:
        print(f"Error: Parameter key '{param_key}' not found in the scene parameters.")
        wandb.finish()
        return
    else:
        print(f"Parameter '{param_key}' found.")

    try:
        true_scene = mi.load_file(true_scene_path)
    except Exception as e:
        print(f"Error loading true scene: {e}")
        wandb.finish()
        return

    # Load the true image and the true parameter values
    try:
        true_image_np, true_params_np = load_true_image_and_values(
            true_image_path,
            true_scene,
            param_key
        )
    except Exception as e:
        print(f"Error loading true image and parameters: {e}")
        wandb.finish()
        return

    print(f"\n[Main] True parameters: {true_params_np}")

    # Set the timestamp for the run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f'output/Adam/spp_{spp}_lr_{learning_rate}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Render the true image
    true_image = mi.TensorXf(true_image_np)
    mi.util.write_bitmap(f'{output_dir}/true_image.png', true_image, write_async=True)
    del true_image
    gc.collect()
    torch.cuda.empty_cache()

    # Log the true parameters as a wandb config
    wandb.config.update({
        "true_parameters": true_params_np.tolist()
    })

    # Run the optimizer
    adam_optimizer(opt_config, scene, params, true_image_np, true_params_np, param_key, output_dir)

    print('Final parameter values:')
    print(params[param_key])

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()