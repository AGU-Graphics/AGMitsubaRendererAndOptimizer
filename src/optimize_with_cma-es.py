import mitsuba as mi
import numpy as np
import wandb
import os
import time

from optimizer import cma_es_optimizer
from loader import load_true_image_and_values

def main():
    # Set the variant to use
    mi.set_variant('cuda_ad_rgb')

    # Parameter key to optimize
    param_key = 'Cube.interior_medium.sigma_t.value.value'

    # Rendering parameters
    spp = 16

    # CMA-ES parameters
    iterations = 5000
    sigma0 = 1.0
    popsize = 4 + 3 * np.log(10)
    tolx = 1e-8
    tolfun = 1e-8
    verbosity = 1
    CMA_diagonal = False
    bounds = [0.0, None]
    seed = 42
    epsilon = 1e-8
    convergence_threshold = 1e-5

    # for the regularizer
    lambda_tv = 0.0

    # True Scene
    true_scene_path = 'scenes/true_scene.xml'
    true_image_path = 'inputs/true_image_16.csv'

    # Scene to optimize
    scene_path = 'scenes/scene.xml'

    opt_config = {
        "optimizer": "CMA-ES",
        "sigma0": sigma0,
        "popsize": popsize,
        "tolx": tolx,
        "tolfun": tolfun,
        "verbosity": verbosity,
        "CMA_diagonal": CMA_diagonal,
        "bounds": bounds,
        "seed": seed,
        "iterations": iterations,
        "spp": spp,
        "param_key": param_key,
        "epsilon": epsilon,
        "convergence_threshold": convergence_threshold,
        "lambda_tv": lambda_tv
    }

    
    # Initialize wandb
    wandb.init(
        project="mitsuba_optimization",
        config=opt_config,
        name=f"cma_spp_{spp}_sigma0_{sigma0}",
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

    if param_key not in params:
        print(f"Error: Parameter key '{param_key}' not found in the scene parameters.")
        wandb.finish()
        return
    
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

    # Set the timestamp for the run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f'outputs/CMA-ES/spp_{spp}_sigma0_{sigma0}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Run the optimizer
    cma_es_optimizer(opt_config, scene, params, true_image_np, true_params_np, param_key, output_dir)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
