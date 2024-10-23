import mitsuba as mi
import numpy as np
import argparse
import json
import torch
import gc
import sys
import wandb

from optimizer import adam_optimizer
from loader import load_true_image_and_values

def main():
    parser = argparse.ArgumentParser(description='Run optimization with Adam.')
    parser.add_argument('--scene', type=str, required=True, help='Path to the Mitsuba scene XML file.')
    parser.add_argument('--true_scene', type=str, required=True, help='Path to the true Mitsuba scene XML file.')
    parser.add_argument('--true_image_path', type=str, required=True, help='Path to the true image CSV file.')
    parser.add_argument('--opt_config', type=str, required=True, help='Path to the optimization config JSON file.')
    parser.add_argument('--venv_path', type=str, default='.venv', help='Path to the virtual environment.')
    args = parser.parse_args()

    mi.set_variant('cuda_ad_rgb')

    # Load opt_config
    try:
        with open(args.opt_config, 'r') as f:
            opt_config = json.load(f)
    except Exception as e:
        print(f'Error loading optimization config: {e}', file=sys.stderr)
        exit(1)

    output_dir = opt_config['output_dir']

    # Load the scene
    try:
        scene = mi.load_file(args.scene)
        params = mi.traverse(scene)
    except Exception as e:
        print(f'Error loading scene: {e}', file=sys.stderr)
        exit(1)

    # Set initial parameter values
    try:
        for param in opt_config["parameters"]:
            param_key = param["param_key"]
            new_value = np.array(param["new_value"]).flatten()
            if param_key in params:
                params[param_key] = new_value
            else:
                print(f'Parameter key {param_key} not found in scene parameters.', file=sys.stderr)
                exit(1)
        params.update()
    except Exception as e:
        print(f'Error setting initial parameters: {e}', file=sys.stderr)
        exit(1)

    # Load true parameters
    try:
        true_scene = mi.load_file(args.true_scene)
        param_keys = [param['param_key'] for param in opt_config["parameters"]]
        true_image_np, true_params_np = load_true_image_and_values(
            args.true_image_path,
            true_scene,
            param_keys,
            output_dir
        )
        if true_image_np is None or true_params_np is None:
            raise ValueError("Failed to load true image or true parameters.")
    except Exception as e:
        print(f'Error loading true image and parameters: {e}', file=sys.stderr)
        exit(1)

    # Initialize wandb
    try:
        wandb.init(
            project="mitsuba_optimization_with_true_values",
            config=opt_config,
            name=opt_config.get('run_name', 'default_run'),
        )

        # Log the true parameters as a wandb config
        wandb.config.update({
            "true_parameters": {k: v.tolist() for k, v in true_params_np.items()}
        })
    except Exception as e:
        print(f'Error initializing wandb: {e}', file=sys.stderr)
        exit(1)

    # Run the optimizer
    try:
        adam_optimizer(config=opt_config, scene=scene, params=params, true_image_np=true_image_np, true_params_np=true_params_np, output_dir=output_dir)
        print('Optimization completed successfully.')
        # Free memory
        del scene, true_scene, params, true_image_np, true_params_np
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'Error during optimization: {e}', file=sys.stderr)
        wandb.finish()
        exit(1)

    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
