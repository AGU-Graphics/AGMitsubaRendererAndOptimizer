import argparse
import json
import sys
import os
import gc
import torch

import mitsuba as mi

from loader import load_true_image_and_values

def main():
    parser = argparse.ArgumentParser(description='Check requirements before optimization.')
    parser.add_argument('--scene', type=str, required=True, help='Path to the Mitsuba scene XML file.')
    parser.add_argument('--true_scene', type=str, required=True, help='Path to the true Mitsuba scene XML file.')
    parser.add_argument('--true_image_path', type=str, required=True, help='Path to the true image CSV file.')
    parser.add_argument('--opt_config', type=str, required=True, help='Path to the optimization config JSON file.')
    # Removed --param_key and --new_value since parameters are in opt_config
    args = parser.parse_args()

    mi.set_variant('cuda_ad_rgb')
    
    try:
        with open(args.opt_config, 'r') as f:
            opt_config = json.load(f)
    except Exception as e:
        # Exit after logging debug information
        print(f'Error loading opt_config: {e}', file=sys.stderr)
        exit(1)


    # Initialize results dictionary
    results = {'success': True, 'errors': []}

    # Load the scene
    try:
        scene = mi.load_file(args.scene)
        params = mi.traverse(scene)
        del scene, params
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        results['success'] = False
        results['errors'].append(f'Error loading scene: {e}')
        check_results_path = os.path.join(opt_config['output_dir'], 'check_results.json')
        with open(check_results_path, 'w') as f:
            json.dump(results, f)
        exit(1)

    # Load true scene
    try:
        true_scene = mi.load_file(args.true_scene)
    except Exception as e:
        results['success'] = False
        results['errors'].append(f'Error loading true scene: {e}')
        check_results_path = os.path.join(opt_config['output_dir'], 'check_results.json')
        with open(check_results_path, 'w') as f:
            json.dump(results, f)
        exit(1)

    # Load the true image and true parameters
    try:
        true_image_path = args.true_image_path
        # Assuming load_true_image_and_values can handle multiple parameters
        param_keys = [param['param_key'] for param in opt_config['parameters']]
        true_image_np, true_params_np = load_true_image_and_values(
            true_image_path,
            true_scene,
            param_keys,
            output_dir=opt_config['output_dir']
        )
        if true_image_np is None or true_params_np is None:
            raise ValueError("Failed to load true image or true parameters.")
    except Exception as e:
        results['success'] = False
        results['errors'].append(f'Error loading true image and parameters: {e}')
        check_results_path = os.path.join(opt_config['output_dir'], 'check_results.json')
        with open(check_results_path, 'w') as f:
            json.dump(results, f)
        exit(1)

    # Render the true image
    try:
        true_image = (true_image_np * 255).astype('uint8')
        mi.util.write_bitmap(f"{opt_config['output_dir']}/true_image.png", true_image, write_async=True)
        del true_image
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        results['success'] = False
        results['errors'].append(f'Error rendering true image: {e}')
        check_results_path = os.path.join(opt_config['output_dir'], 'check_results.json')
        with open(check_results_path, 'w') as f:
            json.dump(results, f)
        exit(1)

    # If everything is successful, write success result
    results['success'] = True
    results['errors'] = []
    check_results_path = os.path.join(opt_config['output_dir'], 'check_results.json')
    with open(check_results_path, 'w') as f:
        json.dump(results, f)
        
    # Free memory
    del true_scene, true_image_np, true_params_np
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
