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
    parser.add_argument('--target_image_path', type=str, required=True, help='Path to the true image CSV file.')
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
        del params
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        results['success'] = False
        results['errors'].append(f'Error loading scene: {e}')
        check_results_path = os.path.join(opt_config['output_dir'], 'check_results.json')
        with open(check_results_path, 'w') as f:
            json.dump(results, f)
        exit(1)

    # Load the target image
    try:
        target_image_path = args.target_image_path
        # Assuming load_true_image_and_values can handle multiple parameters
        param_keys = [param['param_key'] for param in opt_config['parameters']]
        target_image_np, _ = load_true_image_and_values(
            target_image_path,
            scene,
            param_keys,
            output_dir=opt_config['output_dir']
        )
        if target_image_np is None:
            raise ValueError("Failed to load true image or true parameters.")
        del scene
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        results['success'] = False
        results['errors'].append(f'Error loading true image and parameters: {e}')
        check_results_path = os.path.join(opt_config['output_dir'], 'check_results.json')
        with open(check_results_path, 'w') as f:
            json.dump(results, f)
        exit(1)

    # Render the true image
    try:
        true_image = (target_image_np * 255).astype('uint8')
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
    del target_image_np
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
