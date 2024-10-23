import os

import mitsuba as mi
import numpy as np


def load_true_image_and_values(img_path, true_scene, param_keys, output_dir):
    """
    Loads the true image and the true parameter values for the specified parameter keys.

    Args:
        img_path (str): Path to the true image CSV file.
        true_scene (mitsuba.Scene): The true Mitsuba scene object.
        param_keys (list of str): List of parameter keys to extract true values for.

    Returns:
        tuple:
            - true_image_np (np.ndarray): The true image as a NumPy array with shape (height, width, 3).
            - true_param_values (dict): Dictionary mapping each parameter key to its true NumPy array value.

    If an error occurs during loading, returns (None, None) and logs the error to 'error_log.txt'.
    """
    log_path = os.path.join(output_dir, 'logs/log_load_true_image_and_values.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w') as f:
        f.write(f'params: {param_keys}\n')

    # Load the true image
    try:
        # Load the true image from CSV
        true_image_np = np.loadtxt(img_path, delimiter=',')

        # Retrieve the film size from the scene's sensor
        sensors = true_scene.sensors()
        if not sensors:
            raise ValueError("No sensors found in the true scene.")
        film_size = sensors[0].film().size()  # Returns (width, height)

        # Validate the shape of the loaded image
        expected_size = (film_size[1], film_size[0], 3)  # (height, width, channels)
        if true_image_np.size != np.prod(expected_size):
            raise ValueError(f"True image size does not match expected film size. "
                             f"Expected {expected_size}, got {true_image_np.shape}")

        # Reshape the flat image array to (height, width, 3)
        true_image_np = true_image_np.reshape(expected_size)
    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f'Error while loading true image: {e}\n')
        return None, None
    
    try:
        # Traverse the true scene parameters
        true_scene_params = mi.traverse(true_scene)

        # Extract true parameter values for each key
        true_param_values = {}
        for key in param_keys:
            if key in true_scene_params:
                true_params = true_scene_params[key]
                try:
                    true_param_values[key] = np.array(true_params).flatten()
                except:
                    raise ValueError(f"Failed to convert true parameter '{key}' to a NumPy array.")
            else:
                raise KeyError(f"Parameter '{key}' not found in true scene parameters.")
    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f'Error while loading parameters: {e}\n')
        return None, None

    return true_image_np, true_param_values
