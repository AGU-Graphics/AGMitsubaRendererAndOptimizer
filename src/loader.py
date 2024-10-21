import mitsuba as mi
import numpy as np


def load_true_image_and_values(img_path, true_scene, key):
    # Load the true scene and extract the true parameters
    true_scene_params = mi.traverse(true_scene)
    true_params = true_scene_params[key]
    true_params_np = np.array(true_params).flatten()

    # Load the true image
    true_image_np = np.loadtxt(img_path, delimiter=',')
    film_size = true_scene.sensors()[0].film().size()
    true_image_np = true_image_np.reshape((film_size[1], film_size[0], 3))

    return true_image_np, true_params_np
