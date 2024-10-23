
import gc
import torch
import wandb

import mitsuba as mi
import numpy as np

from render import renderer

def cosine_lr(initial_lr, t, T):
    """
    Cosine learning rate schedule.

    Parameters:
    - initial_lr: Initial learning rate.
    - t: Current iteration.
    - T: Total number of iterations.

    Returns:
    - Updated learning rate for the current iteration.
    """
    if T == 0:
        return initial_lr
    return initial_lr * 0.5 * (1 + np.cos(np.pi * t / T))


def compute_loss(scene, params, param_key, param_values, true_image_np, it, output_dir, spp=16):
    try:
        # Update the scene parameters
        params[param_key] = param_values
        params.update()

        # Render the image
        image = renderer(scene, params, spp=spp)

        # Save the image to disk for visualization (optional)
        if it % 50 == 0:
            mi.util.write_bitmap(f'{output_dir}/iter_{it:04d}.png', image, write_async=True)

        # Convert image to numpy array
        image_np = np.array(image)

        # Compute the mean squared error loss
        loss = np.mean((image_np - true_image_np) ** 2)

        # Delete the image to free up memory
        del image
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during compute_loss at iteration {it}: {e}")
        # Log the error to wandb
        wandb.log({"error": str(e), "iteration": it})
        # Return a high loss value to indicate failure
        loss = np.inf

    return loss


def compute_total_loss(scene, params, true_image_np, spp=16):
    try:
        # Render the scene with current parameters
        image = renderer(scene, params, spp=spp)

        # Convert the rendered image to a NumPy array
        image_np = np.array(image)

        # Compute the mean squared error loss between rendered and true images
        loss = np.mean((image_np - true_image_np) ** 2)

        # Clean up to free memory
        del image
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during compute_total_loss: {e}")
        # Log the error to wandb
        wandb.log({"error": str(e)})
        # Return a high loss value to indicate failure
        loss = np.inf

    return loss