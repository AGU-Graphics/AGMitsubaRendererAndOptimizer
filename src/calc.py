
import gc
import torch
import wandb

import mitsuba as mi
import numpy as np

from src.render import renderer

def calc_mse(image1, image2):
    """
    Calculate the Mean Squared Error (MSE) between two images.

    Parameters:
    - image1: numpy array of shape (H, W, C)
    - image2: numpy array of shape (H, W, C)

    Returns:
    - mse: Mean Squared Error between the two images.
    """
    mse = np.mean((image1 - image2) ** 2)
    return mse

def calc_huber_loss(image1, image2, delta=0.05):
    """
    Calculate the Huber loss between two images.

    Parameters:
    - image1: numpy array of shape (H, W, C)
    - image2: numpy array of shape (H, W, C)
    - delta: Threshold for the Huber loss.

    Returns:
    - huber_loss: Huber loss between the two images.
    """
    diff = image1 - image2
    huber_loss = np.mean(np.where(np.abs(diff) < delta, 0.5 * diff ** 2, delta * (np.abs(diff) - 0.5 * delta)))
    return huber_loss

def total_variation(image):
    """
    Compute the total variation of an image.

    Parameters:
    - image: numpy array of shape (H, W, C)

    Returns:
    - tv: Total variation of the image.
    """
    tv = np.sum(np.abs(image[:-1, :, :] - image[1:, :, :])) + np.sum(np.abs(image[:, :-1, :] - image[:, 1:, :]))
    return tv

def compute_loss(scene, params, param_key, param_values, true_image_np, it, output_dir, spp=16, lambda_tv=0.0):
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
        image = np.array(image)

        #
        # # Compute the mean squared error loss
        # loss = calc_mse(image, true_image_np)
        #

        # Compute the Huber loss
        loss = calc_huber_loss(image, true_image_np, delta=0.05)

        # Compute Total Variation regularization
        tv_loss = total_variation(image)
        loss = loss + lambda_tv * tv_loss

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