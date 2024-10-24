import gc
import torch
import wandb
import sys
import cma
import numpy as np

from calc import compute_loss, cosine_lr, compute_total_loss

def compute_gradient_for_param(scene, params, key, param_np, true_image_np, it, output_dir, spp, i, fd_eps, min_val, max_val):
    """
    Compute the gradient for a single parameter index using finite differences.
    Uses forward difference if at minimum, backward difference if at maximum, and central difference otherwise.

    Args:
        scene: The Mitsuba scene object.
        params: The parameters object from Mitsuba.
        key (str): The parameter key.
        param_np (np.ndarray): Current parameter values as a NumPy array.
        true_image_np: The true image data as a NumPy array.
        it (int): Current iteration number.
        output_dir (str): Directory to save outputs.
        spp (int): Samples per pixel.
        i (int): Index of the parameter to compute the gradient for.
        fd_eps (float): Finite difference epsilon for the parameter.
        min_val (float): Minimum allowed value for the parameter.
        max_val (float): Maximum allowed value for the parameter.

    Returns:
        float: The computed gradient for the parameter at index `i`.
    """
    original_value = param_np[i]

    # Compute f(x) - current loss
    f_x = compute_loss(scene, params, key, param_np, true_image_np, it, output_dir, spp)

    # Determine which finite difference to use
    if original_value <= min_val + 1e-8:
        # Forward difference
        param_np[i] = original_value + fd_eps
        params[key] = param_np
        params.update()
        f_x_plus = compute_loss(scene, params, key, param_np, true_image_np, it, output_dir, spp)

        # Reset to original value
        param_np[i] = original_value
        params[key] = param_np
        params.update()

        if not np.isfinite(f_x_plus):
            print(f"Invalid loss encountered at parameter '{key}', index {i} during forward difference.", file=sys.stderr)
            return 0.0  # Zero gradient if loss is invalid
        else:
            grad = (f_x_plus - f_x) / fd_eps
            return grad

    elif original_value >= max_val - 1e-8:
        # Backward difference
        param_np[i] = original_value - fd_eps
        params[key] = param_np
        params.update()
        f_x_minus = compute_loss(scene, params, key, param_np, true_image_np, it, output_dir, spp)

        # Reset to original value
        param_np[i] = original_value
        params[key] = param_np
        params.update()

        if not np.isfinite(f_x_minus):
            print(f"Invalid loss encountered at parameter '{key}', index {i} during backward difference.", file=sys.stderr)
            return 0.0  # Zero gradient if loss is invalid
        else:
            grad = (f_x - f_x_minus) / fd_eps
            return grad

    else:
        # Central difference
        # Compute f(x + epsilon)
        param_np[i] = original_value + fd_eps
        params[key] = param_np
        params.update()
        f_x_plus = compute_loss(scene, params, key, param_np, true_image_np, it, output_dir, spp)

        # Compute f(x - epsilon)
        param_np[i] = original_value - fd_eps
        params[key] = param_np
        params.update()
        f_x_minus = compute_loss(scene, params, key, param_np, true_image_np, it, output_dir, spp)

        # Reset to original value
        param_np[i] = original_value
        params[key] = param_np
        params.update()

        if not (np.isfinite(f_x_plus) and np.isfinite(f_x_minus)):
            print(f"Invalid loss encountered at parameter '{key}', index {i} during central difference.", file=sys.stderr)
            return 0.0  # Zero gradient if any loss is invalid
        else:
            grad = (f_x_plus - f_x_minus) / (2 * fd_eps)
            return grad

def adam_optimizer(config, scene, params, true_image_np, true_params_np, output_dir):
    """
    Perform optimization using the Adam optimizer with per-parameter settings.

    Args:
        config (dict): Optimization configuration.
        scene: The Mitsuba scene object.
        params: The parameters object from Mitsuba.
        true_image_np: The true image data as a NumPy array.
        true_params_np (dict): True parameter values as a dictionary.
        output_dir (str): Directory to save outputs.
    """
    # Initialize parameters
    param_keys = [param['param_key'] for param in config["parameters"]]
    param_initial_values = {param['param_key']: np.array(param['new_value']).flatten() for param in config["parameters"]}
    param_true_values = true_params_np  # Assuming it's a dict with param_key as keys

    # Initialize optimizer states for each parameter
    optimizer_states = {}
    for key in param_keys:
        optimizer_states[key] = {
            'm': np.zeros_like(param_initial_values[key]),
            'v': np.zeros_like(param_initial_values[key]),
            't': 0
        }

    # Set initial parameter values in the scene
    for key, value in param_initial_values.items():
        if key in params:
            params[key] = value
        else:
            print(f'Parameter key {key} not found in scene parameters.', file=sys.stderr)
            raise ValueError(f'Parameter key {key} not found in scene parameters.')
    params.update()

    # Extract Adam optimizer hyperparameters (common settings)
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    epsilon = config["epsilon"]
    iterations = config["iterations"]
    spp = config["spp"]
    convergence_threshold = config["convergence_threshold"]

    for it in range(1, iterations + 1):
        print(f"\n=== Iteration {it} ===")
        # Store the previous parameters for comparison
        prev_params = {key: np.array(params[key]).flatten() for key in param_keys}

        # Compute dynamic learning rate using cosine schedule (per parameter)
        lr_dict = {}
        for key in param_keys:
            param_index = param_keys.index(key)
            initial_lr = config["parameters"][param_index]["learning_rate"]
            lr = cosine_lr(initial_lr, it, iterations)
            lr_dict[key] = lr
            print(f"Learning rate for {key} (cosine decay): {lr}")

        # Initialize gradients dictionary
        gradients = {}

        # Compute gradients for each parameter
        for key in param_keys:
            param_index = param_keys.index(key)
            param_np = np.array(params[key]).flatten()
            grad = np.zeros_like(param_np)
            fd_eps = config["parameters"][param_index]["finite_difference_epsilon"]
            min_val = config["parameters"][param_index]["min_value"]
            max_val = config["parameters"][param_index]["max_value"]

            for i in range(len(param_np)):
                # Compute gradient using the new finite difference method
                gradient = compute_gradient_for_param(
                    scene, params, key, param_np, true_image_np, it, output_dir, spp,
                    i, fd_eps, min_val, max_val
                )
                grad[i] = gradient

            gradients[key] = grad

        # Update parameters using Adam optimizer with per-parameter settings
        for key in param_keys:
            optimizer_states[key]['t'] += 1
            m = optimizer_states[key]['m']
            v = optimizer_states[key]['v']
            g = gradients[key]

            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * g
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (g ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1 ** optimizer_states[key]['t'])
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2 ** optimizer_states[key]['t'])

            # Compute parameter update
            lr = lr_dict[key]
            param_update = lr * m_hat / (np.sqrt(v_hat) + epsilon)
            param_np = np.array(params[key]).flatten() - param_update

            # Clamp the parameter to the specified range
            param_index = param_keys.index(key)
            min_val = config["parameters"][param_index]["min_value"]
            max_val = config["parameters"][param_index]["max_value"]
            param_np = np.clip(param_np, min_val, max_val)

            # Update the parameter in the scene
            params[key] = param_np
            params.update()

            print(f"Updated parameters for {key}: {param_np}")
            print(f"{key} parameters vs True: {param_np - param_true_values[key]}")

        # Compute relative change in parameters for convergence check
        rel_change = np.array([np.abs(params[key] - prev_params[key]) / (np.abs(prev_params[key]) + epsilon) for key in param_keys])

        # Compute total loss across all parameters
        total_loss = compute_total_loss(scene, params, true_image_np, spp)
        print(f"Total Loss: {total_loss}")

        # Check convergence based on relative change
        if np.all(rel_change < convergence_threshold):
            print(f"Convergence criterion met at iteration {it}.")
            for key in param_keys:
                print(f"Fitted parameter '{key}': {params[key]}")
                print(f"True parameter '{key}': {param_true_values[key]}")
            # Log convergence information to wandb
            wandb.log({"iteration": it, "convergence": True, "total_loss": total_loss})
            break

        # Log parameters and their differences to wandb
        log_dict = {"iteration": it, "total_loss": total_loss}
        for key in param_keys:
            np_param = np.array(params[key]).flatten()
            true_param = param_true_values[key]

            # Check if parameter is a vector (like RGB)
            if len(np_param) > 1:
                # Log each component separately
                for i, (param_val, true_val) in enumerate(zip(np_param, true_param)):
                    component = ['r', 'g', 'b'][i] if i < 3 else str(i)  # Handle RGB or other vectors
                    log_dict[f'{key}_{component}'] = float(param_val)
                    log_dict[f'{key}_diff_{component}'] = float(abs(param_val - true_val))
            else:
                # Log the scalar parameter and its difference
                log_dict[f'{key}'] = float(np_param[0])
                log_dict[f'{key}_diff'] = float(abs(np_param[0] - true_param[0]))

        wandb.log(log_dict)

    # Finally delete all items used in the optimization
    del scene, params, true_image_np, true_params_np
    gc.collect()
    torch.cuda.empty_cache()

    print('\nOptimization complete.')



def cma_es_optimizer(scene, params, config, true_image_np, true_params_np, key, output_dir):
    # Initialize the parameter
    param_initial = np.array(params[key]).flatten()
    
    print(f"[Optimizer CMA-ES] Initial parameters: {param_initial}")

    # CMA-ES options taken from config
    sigma0 = config["sigma0"]
    options = {
        'maxiter': config["iterations"],
        'popsize': config["popsize"],
        'tolx': config["tolx"],
        'tolfun': config["tolfun"],
        'verb_disp': config["verbosity"],
        'CMA_diagonal': config["CMA_diagonal"],
        'bounds': config["bounds"],
        'seed': config["seed"]
    }

    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(param_initial, sigma0, options)

    # Initialize iteration counter
    it = 0
    prev_solution = param_initial.copy()  # Keep track of the previous solution

    while not es.stop():
        solutions = es.ask()
        losses = []
        for solution in solutions:
            it += 1
            loss = compute_loss(scene, params, key, solution, true_image_np, it, output_dir, config["spp"])
            losses.append(loss)

            # Log intermediate results every 50 iterations
            if it % 50 == 0:
                param_dict = {f'param_{i}': solution[i] for i in range(len(solution))}
                param_diff_dict = {f'param_diff_{i}': solution[i] - true_params_np[i] for i in range(len(solution))}
                wandb.log({
                    "loss": loss,
                    **param_dict,
                    **param_diff_dict
                })

        # Tell CMA-ES about the losses
        es.tell(solutions, losses)
        es.disp()

        # Calculate relative change between current best solution and the previous solution
        current_solution = es.result.xbest
        rel_change = np.abs(current_solution - prev_solution) / (np.abs(prev_solution) + config["epsilon"])

        # Check for convergence based on the relative change
        if np.all(rel_change < config["convergence_threshold"]):
            print(f"Convergence criterion met at iteration {it}.")
            print(f"Best parameters: {current_solution}")
            wandb.log({"iteration": it, "convergence": True})
            break

        # Update the previous solution
        prev_solution = current_solution.copy()

    print('\nOptimization complete with CMA-ES.')

    # Get the best solution
    best_solution = es.result.xbest
    print('Best parameter values found by CMA-ES:')
    print(best_solution)

    # Update the scene parameters with the optimized values
    params[key] = best_solution
    params.update()
