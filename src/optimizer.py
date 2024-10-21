import numpy as np
import wandb
import cma

from src.calc import compute_loss, cosine_lr

def adam_optimizer(config, scene, params, true_image_np, true_params_np, key, output_dir):
    # Initialize the parameter
    param_np = np.array(params[key]).flatten()
    print(f"[Optimizer] Initial parameters: {param_np}")

    # Adam optimizer hyperparameters
    initial_lr = config["learning_rate"]  # Use initial learning rate
    fd_epsilon = config["finite_difference_epsilon"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    epsilon = config["epsilon"]
    
    m = np.zeros_like(param_np)
    v = np.zeros_like(param_np)
    t = 0

    for it in range(1, config["iterations"] + 1):
        print(f"\n=== Iteration {it} ===")
        print(f"Current parameters: {param_np}")        
        # Store previous parameters
        prev_param_np = param_np.copy()

        # Compute dynamic learning rate using cosine schedule
        lr = cosine_lr(initial_lr, it, config["iterations"])
        print(f"Learning rate (cosine decay): {lr}")

        # Initialize gradient
        grad = np.zeros_like(param_np)

        # Compute gradient using finite differences
        for i in range(len(param_np)):
            original_value = param_np[i]

            # Compute loss for param + epsilon
            param_np[i] = original_value + fd_epsilon
            loss1 = compute_loss(scene, params, key, param_np, true_image_np, it, output_dir, config["spp"], config["lambda_tv"])

            # Compute loss for param - epsilon
            param_np[i] = original_value - fd_epsilon
            loss2 = compute_loss(scene, params, key, param_np, true_image_np, it, output_dir, config["spp"], config["lambda_tv"])

            # Reset the parameter
            param_np[i] = original_value

            # Check if losses are finite
            if not np.isfinite(loss1) or not np.isfinite(loss2):
                print(f"Invalid loss encountered at parameter index {i} during iteration {it}")
                grad[i] = 0.0  # Set gradient to zero if loss is invalid
            else:
                # Compute the gradient
                grad[i] = (loss1 - loss2) / (2 * fd_epsilon)

        # Update m and v (Adam updates)
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update parameters
        param_update = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        param_np = param_np - param_update

        # Clamp the parameter to ensure legal values
        param_np = np.clip(param_np, 0.0, None)

        print(f"Updated parameters: {param_np}")
        print(f"Parameters vs True: {param_np - true_params_np}")
        
        # Compute relative change
        rel_change = np.abs(param_np - prev_param_np) / (np.abs(prev_param_np) + epsilon)

        # Check convergence
        if np.all(rel_change < config["convergence_threshold"]):
            print(f"Convergence criterion met at iteration {it}.")
            print(f"Fitted parameters: {param_np}")
            print(f"True parameters: {true_params_np}")
            # Log convergence information to wandb
            wandb.log({"iteration": it, "convergence": True})
            break

        # Create dictionaries for individual parameter logging
        param_dict = {f'param_{i}': param_np[i] for i in range(len(param_np))}
        param_diff_dict = {f'param_diff_{i}': param_np[i] - true_params_np[i] for i in range(len(param_np))}

        # Log to wandb
        try:
            current_loss = compute_loss(scene, params, key, param_np, true_image_np, it, output_dir, config["spp"], config["lambda_tv"])
        except Exception as e:
            print(f"Error during loss computation at iteration {it}: {e}")
            current_loss = np.inf

        wandb.log({
            "iteration": it,
            "loss": current_loss,
            "learning_rate": lr,
            **param_dict,
            **param_diff_dict
        })

    print('\nOptimization complete.')

    # Update the scene parameters with the optimized values
    params[key] = param_np
    params.update()



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
            loss = compute_loss(scene, params, key, solution, true_image_np, it, output_dir, config["spp"], config["lambda_tv"])
            losses.append(loss)

            # Log intermediate results every 50 iterations
            if it % 50 == 0:
                param_dict = {f'param_{i}': solution[i] for i in range(len(solution))}
                param_diff_dict = {f'param_diff_{i}': solution[i] - true_params_np[i] for i in range(len(solution))}
                wandb.log({
                    "iteration": it,
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
