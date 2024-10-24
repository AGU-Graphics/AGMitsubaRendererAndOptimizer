import os
import subprocess
import json
import ast
import time

import numpy as np
import streamlit as st

def main():
    st.title('Optimizer using Virtual Scenes')

    # Number of parameters to optimize
    num_params = st.sidebar.number_input('Number of Parameters to Optimize', value=1, min_value=1, step=1)

    if 'output_dir' not in st.session_state:
        st.session_state['output_dir'] = None

    if 'checking_config' not in st.session_state:
        st.session_state['checking_config'] = False

    if 'opt_config_path' not in st.session_state:
        st.session_state['opt_config_path'] = None

    venv_path = st.sidebar.text_input('Path to Virtual Environment', value='.venv')

    # Helper function to construct Python executable path
    def get_python_exec():
        if os.name == 'nt':  # Windows
            return os.path.join(venv_path, 'Scripts', 'python.exe')
        else:  # Unix/Linux/Mac
            return os.path.join(venv_path, 'bin', 'python')

    python_exec = get_python_exec()
    if not os.path.exists(python_exec):
        st.error(f'Python executable not found at {python_exec}')
        return

    spp = st.sidebar.number_input('Samples per pixel (spp)', value=16, min_value=1, step=1)
    iterations = st.sidebar.number_input('Iterations', value=100, min_value=1, step=1)
    convergence_threshold = st.sidebar.number_input('Convergence Threshold', value=1e-4, min_value=0.0, step=1e-5, format="%.5f")
    learning_rate = st.sidebar.number_input('Learning Rate', value=0.002, min_value=0.0, step=0.001, format="%.4f")
    fd_epsilon = st.sidebar.number_input('Finite Difference Epsilon', value=0.004, min_value=0.0001, step=1e-3, format="%.4f")
    beta1 = st.sidebar.number_input('Beta1', value=0.9, min_value=0.0, max_value=1.0, step=0.01)
    beta2 = st.sidebar.number_input('Beta2', value=0.999, min_value=0.0, max_value=1.0, step=0.001)
    epsilon = st.sidebar.number_input('Epsilon', value=1e-8, min_value=1e-10, step=1e-9, format="%.9f")

    user_given_name = st.text_input('Experiment Name')

    # Initialize output_dir when user_given_name is provided
    if st.button('Initialize Output Directory'):
        if not user_given_name:
            st.error('Please provide an experiment name.')
            return
        st.session_state.timestamp = time.strftime("%Y%m%d-%H%M%S")
        st.session_state.output_dir = f'outputs/Adam/{user_given_name}_spp_{spp}_lr_{learning_rate}_{st.session_state.timestamp}'
        os.makedirs(st.session_state.output_dir, exist_ok=True)
        st.success(f'Output directory initialized: {st.session_state.output_dir}')

    if st.session_state.output_dir is not None:

        # Section 3: Select scenes
        st.header('Select Scenes and True Image')

        # Scan 'scenes/' directory for XML files
        scene_dir = 'scenes/'
        if not os.path.isdir(scene_dir):
            st.error(f"The directory '{scene_dir}' does not exist.")
            return

        xml_files = [f for f in os.listdir(scene_dir) if f.endswith('.xml')]
        if not xml_files:
            st.error('No XML files found in scenes/ directory.')
            return

        scene_file = st.selectbox('Select Scene to Optimize', xml_files)
        true_scene_file = st.selectbox('Select True Scene', xml_files)

        scene_path = os.path.join(scene_dir, scene_file)
        true_scene_path = os.path.join(scene_dir, true_scene_file)

        image_dir = 'inputs/'
        if not os.path.isdir(image_dir):
            st.error(f"The directory '{image_dir}' does not exist.")
            return

        csv_files = [f for f in os.listdir(image_dir) if f.endswith('.csv')]
        if not csv_files:
            st.error('No CSV files found in inputs/ directory.')
            return

        true_image_path = st.selectbox('Select True Image', csv_files)

        # Load the scenes and get parameters when a button is clicked
        if st.button('Load Scenes'):
            with st.spinner('Loading scenes and retrieving parameters...'):
                # Call get_parameters_from_scene.py via subprocess
                try:
                    result = subprocess.run(
                        [python_exec, 'src/get_parameters_from_scene.py', '--scene', scene_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if result.returncode != 0:
                        st.error(f'Error loading scene:\n{result.stderr}')
                        return
                    params = json.loads(result.stdout)
                    param_keys = list(params.keys())
                    st.session_state['params'] = params
                    st.session_state['param_keys'] = param_keys
                    st.session_state['scene_path'] = scene_path
                except Exception as e:
                    st.error(f'Error during subprocess execution: {e}')
                    return

                # Repeat for true_scene
                try:
                    result = subprocess.run(
                        [python_exec, 'src/get_parameters_from_scene.py', '--scene', true_scene_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if result.returncode != 0:
                        st.error(f'Error loading true scene:\n{result.stderr}')
                        return
                    true_params = json.loads(result.stdout)
                    st.session_state['true_params'] = true_params
                    st.session_state['true_scene_path'] = true_scene_path
                except Exception as e:
                    st.error(f'Error during subprocess execution: {e}')
                    return

            st.success('Scenes loaded successfully.')

        # Section 4: Select parameters to optimize
        st.header('Select Parameters to Optimize')

        # After selecting parameters to optimize
        if 'param_keys' in st.session_state:
            selected_params = st.multiselect(
                'Parameters to Optimize',
                options=st.session_state['param_keys'],
                default=[],
            )

            if len(selected_params) > num_params:
                st.error(f'You can only select up to {num_params} parameters.')
                return

            def format_value(value):
                if isinstance(value, (int, float)):
                    return f'{value:.4g}'
                elif isinstance(value, (list, tuple, np.ndarray)):
                    return [format_value(v) for v in value]
                return value

            def array_to_string(arr):
                if isinstance(arr, (list, tuple, np.ndarray)):
                    formatted = format_value(arr)
                    # Convert single-element arrays to scalar
                    if len(formatted) == 1:
                        return str(formatted[0]).replace("'", "")
                    return str(formatted).replace("'", "")
                return str(arr)

            param_new_values = {}
            param_learning_rates = {}
            param_fd_epsilons = {}
            param_ranges = {}

            for param in selected_params:
                current_value = st.session_state['params'][param]
                formatted_current = array_to_string(current_value)
                new_value_input = st.text_input(
                    f'New Initial Value for {param} (comma-separated for arrays)',
                    value=formatted_current,
                    key=f'new_value_{param}'
                )
                try:
                    new_value = ast.literal_eval(new_value_input)
                    if not isinstance(new_value, (int, float, list, tuple, np.ndarray)):
                        raise ValueError("Value must be a number or a list/tuple/array of numbers.")
                    param_new_values[param] = new_value
                except Exception as e:
                    st.error(f'Invalid input for new value of {param}: {e}')
                    return

                # Input for individual learning rate
                learning_rate_input = st.text_input(
                    f'Learning Rate for {param}',
                    value=str(learning_rate),
                    key=f'learning_rate_{param}'
                )
                try:
                    lr = float(learning_rate_input)
                    if lr <= 0:
                        raise ValueError("Learning rate must be positive.")
                    param_learning_rates[param] = lr
                except Exception as e:
                    st.error(f'Invalid learning rate for {param}: {e}')
                    return

                # Input for individual finite difference epsilon
                fd_epsilon_input = st.text_input(
                    f'Finite Difference Epsilon for {param}',
                    value=str(fd_epsilon),
                    key=f'fd_epsilon_{param}'
                )
                try:
                    fd_eps = float(fd_epsilon_input)
                    if fd_eps <= 0:
                        raise ValueError("Finite difference epsilon must be positive.")
                    param_fd_epsilons[param] = fd_eps
                except Exception as e:
                    st.error(f'Invalid finite difference epsilon for {param}: {e}')
                    return

                # Inputs for parameter range
                min_input = st.text_input(
                    f'Minimum Value for {param}',
                    value=str(0.0),  # Default min
                    key=f'min_value_{param}'
                )
                max_input = st.text_input(
                    f'Maximum Value for {param}',
                    value=str(float('inf')),  # Default max
                    key=f'max_value_{param}'
                )
                try:
                    min_val = float(min_input)
                    max_val = float(max_input)
                    if min_val > max_val:
                        raise ValueError("Minimum value cannot be greater than maximum value.")
                    param_ranges[param] = {"min": min_val, "max": max_val}
                except Exception as e:
                    st.error(f'Invalid range for {param}: {e}')
                    return
                
                st.divider()

            true_params = st.session_state['true_params']
            true_param_values = {}
            for param in selected_params:
                if param in true_params:
                    true_value = true_params[param]
                    formatted_true = array_to_string(true_value)
                    true_param_values[param] = true_value
                    st.write(f"True parameter value for {param}: {formatted_true}")
                else:
                    st.error(f"Parameter '{param}' not found in true scene parameters.")
                    return
            st.session_state['selected_params'] = selected_params
            st.session_state['param_new_values'] = param_new_values
            st.session_state['true_param_values'] = true_param_values
            st.session_state['param_learning_rates'] = param_learning_rates
            st.session_state['param_fd_epsilons'] = param_fd_epsilons
            st.session_state['param_ranges'] = param_ranges
        else:
            st.write('Please load scenes to select parameters.')
            return

        # Section 5: Check requirements
        if st.button('Update Settings and Check Requirements'):
            if all(key in st.session_state for key in ['params', 'selected_params', 'param_new_values', 'true_param_values', 'param_learning_rates', 'param_fd_epsilons', 'param_ranges']):
                # Prepare optimization config
                opt_config = {
                    "optimizer": "Adam with finite differences",
                    "learning_rate": learning_rate,  # This can be deprecated or kept as a default
                    "finite_difference_epsilon": fd_epsilon,  # Deprecated or default
                    "beta1": beta1,
                    "beta2": beta2,
                    "epsilon": epsilon,
                    "iterations": iterations,
                    "spp": spp,
                    "convergence_threshold": convergence_threshold,
                    "run_name": f"{user_given_name}_adam_spp_{spp}_lr_{learning_rate}_fd_{fd_epsilon}",
                    "output_dir": st.session_state.output_dir,
                    "user_given_name": user_given_name,
                    "timestamp": st.session_state.timestamp,
                    "parameters": []  # List of parameters with individual settings
                }

                for param in st.session_state['selected_params']:
                    opt_config["parameters"].append({
                        "param_key": param,
                        "new_value": st.session_state['param_new_values'][param],
                        "learning_rate": st.session_state['param_learning_rates'][param],
                        "finite_difference_epsilon": st.session_state['param_fd_epsilons'][param],
                        "min_value": st.session_state['param_ranges'][param]['min'],
                        "max_value": st.session_state['param_ranges'][param]['max']
                    })

                # Save opt_config to a JSON file to pass to the subprocess
                st.session_state.opt_config_path = os.path.join(st.session_state.output_dir, 'opt_config.json')
                try:
                    with open(st.session_state.opt_config_path, 'w') as f:
                        json.dump(opt_config, f, indent=4)
                except Exception as e:
                    st.error(f'Error saving optimization config: {e}')
                    return

                # Before running optimization, run check_requirements.py
                with st.spinner('Checking requirements...'):
                    try:
                        result = subprocess.run(
                            [
                                python_exec,
                                'src/check_requirements_virtual_scene.py',
                                '--scene', st.session_state['scene_path'],
                                '--true_scene', st.session_state['true_scene_path'],
                                '--true_image_path', os.path.join(image_dir, true_image_path),
                                '--opt_config', st.session_state.opt_config_path,
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        if result.returncode != 0:
                            st.error(f'Error during requirements check:\n{result.stderr}')
                            return
                        # Read check_results.json from output_dir
                        check_results_path = os.path.join(st.session_state.output_dir, 'check_results.json')
                        # Show the current contents of check_results.json
                        with open(st.session_state.opt_config_path, 'r') as f:
                            current_settings = json.load(f)
                        st.write("Current settings:")
                        st.write(current_settings)
                        if not os.path.exists(check_results_path):
                            st.error('Check results file not found.')
                            return
                        with open(check_results_path, 'r') as f:
                            check_results = json.load(f)
                        if not check_results.get('success', False):
                            errors = check_results.get('errors', [])
                            st.error('Errors during checks:\n' + '\n'.join(errors))
                            return
                        st.success('Requirements check passed!')
                        st.session_state['checking_config'] = True
                    except Exception as e:
                        st.error(f'Error during requirements check: {e}')
                        return
            else:
                st.error('Please ensure all necessary data is loaded before running optimization.')

        # Section 6: Run optimization
        if st.session_state.get('checking_config', False):
            if st.button('Run Optimization'):
                # If checks pass, proceed to run optimization
                with st.spinner('Running optimization...'):
                    try:
                        result = subprocess.run(
                            [
                                python_exec,
                                'src/optimize_with_adam_virtual_scene.py',
                                '--scene', st.session_state['scene_path'],
                                '--true_scene', st.session_state['true_scene_path'],
                                '--true_image_path', os.path.join(image_dir, true_image_path),
                                '--opt_config', st.session_state.opt_config_path,
                                '--venv_path', venv_path
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        if result.returncode != 0:
                            st.error(f'Error during optimization:\n{result.stderr}')
                            return
                        else:
                            st.success('Optimization completed successfully.')
                            st.text(result.stdout)
                            st.session_state.output_dir = None
                    except Exception as e:
                        st.error(f'Error during subprocess execution: {e}')
                        return

if __name__ == '__main__':
    main()
