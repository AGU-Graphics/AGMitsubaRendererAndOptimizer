import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import pyperclip
import os
import rawpy
import shutil
from PIL import Image, ExifTags
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
import sys

from streamlit import latex
from tqdm import tqdm

# ===============================
# Constants
# ===============================
CACHE_CSV_FILE_PATH = 'uploads/data.csv'
LIMIT_LUMINANCE = 2 ** 14 - 1
CACHE_COLUMNS = ["Shutter_Speed", "Max_Pixel_Luminance", "ZIP_File"]
INITIAL_PARAMS = [1, 1]

# ===============================
# Image Processing Functions
# ===============================

def extract_exif_data(image):
    """
    Extract EXIF data from an image.

    Parameters:
        image (PIL.Image): The image from which to extract EXIF data.

    Returns:
        dict: A dictionary of EXIF tags and their values.
    """
    try:
        exif_data = image._getexif()
        if exif_data is not None:
            return {
                ExifTags.TAGS.get(k, k): v
                for k, v in exif_data.items()
            }
    except Exception as e:
        st.error(f'Error extracting EXIF data: {e}')
    return None

def process_jpeg(file):
    """
    Process a JPEG file to extract the shutter speed.

    Parameters:
        file (str): Path to the JPEG file.

    Returns:
        float or None: The exposure time if available, else None.
    """
    try:
        jpeg_image = Image.open(file)
        jpeg_exif = extract_exif_data(jpeg_image)
        jpeg_image.close()
        if jpeg_exif:
            return float(jpeg_exif.get("ExposureTime", None))
    except Exception as e:
        st.error(f'Error processing JPEG: {e}')
    return None

def process_cr3(file):
    """
    Process a CR3 file to calculate the maximum pixel luminance.

    Parameters:
        file (str): Path to the CR3 file.

    Returns:
        tuple: Adjusted max R, G, B, and max luminance values.
    """
    try:
        with rawpy.imread(file) as raw:
            rgb_image = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
            return calculate_luminance(rgb_image)
    except Exception as e:
        st.error(f'Error processing CR3: {e}')
    return None, None, None, None

def calculate_luminance(rgb_image):
    """
    Calculate the luminance from an RGB image.

    Parameters:
        rgb_image (numpy.ndarray): The RGB image array.

    Returns:
        tuple: Adjusted max R, G, B, and max luminance values.
    """
    max_r = np.max(rgb_image[:, :, 0])
    max_g = np.max(rgb_image[:, :, 1])
    max_b = np.max(rgb_image[:, :, 2])
    scaling_factor = 255 / (2 ** 14 - 1)
    adjusted_max_r = max_r / scaling_factor
    adjusted_max_g = max_g / scaling_factor
    adjusted_max_b = max_b / scaling_factor
    max_luminance = np.max(0.2126 * (rgb_image[:, :, 0] / scaling_factor) +
                           0.7152 * (rgb_image[:, :, 1] / scaling_factor) +
                           0.0722 * (rgb_image[:, :, 2] / scaling_factor))
    return adjusted_max_r, adjusted_max_g, adjusted_max_b, max_luminance

# ===============================
# File Handling Functions
# ===============================

def extract_files_from_zip(zip_file_path, temp_dir):
    """
    Extract files from a ZIP archive.

    Parameters:
        zip_file_path (str): Path to the ZIP file.
        temp_dir (str): Temporary directory to extract files into.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

def create_file_pairs(temp_dir):
    """
    Create pairs of JPEG and CR3 files based on their base filenames.

    Parameters:
        temp_dir (str): Directory containing extracted files.

    Returns:
        list: List of tuples containing paired JPEG and CR3 file paths.
    """
    jpeg_files, cr3_files = {}, {}
    for file_name in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file_name)
        base_name, ext = os.path.splitext(file_name)
        ext = ext.lower()
        if ext in ['.jpg', '.jpeg']:
            jpeg_files[base_name] = file_path
        elif ext == '.cr3':
            cr3_files[base_name] = file_path
    return [(jpeg_files[base_name], cr3_files[base_name]) for base_name in jpeg_files.keys() if base_name in cr3_files]

# ===============================
# Data Processing Functions
# ===============================

def process_zip_files(zip_files, existing_df):
    """
    Process multiple ZIP files containing JPEG and CR3 images.

    Parameters:
        zip_files (list): List of ZIP file paths.
        existing_df (pandas.DataFrame): Existing DataFrame to append new data to.

    Returns:
        pandas.DataFrame: DataFrame containing all processed data.
    """
    all_data = existing_df.to_dict('records') if existing_df is not None else []
    existing_zip_files = get_existing_zip_files(existing_df)

    for zip_file in tqdm(zip_files):
        if os.path.basename(zip_file) in existing_zip_files:
            continue  # Skip already processed ZIP files

        temp_dir = f"temp_{os.path.basename(zip_file)}"
        extract_files_from_zip(zip_file, temp_dir)
        pairs = create_file_pairs(temp_dir)

        data = process_file_pairs(pairs, os.path.basename(zip_file))
        shutil.rmtree(temp_dir)
        all_data.extend(data)

    return pd.DataFrame(all_data)

def get_existing_zip_files(existing_df):
    """
    Get a set of ZIP files that have already been processed.

    Parameters:
        existing_df (pandas.DataFrame): DataFrame containing existing data.

    Returns:
        set: Set of ZIP file names.
    """
    if existing_df is not None and 'ZIP_File' in existing_df.columns:
        return set(existing_df['ZIP_File'])
    return set()

def process_file_pairs(pairs, zip_file_name):
    """
    Process pairs of JPEG and CR3 files.

    Parameters:
        pairs (list): List of tuples containing JPEG and CR3 file paths.
        zip_file_name (str): Name of the ZIP file being processed.

    Returns:
        list: List of dictionaries containing processed data.
    """
    data = []
    for jpeg_path, cr3_path in tqdm(pairs, leave=False):
        jpeg_shutter_speed = process_jpeg(jpeg_path)
        cr3_max_r, cr3_max_g, cr3_max_b, cr3_max_luminance = process_cr3(cr3_path)
        tqdm.write(f'ZIP File: {zip_file_name}, Shutter Speed: {jpeg_shutter_speed}, Max Pixel Luminance: {cr3_max_luminance}')

        if cr3_max_r < LIMIT_LUMINANCE and cr3_max_g < LIMIT_LUMINANCE and cr3_max_b < LIMIT_LUMINANCE and cr3_max_luminance < LIMIT_LUMINANCE:
            data.append({
                "Shutter_Speed": jpeg_shutter_speed,
                "Max_Pixel_Luminance": cr3_max_luminance / LIMIT_LUMINANCE,  # 正規化
                "ZIP_File": zip_file_name
            })
    return data

# ===============================
# Cache Handling Functions
# ===============================

def load_cached_dataframe(cache_csv_path):
    """
    Load cached data from a CSV file.

    Parameters:
        cache_csv_path (str): Path to the cache CSV file.

    Returns:
        pandas.DataFrame: Loaded DataFrame or an empty one if not found.
    """
    if cache_csv_path and os.path.isfile(cache_csv_path):
        return pd.read_csv(cache_csv_path)
    else:
        return pd.DataFrame(columns=CACHE_COLUMNS)

# ===============================
# Display Functions
# ===============================

def display_dataframe(df):
    """
    Display the DataFrame and its corresponding graph.

    Parameters:
        df (pandas.DataFrame): DataFrame to display.
    """
    if not df.empty:
        st.write("### Shutter Speed and Luminance Table")
        st.dataframe(df)
        display_graph(df)

def display_graph(df):
    """
    Display a graph of shutter speed vs. luminance for multiple ZIP files.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the data to plot.
    """
    fig = go.Figure()

    for zip_file in df['ZIP_File'].unique():
        subset = df[df['ZIP_File'] == zip_file]
        fig.add_trace(go.Scatter(
            x=subset['Shutter_Speed'],
            y=subset['Max_Pixel_Luminance'],
            mode='lines+markers',
            name=zip_file
        ))

    fig.update_layout(
        title='Shutter Speed vs. Luminance for Multiple ZIP Files',
        xaxis_title='Shutter Speed',
        yaxis_title='Normalized Luminance',
        legend_title='ZIP Files',
        template='plotly_white'
    )

    st.plotly_chart(fig)

# ===============================
# Scaling and Fitting Functions
# ===============================

def prepare_patterns(data):
    """
    Prepare patterns from the DataFrame.

    Parameters:
        data (pandas.DataFrame): DataFrame containing the data.

    Returns:
        dict: Dictionary of patterns keyed by ZIP file names.
    """
    patterns = {}
    for zip_file in data['ZIP_File'].unique():
        pattern_data = data[data['ZIP_File'] == zip_file]
        x = pattern_data['Shutter_Speed'].values
        y = pattern_data['Max_Pixel_Luminance'].values  # すでに正規化済み
        patterns[zip_file] = (x, y)
    return patterns

def find_reference_pattern_based_on_luminance_range(patterns):
    """
    自動的に輝度の最小値と最大値の差が最も大きいパターンを見つける。

    Parameters:
        patterns (dict): パターンの辞書。

    Returns:
        str: 最も適切な参照パターン名。
    """
    max_luminance_range = -np.inf
    selected_pattern = None

    for zip_file, (x, y) in patterns.items():
        luminance_range = np.max(y) - np.min(y)
        if luminance_range > max_luminance_range:
            max_luminance_range = luminance_range
            selected_pattern = zip_file

    return selected_pattern

def interpolate_data(y, x, overlap_y):
    """
    Interpolate data for distance calculation.

    Parameters:
        y (array): Original y-values.
        x (array): Original x-values.
        overlap_y (array): Overlapping y-values.

    Returns:
        array: Interpolated x-values corresponding to overlap_y.
    """
    return interp1d(y, x, kind='linear', bounds_error=False, fill_value="extrapolate")(overlap_y)

def calculate_distance(x1, y1, x2, y2):
    """
    Calculate the distance between two patterns.

    Parameters:
        x1, y1 (array): x and y values of the first pattern.
        x2, y2 (array): x and y values of the second pattern.

    Returns:
        float: The mean distance between the two patterns.
    """
    min_y, max_y = max(min(y1), min(y2)), min(max(y1), max(y2))
    if min_y > max_y:
        return np.inf  # Return infinite distance if no overlap
    overlap_y = np.linspace(min_y, max_y, num=max(len(y1), len(y2)))
    interp_x1 = interpolate_data(y1, x1, overlap_y)
    interp_x2 = interpolate_data(y2, x2, overlap_y)
    distances = np.abs(interp_x1 - interp_x2)
    return np.mean(distances)

def scale_patterns(patterns, alphas):
    """
    Scale patterns using the provided alphas.

    Parameters:
        patterns (dict): Original patterns.
        alphas (dict): Scaling factors for each pattern.

    Returns:
        dict: Scaled patterns.
    """
    scaled_patterns = {}
    for zip_file, (x, y) in patterns.items():
        alpha = alphas.get(zip_file, 1)
        scaled_patterns[zip_file] = (x * alpha, y)
    return scaled_patterns

def objective(alpha, x, y, ref_x, ref_y):
    """
    Objective function for optimization.

    Parameters:
        alpha (float): Scaling factor.
        x, y (array): x and y values of the pattern to scale.
        ref_x, ref_y (array): x and y values of the reference pattern.

    Returns:
        float: Distance between the scaled pattern and the reference pattern.
    """
    scaled_x = x * alpha
    return calculate_distance(scaled_x, y, ref_x, ref_y)

def find_optimal_alphas(patterns, ref_pattern_name):
    """
    Find optimal scaling factors (alphas) to align each pattern with the reference pattern.

    Parameters:
        patterns (dict): Dictionary of patterns.
        ref_pattern_name (str): Name of the reference pattern.

    Returns:
        dict: Optimal alphas for each pattern.
    """
    ref_x, ref_y = patterns[ref_pattern_name]
    optimal_alphas = {}
    for zip_file, (x, y) in patterns.items():
        result = minimize(objective, 1.0, args=(x, y, ref_x, ref_y),
                          bounds=[(sys.float_info.epsilon, sys.float_info.max)])
        optimal_alphas[zip_file] = result.x[0]
    return optimal_alphas

def adjust_alphas(optimal_alphas, light_source_pattern):
    """
    Adjust alphas so that the light source pattern has an alpha of 1.

    Parameters:
        optimal_alphas (dict): Optimal alphas for each pattern.
        light_source_pattern (str): Name of the light source pattern.

    Returns:
        dict: Adjusted alphas.
    """
    alpha_L = optimal_alphas[light_source_pattern]
    adjusted_alphas = {zip_file: alpha / alpha_L for zip_file, alpha in optimal_alphas.items()}
    return adjusted_alphas

def log_func(x, a, b):
    """
    Logarithmic function used for fitting.

    Parameters:
        x (array): Independent variable.
        a, b (float): Fitting parameters.

    Returns:
        array: Dependent variable.
    """
    return a * np.log(b * x + 1)

def fit_log_function(scaled_patterns):
    """
    Fit the logarithmic function to the scaled patterns.

    Parameters:
        scaled_patterns (dict): Scaled patterns.

    Returns:
        tuple: Fitted parameters a and b.
    """
    all_x = np.concatenate([x for x, _ in scaled_patterns.values()])
    all_y = np.concatenate([y for _, y in scaled_patterns.values()])
    return curve_fit(log_func, all_x, all_y, p0=INITIAL_PARAMS)[0]

def create_figure(scaled_patterns=None, fitted_params=None):
    """
    Create a plotly figure for the scaled patterns and fitted curve.

    Parameters:
        scaled_patterns (dict): Scaled patterns to plot.
        fitted_params (tuple): Fitted parameters a and b.

    Returns:
        plotly.graph_objects.Figure: The created figure.
    """
    fig = go.Figure()
    for zip_file, (x, y) in scaled_patterns.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=zip_file))

    if fitted_params is not None:
        log_x = np.linspace(1e-6, max(max(x) for x, _ in scaled_patterns.values()) * 1.2, 100)
        log_y = log_func(log_x, fitted_params[0], fitted_params[1])
        fig.add_trace(go.Scatter(x=log_x, y=log_y, mode='lines', name='Fitted Log Function', line=dict(dash='dash')))

    return fig

def f_M(x, a, b, alpha_M):
    """
    Function to calculate f_M(x) for a given pattern.

    Parameters:
        x (array): Independent variable.
        a, b (float): Fitting parameters.
        alpha_M (float): Scaling factor for the pattern.

    Returns:
        array: Calculated values of f_M(x).
    """
    return a * np.log(b * alpha_M * x + 1)

def f_C(y_C, a, b, c):
    """
    Correction function f_C(y_C).

    Parameters:
        y_C (array): Measured luminance values.
        a, b, c (float): Constants.

    Returns:
        array: Corrected luminance values.
    """
    return c / b * (np.exp(y_C / a) - 1)

# ===============================
# Main Function
# ===============================

def main():
    st.title('Process ZIP Files Containing JPEG and CR3 Images')

    if 'df' not in st.session_state:
        st.session_state.df = load_cached_dataframe(CACHE_CSV_FILE_PATH)
    else:
        display_dataframe(st.session_state.df)

    folder_path = st.sidebar.text_input("Enter the path to the folder containing ZIP files")
    if st.sidebar.button('Process ZIP Files and Update DataFrame'):
        if folder_path and os.path.isdir(folder_path):
            zip_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.zip')]
            st.session_state.df = process_zip_files(zip_files, st.session_state.df)
            st.session_state.df.to_csv(CACHE_CSV_FILE_PATH, index=False)

    # Prepare patterns from the DataFrame
    st.session_state.patterns = prepare_patterns(st.session_state.df)

    # 自動的に参照パターンを決定
    reference_pattern_name = find_reference_pattern_based_on_luminance_range(st.session_state.patterns)
    st.write(f"自動選択された参照パターン: {reference_pattern_name}")

    # 光源パターンは引き続き手動で選択
    light_source_pattern = st.selectbox('Select the light source pattern:', list(st.session_state.patterns.keys()))

    # Selection of patterns to plot
    selected_patterns = st.multiselect(
        'Select patterns to plot:',
        list(st.session_state.patterns.keys()),
        default=list(st.session_state.patterns.keys())
    )

    # Calculate optimal scaling factors (alpha) to align with reference pattern
    optimal_alphas = find_optimal_alphas(st.session_state.patterns, reference_pattern_name)

    # Recalculate alphas to set alpha of light source pattern to 1
    adjusted_alphas = adjust_alphas(optimal_alphas, light_source_pattern)

    # Scale patterns with recalculated alphas
    scaled_patterns = scale_patterns(st.session_state.patterns, adjusted_alphas)

    # Fit log function to scaled patterns
    fitted_params = fit_log_function(scaled_patterns)
    a_param, b_param = fitted_params  # Fitting parameters


    st.write('### After Scaling Shutter Speed (Aligned to Light Source Pattern)')

    st.latex(r'f_L(x) = a \cdot \log(b \cdot x + 1)')

    # Display fitted parameters
    st.write("#### Fitted Parameters:")
    latex_string_a = rf'''a = {a_param:.6f}'''
    st.latex(latex_string_a)
    if st.button("Copy a parameter"):
        pyperclip.copy(f"{a_param}")
    latex_string_b = rf'''b = {b_param:.6f}'''
    st.latex(latex_string_b)
    if st.button("Copy b parameter"):
        pyperclip.copy(f"{b_param}")

    # Plot scaled patterns and fitted curve
    scaled_patterns_to_plot = {k: v for k, v in scaled_patterns.items() if k in selected_patterns}
    fig = create_figure(scaled_patterns_to_plot, fitted_params)
    st.plotly_chart(fig)

    # Display adjusted alphas
    df_optimal_alphas = pd.DataFrame({
        'ZIP_File': list(adjusted_alphas.keys()),
        'Adjusted_Alpha': list(adjusted_alphas.values())
    })
    st.write(df_optimal_alphas)

    # Merge adjusted alphas into session dataframe
    df_alphas = df_optimal_alphas.copy()
    if 'Adjusted_Alpha' in st.session_state.df.columns:
        st.session_state.df = st.session_state.df.drop(columns=['Adjusted_Alpha'])
    st.session_state.df = st.session_state.df.merge(df_alphas, on='ZIP_File', how='left')
    st.session_state.df.to_csv(CACHE_CSV_FILE_PATH, index=False)

    st.write('### Predicted Lines for Selected Patterns')

    # Input X range slider
    max_x_value = max(max(x) for x, _ in st.session_state.patterns.values())
    input_x_max = st.slider('Input X Range (s)', min_value=0.001, max_value=2.0, value=float(max_x_value), step=0.001)

    # Input L_e
    L_e = st.slider('Luminance at input_x_max (L_e)', min_value=0.0, max_value=100.0, value=5.0, step=0.5)

    # Calculate c: c is L_e / (shutter speed)
    c = L_e / input_x_max

    # Create plot
    fig = go.Figure()

    # Plot original data points
    for zip_file, (x, y) in st.session_state.patterns.items():
        if zip_file in selected_patterns:
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'Original {zip_file}'))

    # Compute fi_x
    fi_x = np.linspace(0, input_x_max, 100)

    # Plot f_L(x)
    alpha_L = 1.0  # Alpha of light source pattern is 1
    f_L_x = f_M(fi_x, a_param, b_param, alpha_L)
    fig.add_trace(go.Scatter(x=fi_x, y=f_L_x, mode='lines', name='f_L(x)', line=dict(color='purple')))

    # Plot f_M(x) for each medium pattern
    for zip_file in selected_patterns:
        alpha_M = adjusted_alphas[zip_file]
        if zip_file != light_source_pattern:
            f_M_x = f_M(fi_x, a_param, b_param, alpha_M)
            fig.add_trace(go.Scatter(x=fi_x, y=f_M_x, mode='lines', name=f'f_M(x) for {zip_file}', line=dict(dash='dashdot')))

    # Plot f_I_L(x) = c * x
    f_IL_x = c * fi_x
    fig.add_trace(go.Scatter(x=fi_x, y=f_IL_x, mode='lines', name='f_I_L(x) = c * x', line=dict(color='purple', dash='dot')))

    # Plot f_I_M(x) = c * alpha_M * x for each medium pattern
    for zip_file in selected_patterns:
        alpha_M = adjusted_alphas[zip_file]
        f_IM_x = c * alpha_M * fi_x
        fig.add_trace(go.Scatter(x=fi_x, y=f_IM_x, mode='lines', name=f'f_I_M(x) for {zip_file}', line=dict(dash='dash')))

    # Plot corrected data points using f_C
    for zip_file, (x, y) in st.session_state.patterns.items():
        if zip_file in selected_patterns:
            f_C_y = f_C(y, a_param, b_param, c)
            fig.add_trace(go.Scatter(x=x, y=f_C_y, mode='markers', name=f'Corrected Data for {zip_file}', marker=dict(size=5)))

    # Update figure layout
    fig.update_layout(
        title='Final Graph Aligned to Light Source Pattern',
        xaxis_title='Shutter Speed',
        yaxis_title='Luminance',
        legend_title='Patterns',
        template='plotly_white',
        xaxis_range=[0, input_x_max],
    )

    st.plotly_chart(fig)

    # Prepare DataFrame for luminance values at input_x_max
    COLUMN_ZIP_FILE = 'ZIP_File'
    COLUMN_LUMINANCE = f'Luminance at {input_x_max}'

    data_to_append = []

    for zip_file in selected_patterns:
        optimal_alpha = optimal_alphas.get(zip_file, 1)
        scaled_b = optimal_alpha * fitted_params[1]
        log_y = log_func(input_x_max, fitted_params[0], scaled_b)
        data_to_append.append({
            COLUMN_ZIP_FILE: zip_file,
            COLUMN_LUMINANCE: log_y,
        })

    df_luminance = pd.DataFrame(data_to_append)

    # Remove duplicate column before merging
    if COLUMN_LUMINANCE in st.session_state.df.columns:
        st.session_state.df = st.session_state.df.drop(columns=[COLUMN_LUMINANCE])

    # Merge luminance data into session dataframe
    st.session_state.df = st.session_state.df.merge(df_luminance, on='ZIP_File', how='left')

    # Save light source pattern and reference pattern into session dataframe
    st.session_state.df['Light_Source_Pattern'] = light_source_pattern
    st.session_state.df['Reference_Pattern'] = reference_pattern_name
    st.session_state.df.to_csv(CACHE_CSV_FILE_PATH, index=False)

if __name__ == "__main__":
    main()
