import mitsuba as mi
import json
import sys
import argparse
import numpy as np
import drjit as dr

class MitsubaJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle ScalarVector2u and other drjit vector types
        if hasattr(obj, '__class__') and ('ScalarVector' in obj.__class__.__name__ or 
                                        'Vector' in obj.__class__.__name__):
            return list(obj)
        
        # Handle Transform4f
        if hasattr(obj, '__class__') and 'Transform' in obj.__class__.__name__:
            try:
                # Extract the matrix data from the transform
                matrix = obj.matrix
                return matrix.numpy().tolist()
            except:
                # Fallback: try to convert to list directly
                return list(obj)
        
        # Handle drjit arrays
        if dr.is_array_v(obj):
            try:
                return dr.detach(obj).numpy().tolist()
            except:
                return list(obj)
        
        # Handle numpy arrays and matrices
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # Handle other array-like objects
        try:
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, dict)):
                return list(obj)
        except:
            pass
            
        # Handle nested structures
        if isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
            
        # For all other types, try to convert to a basic Python type
        try:
            # Handle float types
            if hasattr(obj, 'item'):
                return obj.item()
            # Handle other numeric types
            elif hasattr(obj, '__float__'):
                return float(obj)
            elif hasattr(obj, '__int__'):
                return int(obj)
        except:
            pass
            
        return super().default(obj)

def serialize_mitsuba_scene(params_dict):
    """
    Serialize Mitsuba scene parameters to JSON with proper type handling.
    
    Args:
        params_dict (dict): Dictionary containing Mitsuba scene parameters
        
    Returns:
        str: JSON string representation of the scene parameters
    
    Example:
        scene = mi.load_file('scene.xml')
        params = mi.traverse(scene)
        params_dict = {a[0]: a[1] for a in params}
        json_output = serialize_mitsuba_scene(params_dict)
    """
    return json.dumps(params_dict, cls=MitsubaJSONEncoder)

def deserialize_mitsuba_scene(json_str):
    """
    Deserialize JSON string back into a Python dictionary.
    Note: This converts the data back into basic Python types,
    not into Mitsuba-specific types.
    
    Args:
        json_str (str): JSON string of scene parameters
        
    Returns:
        dict: Dictionary containing the deserialized scene parameters
    """
    return json.loads(json_str)



def main():
    parser = argparse.ArgumentParser(description='Get parameters from a Mitsuba scene.')
    parser.add_argument('--scene', type=str, required=True, help='Path to the Mitsuba scene XML file.')
    args = parser.parse_args()

    mi.set_variant('cuda_ad_rgb')

    try:
        scene = mi.load_file(args.scene)
        params = mi.traverse(scene)
        # Convert parameters to JSON-serializable format
        params_dict = {}

        for param in params:
            params_dict[param[0]] = param[1]

        print(serialize_mitsuba_scene(params_dict))
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        exit(1)

if __name__ == '__main__':
    main()
