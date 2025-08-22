import numpy as np

def process_line(line):
    """
    Process a single line of data containing 4 arrays.
    Returns the four arrays as separate numpy arrays.
    """
    try:
        # Remove outer brackets and split by ']'
        arrays_str = line.strip()[1:-1].split(']')
        
        # Process each array string
        processed_arrays = []
        for arr_str in arrays_str:
            # Clean the string and convert to array
            # Remove leading '[' and split by spaces
            clean_str = arr_str.strip().strip('[').strip()
            if clean_str:  # Check if there's data to process
                numbers = np.array([float(x) for x in clean_str.split() if x])
                processed_arrays.append(numbers)
        
        return processed_arrays
    except Exception as e:
        print(f"Error processing line: {e}")
        return None

def load_data_file(file_path):
    """
    Load data from file line by line and process each line.
    Returns a list of processed lines, where each line contains the four arrays.
    """
    all_data = []
    
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file, 1):
                if line.strip():  # Skip empty lines
                    result = process_line(line)
                    if result is not None:
                        all_data.append(result)
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return all_data