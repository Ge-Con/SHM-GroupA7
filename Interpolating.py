#importing modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Function to find the smallest array size
def find_smallest_array_size(array_list):
    min_size = float('inf')  # Initialize to a very large number

    for arr in array_list:  # Loop through all arrays
        if isinstance(arr, np.ndarray):  # Check if it's a numpy array
            size = arr.size  # Get its size
            if size < min_size:  # Check if the size is smaller than the previously determined min size
                min_size = size  # Update min_size

    return min_size  # Return the smallest size

def scale_HIs(HI_arr):
    minimum = find_smallest_array_size(HI_arr)  # Find the smallest array size
    for i in range(len(HI_arr)):  # Loop through each health indicator array
        if HI_arr[i].size > minimum:  # If the size is greater than the minimum
            arr_interp = interp1d(np.arange(HI_arr[i].size), HI_arr[i])  # Create an interpolation function
            arr_compress = arr_interp(np.linspace(0, HI_arr[i].size - 1, minimum))  # Compress to the minimum size
            HI_arr[i] = arr_compress  # Update the array
    HI_arr = np.vstack(HI_arr)  # Stack the arrays
    return HI_arr

def scale_exact(HI_list, minimum=30):
    if HI_list.size > minimum:  # If the size is greater than the minimum
        arr_interp = interp1d(np.arange(HI_list.size), HI_list)  # Create an interpolation function
        arr_compress = arr_interp(np.linspace(0, HI_list.size - 1, minimum))  # Compress to the minimum size
    else:
        arr_compress = HI_list
    return np.array(arr_compress)


'''''
#WITH Z_ARRAY, NOT NEEDED WIH GENERAL USAGE

# Function to scale arrays to the smallest size
def scale_HIs(HI_arr, z_arr):
    minimum = find_smallest_array_size(HI_arr + [z_arr])  # Find the smallest array size
    for i in range(len(HI_arr)):  # Loop through each health indicator array
        if HI_arr[i].size > minimum:  # If the size is greater than the minimum
            arr_interp = interp1d(np.arange(HI_arr[i].size), HI_arr[i])  # Create an interpolation function
            arr_compress = arr_interp(np.linspace(0, HI_arr[i].size - 1, minimum))  # Compress to the minimum size
            HI_arr[i] = arr_compress  # Update the array
    HI_arr = np.vstack(HI_arr)  # Stack the arrays
    if z_arr.size != minimum:  # Check if the z_array size is not the same as the minimum
        arr_interp = interp1d(np.arange(z_arr.size), z_arr)  # Create an interpolation function
        arr_compress = arr_interp(np.linspace(0, z_arr.size - 1, minimum))  # Compress to the smallest size
        z_arr = arr_compress
    full = np.append(HI_arr, [z_arr], axis=0)  # Append the z_arr into the vstacked HI array
    return full, HI_arr, z_arr
    
'''''

'''''
#------------------------Testing wth random shit----------------------
# Number of lines
num_lines = 50

# Generate random x-coordinates, start, and end points for the lines
np.random.seed(0)  # For reproducibility
x_positions = np.linspace(1, 10, num_lines)  # x-coordinates for the vertical lines
starts = np.random.uniform(0, 5, num_lines)  # Random y-coordinates where the lines start
ends = np.random.uniform(5, 10, num_lines)   # Random y-coordinates where the lines end

# Generate the vertical lines' y-values with varying lengths
y_values_list = []
for y_start, y_end in zip(starts, ends):
    num_points = np.random.randint(50, 150)  # Varying number of points per line
    y_values = np.linspace(y_start, y_end, num_points)
    y_values_list.append(y_values)

# Example z_arr for demonstration, could be another line or different data
z_arr = np.random.uniform(0, 10, 120)  # Random example array with a different size

# Print the lengths of a specific line before scaling
print("Original length of line 10:", len(y_values_list[10]))

# Plot the original vertical lines
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i, x in enumerate(x_positions):
    plt.plot(np.full(y_values_list[i].shape, x), y_values_list[i])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Original Vertical Lines')
plt.grid(True)

# Apply the scaling function
full, HI_arr, z_arr = scale_HIs(y_values_list, z_arr)
print('Smallest array size',find_smallest_array_size(y_values_list))
# Print the lengths of the same line after scaling
print("Scaled length of line 10:", len(HI_arr[10]))

# Plot the scaled vertical lines
plt.subplot(1, 2, 2)
for i, x in enumerate(x_positions):
    plt.plot(np.full(HI_arr[i].shape, x), HI_arr[i])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scaled Vertical Lines')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()

#code to stretch all HI to the largets one(if we change our mind)
'''''

'''''
def find_largest_array_size(array_list):
    max_size = 0

    for arr in array_list:
        if isinstance(arr, np.ndarray):
            size = arr.size
            if size > max_size:
                max_size = size

    return max_size

def scale_HIs(HI_arr, z_arr):
    """ Takes list of health indicator arrays for the test data and train data and scales them up to match the one with the greatest length, outputs a numpy array containing the health indicator of all panels """
    maximum = find_largest_array_size(HI_arr + [z_arr])
    for i in range(len(HI_arr)):
        if HI_arr[i].size < maximum:
            arr_interp = interp.interp1d(np.arange(HI_arr[i].size), HI_arr[i])
            arr_stretch = arr_interp(np.linspace(0, HI_arr[i].size - 1, maximum))
            HI_arr[i] = arr_stretch
    HI_arr = np.vstack(HI_arr)
    if z_arr.size != HI_arr.shape[1]:
        arr_interp = interp.interp1d(np.arange(z_arr.size), z_arr)
        arr_stretch = arr_interp(np.linspace(0, z_arr.size - 1, HI_arr.shape[1]))
        z_arr = arr_stretch
    full = np.append(HI_arr, z_arr, axis=0)
    return full, HI_arr, z_arr

'''''
