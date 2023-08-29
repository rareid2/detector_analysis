

# first a quick function to save the array to a file
def save_array_to_file(array, file_path):
    with open(file_path, 'w') as file:
        for item in array:
            file.write(str(item) + '\n')

def read_array_from_file(file_path):
    with open(file_path, 'r') as file:
        array = []
        for line in file:
            line = line.strip()  # Remove any leading or trailing whitespace
            if line:  # Ignore empty lines
                array.append(float(line))  # Assuming the array elements are floats

        return array

def save_2d_array_to_file(array, file_path):
    with open(file_path, 'w') as file:
        for row in array:
            line = '\t'.join(str(item) for item in row)  # Join row elements with tab separator
            file.write(line + '\n')

def read_2d_array_from_file(file_path):
    with open(file_path, 'r') as file:
        array = []
        for line in file:
            line = line.strip()  # Remove any leading or trailing whitespace
            if line:  # Ignore empty lines
                row = [float(item) for item in line.split('\t')]  # Assuming tab-separated values
                array.append(row)

        return array