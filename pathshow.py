from zipfile import ZipFile
import os

# Function to list the directory structure with a diagram-like format
def list_directory_structure_diagram(dir_path):
    for root, dirs, files in os.walk(dir_path):
        # Calculate the indentation level. More indentation means deeper in the directory structure
        level = root.replace(dir_path, '').count(os.sep)
        indent = ' ' * 4 * (level - 1)
        print(f"{indent}{'|-- ' if level > 0 else ''}{os.path.basename(root)}/")
        subindent = ' ' * 4 * level
        for f in sorted(files):
            print(f"{subindent}|-- {f}")

# Example usage with a dummy path. Replace these paths with your actual paths or variables.
zip_path = r'C:/Users/nextCode24\Desktop/FlaskOpencv_FaceRecognition.zip'
extract_dir = r'C:/Users/nextCode24/Desktop/data/FlaskOpencv_FaceRecognition'

# Extracting the ZIP file (ensure this is done before listing the directory structure)
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Listing the directory structure of the extracted contents in a diagram-like format
list_directory_structure_diagram(extract_dir)
