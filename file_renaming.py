import os

def rename_images(directory_path):
    # Change the current working directory to the specified directory
    os.chdir(directory_path)

    # Get a list of all files in the directory
    files = os.listdir()

    # Filter only image files (you can customize this based on your image file extensions)
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    # Rename each image file with the desired format ('galvanic_1', 'galvanic_2', ...)
    for index, image_file in enumerate(image_files, start=1):
        new_name = f'noCorrosion_{index}{os.path.splitext(image_file)[1]}'
        os.rename(image_file, new_name)
        print(f'Renamed: {image_file} -> {new_name}')

if __name__ == "__main__":
    # Specify the directory path where your images are located
    directory_path = 'dataset/non-corroded'

    # Call the function to rename images
    rename_images(directory_path)
