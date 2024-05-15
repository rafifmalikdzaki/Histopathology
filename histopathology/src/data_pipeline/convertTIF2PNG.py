import os
import sys
from PIL import Image
from tqdm.auto import tqdm

def convert_images(tiff_folder: str, png_folder: str) -> None:
    """
    Convert JPG images to PNG format.

    Args:
        tiff_folder (str): The path of the folder containing tiff files.
        png_folder (str): The path of the folder to save PNG files.

    Returns:
        None
    """

    # Check if the PNG folder exists, create it if it doesn't
    if not os.path.exists(png_folder):
        os.mkdir(png_folder)

    # Get a list of files in the tiff folder
    pictures_to_convert = os.listdir(tiff_folder)

    # Iterate over each picture in the tiff folder
    for picture in tqdm(pictures_to_convert):
        # Replace the file extension with ".png"
        png_ext = picture.replace(".tiff", ".png")

        try:
            # Open the tiff image using PIL
            image = Image.open(os.path.join(tiff_folder, picture))

            # Save the image in PNG format to the PNG folder
            image.save(os.path.join(png_folder, png_ext))
        except IOError as e:
            # Handle any IO errors that occur during conversion
            print("Error converting {}: {}".format(picture, e))


if __name__ == "__main__":
    # Check if the correct number of command line arguments are provided
    if len(sys.argv) != 3:
        # Print the correct usage of the script
        print("Usage: python image_conversion.py [jpg_folder] [png_folder]")
    else:
        # Get the command line arguments for the JPG and PNG folders
        tiff_folder = sys.argv[1]
        png_folder = sys.argv[2]

        # Call the convert_images function with the provided folders
        convert_images(tiff_folder, png_folder)