import numpy as np
import cv2
import argparse

def load_csv(csv_path):
    """Load the CSV file and reshape it into an array of RGB values."""
    image = np.loadtxt(csv_path, delimiter=',')
    return image.reshape((-1, 3))

def create_image(image_data, width, height):
    """Create an empty image and fill it with the pixel values from the CSV data."""
    new_image = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            new_image[i, j] = image_data[i * width + j]
    return new_image

def convert_to_uint8(image):
    """Convert the image to uint8 format with values between 0 and 255."""
    return (image * 255).astype(np.uint8)

def save_image(image, output_path):
    """Save the image to the specified output path."""
    cv2.imwrite(output_path, image)
    print(f'Image saved as {output_path}')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert a CSV file to an image.',
        usage='python script.py <csv_path> [--width WIDTH] [--height HEIGHT]'
    )
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--width', type=int, default=256, help='Width of the image (default: 256)')
    parser.add_argument('--height', type=int, default=256, help='Height of the image (default: 256)')
    
    args = parser.parse_args()

    # Load and process the image data
    csv_path = args.csv_path
    image_data = load_csv(csv_path)
    new_image = create_image(image_data, args.width, args.height)
    new_image_uint8 = convert_to_uint8(new_image)

    # Save the image
    # remove the extension from the csv_path and replace it with .png
    # EX: ./data/true_image_gray.csv -> ./data/true_image_gray.png
    image_path = csv_path[:csv_path.rfind('.')] + '.png'
    save_image(new_image_uint8, image_path)

if __name__ == "__main__":
    main()
