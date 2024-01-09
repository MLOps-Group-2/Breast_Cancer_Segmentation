import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def clip_image_by_order(image_path, label_path, image_clip_dir, label_clip_dir, clip_size):
    """
    Clip the image and corresponding label into smaller patches of specified size.

    :param image_path: Path to the original large image.
    :param label_path: Path to the original large label image (mask).
    :param image_clip_dir: Directory to store clipped images.
    :param label_clip_dir: Directory to store clipped labels.
    :param clip_size: Size of the square patch (e.g., 512x512 pixels).
    """
    # Read the original image and label
    image_array = np.array(Image.open(image_path))
    label_array = np.array(Image.open(label_path))
    H, W = image_array.shape[:2]  # Height and Width of the image

    # Base name of the image file
    image_name = os.path.basename(image_path).split(".")[0]

    # Loop through the image and clip into patches
    for curr_H in range(0, H, clip_size):
        for curr_W in range(0, W, clip_size):
            # Skip the patch if it doesn't fit the desired size
            if curr_H + clip_size > H or curr_W + clip_size > W:
                continue

            # Construct the file paths for the clipped image and label
            clip_image_path = os.path.join(image_clip_dir, f"{image_name}_{curr_H}_{curr_W}_size{clip_size}.png")
            clip_label_path = os.path.join(label_clip_dir, f"{image_name}_{curr_H}_{curr_W}_size{clip_size}.png")

            # Clip the image and label
            out_img_array = image_array[curr_H:curr_H + clip_size, curr_W:curr_W + clip_size, :]
            out_label_array = label_array[curr_H:curr_H + clip_size, curr_W:curr_W + clip_size]

            # Save the clipped image and label
            out_img = Image.fromarray(out_img_array)
            out_img.save(clip_image_path)
            out_label = Image.fromarray(out_label_array)
            out_label.save(clip_label_path)


if __name__ == '__main__':
    # Directories containing the original images and labels
    image_dir = '0_Public-data-Amgad2019_0.25MPP/rgbs_colorNormalized'
    label_dir = '0_Public-data-Amgad2019_0.25MPP/masks'

    # Directories for storing clipped images and labels for training and validation
    train_dir = 'BCSS/train'
    train_mask_dir = 'BCSS/train_mask'
    val_dir = 'BCSS/val'
    val_mask_dir = 'BCSS/val_mask'

    # Create directories if they don't exist
    for dir_path in [train_dir, train_mask_dir, val_dir, val_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # List of image names in the image directory
    image_names = os.listdir(image_dir)

    # Institutes for validation set
    val_institutes = ['OL', 'LL', 'E2', 'EW', 'GM', 'S3']

    # Process each image
    for image_name in tqdm(image_names):
        institute = image_name[5:7]
        image_file = os.path.join(image_dir, image_name)
        label_file = os.path.join(label_dir, image_name)

        # Distribute images to training or validation set based on institute
        if institute in val_institutes:
            clip_image_by_order(image_file, label_file, val_dir, val_mask_dir, 512)
        else:
            clip_image_by_order(image_file, label_file, train_dir, train_mask_dir, 512)
