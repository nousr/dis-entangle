"""Predict the mask for an image."""

import os
from glob import glob

import click
from tqdm import tqdm
from dis_entangle import build_model, load_image, predict
from PIL import Image


IMG_EXTENSIONS = ["jpg", "png", "gif", "jpeg"]

@click.command()
@click.option("--image_folder", default="images", help="input path to load images from.")
@click.option("--mask_folder", default="masks", help="output path to save the masks in.")
def main(image_folder: str, mask_folder: str):
    """Main function."""
    model = build_model()

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    image_files =[]

    for ext in IMG_EXTENSIONS:
        image_files.extend(glob(os.path.join(image_folder, f"*.{ext}")))

    for image in tqdm(image_files):
        image_tensor, original_size = load_image(image)
        mask = predict(model, image_tensor, original_size)
        mask = Image.fromarray(mask)
        save_path = os.path.join(mask_folder, os.path.basename(image))
        print(f"Saving mask to {save_path}")
        mask.save(save_path)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
