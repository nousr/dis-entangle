"""Predict the mask for an image."""

import click
from PIL import Image
from dis_entangle.helpers import build_model, predict, load_image


@click.command()
@click.option("--image_path", default="images/baby_duck.jpg", help="Path to the image to be masked.")
def main(image_path):
    """Main function."""
    model = build_model()

    # load an image from the path
    image_tensor, original_size = load_image(image_path)

    # predict the mask
    mask = predict(model, image_tensor, original_size)

    # save the mask
    mask = Image.fromarray(mask)

    mask.save("mask.png")


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
