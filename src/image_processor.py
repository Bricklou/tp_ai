import os
from datetime import datetime

from PIL import Image


class ImageProcessor:
    def __init__(self, input_folder: str):
        self.input_folder = input_folder

    def process_folder(self) -> None:
        """
        Process all images from the input folder specified in the constructor.
        :return:
        """

        files = os.listdir(self.input_folder)

        files = list(filter(lambda f: f.endswith(".jpg"), files))
        print(files)
        print("Found %d files" % len(files))

        target_size = 640
        output_dir = "dataset/%s" % ImageProcessor._generate_dir_name()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file in files:
            self._process_image(
                os.path.join(self.input_folder, file), output_dir, target_size
            )

    @staticmethod
    def _process_image(input_path: str, output_dir: str, target_size: int) -> None:
        """
        Process image based on the following steps:
        - resize images to fit inside the target size
        - fill the image with padding
        - save the image to disk

        :param input_path: path to the input image
        :param output_dir: output directory path for the image
        :param target_size: target size of the image
        :return:
        """

        with Image.open(input_path) as img:
            img = img.convert("RGB")
            ImageProcessor._resize_image(img, target_size)
            img = ImageProcessor._add_padding(
                img, (target_size, target_size), (0, 0, 0)
            )

            filename = os.path.basename(input_path)
            img.save(os.path.join(output_dir, filename), "JPEG")

    @staticmethod
    def _resize_image(img: Image.Image, target_size: int) -> None:
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    @staticmethod
    def _add_padding(
        img: Image.Image, target_size: tuple[int, int], color: tuple[int, int, int]
    ) -> Image.Image:
        """
        Add margin to an image to fit the target size

        :param img: image to add margin to
        :param target_size: size that the margin need to fit into
        :param color: color to fill the new image with
        :return: the new image with padding
        """

        new_image = Image.new(img.mode, target_size, color)
        new_image.paste(img, (0, 0))

        return new_image

    @staticmethod
    def _generate_dir_name() -> str:
        return datetime.now().strftime("%Y%m%d%H%M%S")
