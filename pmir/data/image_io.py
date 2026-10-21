import numpy as np
from PIL import Image


def load_image_rgb(path, size):
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0
