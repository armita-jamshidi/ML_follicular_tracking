"""Remove what is not anatomy: on-screen text, rulers, scanner UI and borders.

The ultrasound image itself is a bright, roughly fan-shaped region. Text labels, depth
rulers and logos are thin or small, so a morphological opening removes them and the
largest remaining bright component is taken to be the fan.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def fan_mask(gray: np.ndarray, threshold: int = 20, open_size: int = 9) -> np.ndarray:
    """Boolean mask of the largest bright region after removing thin structures."""
    bright = gray > threshold
    opened = ndimage.binary_opening(bright, structure=np.ones((open_size, open_size)))
    labels, n = ndimage.label(opened)
    if n == 0:
        return np.ones_like(bright)
    sizes = ndimage.sum(opened, labels, index=range(1, n + 1))
    largest = labels == (int(np.argmax(sizes)) + 1)
    # Close small dark gaps (e.g. anechoic follicles touching the edge) and fill holes,
    # so follicles inside the fan are never masked out.
    largest = ndimage.binary_closing(largest, structure=np.ones((15, 15)))
    largest = ndimage.binary_fill_holes(largest)
    # The fan is convex along each row, so fill between the outermost pixels of each row.
    # This keeps dark tissue at the fan's edge that the threshold dropped.
    filled = np.zeros_like(largest)
    for y in np.nonzero(largest.any(axis=1))[0]:
        xs = np.nonzero(largest[y])[0]
        filled[y, xs[0] : xs[-1] + 1] = True
    return filled


def crop_to_fan(img: Image.Image, margin: int = 4, mask_outside: bool = True) -> Image.Image:
    """Crop a grayscale image to the ultrasound fan and blank everything outside it."""
    gray = np.asarray(img.convert("L"))
    mask = fan_mask(gray)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return img.convert("L")
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, gray.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, gray.shape[1])
    out = gray.copy()
    if mask_outside:
        keep = ndimage.binary_dilation(mask, iterations=margin)
        out[~keep] = 0
    return Image.fromarray(out[y0:y1, x0:x1])


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Resize keeping aspect ratio and pad with black to ``size x size``.

    Stretching would distort images from different sources by different amounts,
    which is itself a shortcut a model could learn.
    """
    img = img.copy()
    img.thumbnail((size, size), Image.BILINEAR)
    canvas = Image.new(img.mode, (size, size), 0)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas
