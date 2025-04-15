from matplotlib.axes import Axes
from common import OrderedImage, FloatArray, Uint8Array, ImageArray
from collections.abc import Sequence
import cv2
import math
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Callable

def plot_channels(img: OrderedImage) -> None:
    split = [channel.astype(np.float32) for channel in cv2.split(img.cl)]
    plot_imgs([img, *split], titles=('color', 'b', 'g', 'r'))

def plot_imgs(imgs: list[OrderedImage | ImageArray], grouped: bool = True,
              max_cols: int = 2, titles: Sequence[str] | None = None) -> None:
    if titles is None:
        titles = [str(x) for x in range(len(imgs))]

    if grouped:
        num_rows = math.ceil(len(imgs) / max_cols)
        fig, axs = plt.subplots(nrows=num_rows, ncols=max_cols, sharex=True, sharey=True)
        for i, ax in enumerate(axs.flat):
            if i < len(imgs):
                plot_img(imgs[i], get_axes=lambda: ax, show_plot=False)
                ax.title.set_text(titles[i])
        fig.tight_layout()
        plt.show()
    else:
        for img in imgs:
            plot_img(img)

def plot_img(img: OrderedImage | FloatArray | Uint8Array, show_plot: bool = True, get_axes: Callable[[], Axes] = lambda: plt.gca()) -> None:
    # TODO: mypy is having a seizure over these types
    if type(img) is OrderedImage:
        plot_ordered_img(img=img, show_plot=show_plot, get_axes=get_axes)
    else:
        plot_img_array(img=img, show_plot=show_plot, get_axes=get_axes)

def plot_ordered_img(img: OrderedImage, show_plot: bool = True, get_axes = lambda: plt.gca()) -> None:
    plot_img_array(img=img.cl, show_plot=show_plot, get_axes=get_axes)

def plot_img_array(img: ImageArray, show_plot: bool = True, get_axes = lambda: plt.gca()) -> None:
    if len(img.shape) == 3:
        get_axes().imshow(img[:,:,::-1])  # Reverse BGR -> RGB
    elif type(img) is Uint8Array:
        get_axes().imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        get_axes().imshow(img, cmap='gray', vmin=0.0, vmax=1.0)

    if show_plot:
        plt.show()