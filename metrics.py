import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio

from common import OrderedImage, FloatArray, Uint8Array, Stats
import config
from main import load_img

def uciqe(img: OrderedImage[FloatArray]) -> float:
    lab_img = OrderedImage(cv2.cvtColor(img.cl, cv2.COLOR_BGR2Lab).astype(np.float32))

    luminances = lab_img.cf[0]
    a = lab_img.cf[1]
    b = lab_img.cf[1]

    chromas = np.sqrt(a ** 2 + b ** 2)
    chroma_std_dev = np.std(chromas)

    luminance_contrast = np.percentile(luminances, 99) - np.percentile(luminances, 1)

    saturations = chromas / (luminances + 0.01)  # chroma / (luminance + safety)
    sat_mean = np.mean(saturations)

    # UCIQE = c1​ * sigma_c​ + c2 * con_l​ + c3​ * mu_s
    return config.Metrics.UCIQE.C1 * np.sqrt(chroma_std_dev) + config.Metrics.UCIQE.C2 * luminance_contrast + config.Metrics.UCIQE.C3 * sat_mean

def ssim(img1: OrderedImage, img2: OrderedImage) -> float:
    return structural_similarity(img1.cf, img2.cf, channel_axis=0, data_range=1.0)

def psnr(gnd_truth: OrderedImage, test_img: OrderedImage) -> float:
    return peak_signal_noise_ratio(gnd_truth.cl, test_img.cl, data_range=1.0)

def mse(gnd_truth: OrderedImage, test_img: OrderedImage) -> float:
    return mean_squared_error(gnd_truth.cl, test_img.cl)

if __name__ == '__main__':
    print(uciqe(load_img(Path('cup') / '0.png')))
    print(uciqe(load_img(Path('milk') / '2.jpg')))