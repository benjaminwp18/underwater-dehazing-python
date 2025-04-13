from dataclasses import dataclass

import cv2
import numpy as np

from common import OrderedImage, FloatArray, Uint8Array, float_to_uint8_array, uint8_to_float_array

def basic_CLAHE_float(img: OrderedImage[FloatArray], clip_limit: float = 0.02 * 256, tile_size: int = 128) -> OrderedImage[FloatArray]:
    unpadded_h, unpadded_w = img.cf[0].shape
    num_tiles_x = unpadded_w // tile_size + 1
    num_tiles_y = unpadded_h // tile_size + 1

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(num_tiles_y, num_tiles_x))
    u8_img = img.to_uint8s()
    for c in (0, 1, 2):
        u8_img.cf[c] = clahe.apply(u8_img.cf[c])
    return u8_img.to_floats()

def basic_CLAHE_u8_cl(img: OrderedImage[FloatArray], clip_limit: float = 0.02 * 256, tile_size: int = 128) -> OrderedImage[Uint8Array]:
    unpadded_h, unpadded_w = img.cf[0].shape
    num_tiles_x = unpadded_w // tile_size + 1
    num_tiles_y = unpadded_h // tile_size + 1

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(num_tiles_y, num_tiles_x))
    u8_img = img.to_uint8s()
    for c in (0, 1, 2):
        u8_img.cf[c] = clahe.apply(u8_img.cf[c])
    return u8_img

def clip_histogram(hist: FloatArray, clip_limit: int) -> FloatArray:
    # hist contains bin counts expressed as floats
    excess = np.maximum(hist - clip_limit, 0)
    clipped_hist = np.minimum(hist, clip_limit)
    total_excess: float = excess.sum()

    bin_incr = total_excess // hist.size
    remainder: int = int(total_excess) % hist.size
    clipped_hist += bin_incr
    if remainder > 0:
        clipped_hist[:remainder] += 1

    return clipped_hist

def compute_tile_lut(tile: Uint8Array, clip_limit: int) -> Uint8Array:
    hist = cv2.calcHist([tile], channels=[0], mask=None, histSize=[256], ranges=[0, 256]).flatten().astype(np.float32)
    clipped_hist = clip_histogram(hist, clip_limit)
    cdf = clipped_hist.cumsum().astype(np.float32)
    cdf_normalized = np.asarray(cdf * 255.0 / cdf[-1]).astype(np.uint8)
    return cdf_normalized

def adaptive_clahe(img: OrderedImage[FloatArray], clip_limits: np.ndarray | float = 40.0, tile_size: int = 64, interpolate: bool = True) -> OrderedImage[FloatArray]:
    channels: list[FloatArray] = []
    for channel in img.cf:
        channels.append(adaptive_clahe_channel(channel, clip_limits=clip_limits, tile_size=tile_size, interpolate=interpolate))

    return OrderedImage(cv2.merge(channels).astype(np.float32))

def adaptive_clahe_channel(float_channel: FloatArray, clip_limits: FloatArray | float = 40.0, tile_size: int = 64, interpolate: bool = True) -> FloatArray:
    channel = float_to_uint8_array(float_channel)

    unpadded_h, unpadded_w = channel.shape

    num_tiles_x = unpadded_w // tile_size + 1
    padded_w = num_tiles_x * tile_size
    num_tiles_y = unpadded_h // tile_size + 1
    padded_h = num_tiles_y * tile_size

    padding_w = padded_w - unpadded_w
    padding_h = padded_h - unpadded_h

    channel = cv2.copyMakeBorder(channel, 0, padding_h, 0, padding_w, cv2.BORDER_DEFAULT).astype(np.uint8)

    h, w = channel.shape

    # Handle clip_limits as array or scalar
    if np.isscalar(clip_limits):
        clip_limits = np.full((num_tiles_y, num_tiles_x), clip_limits)
    else:
        clip_limits = np.array(clip_limits)
        assert clip_limits.shape == (num_tiles_y, num_tiles_x), \
            f"clip_limits must match tile grid size {(num_tiles_y, num_tiles_x)}"

    luts = np.zeros((num_tiles_y, num_tiles_x, 256), dtype=np.uint8)
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            tile = channel[y * tile_size:y * tile_size + tile_size,
                           x * tile_size:x * tile_size + tile_size]
            int_clip_limit = int((clip_limits[y][x] / 256) * (tile_size ** 2))  # TODO: rm `/ 256`; just here to match opencv createCLAHE behavior
            luts[y][x] = compute_tile_lut(tile, int_clip_limit)

    # Interpolate each pixel
    output = np.zeros_like(channel)

    if interpolate:
        y = np.arange(h)
        x = np.arange(w)
        yy, xx = np.meshgrid(y, x, indexing='ij')

        gy = yy / tile_size - 0.5
        gx = xx / tile_size - 0.5

        tile_ys = np.clip(np.floor(gy).astype(int), 0, num_tiles_y - 2).astype(int)
        tile_xs = np.clip(np.floor(gx).astype(int), 0, num_tiles_x - 2).astype(int)

        dy = gy - tile_ys
        dx = gx - tile_xs

        p = channel

        # Fetch LUT-mapped values from 4 surrounding tiles
        val00 = luts[tile_ys, tile_xs, p]
        val01 = luts[tile_ys, tile_xs + 1, p]
        val10 = luts[tile_ys + 1, tile_xs, p]
        val11 = luts[tile_ys + 1, tile_xs + 1, p]

        top = (1 - dx) * val00 + dx * val01
        bottom = (1 - dx) * val10 + dx * val11
        output = ((1 - dy) * top + dy * bottom).astype(np.uint8)
    else:
        output = np.zeros_like(channel)
        for y in range(h):
            for x in range(w):
                tile_ys = y // tile_size
                tile_x = x // tile_size
                output[y][x] = luts[tile_ys][tile_x][channel[y][x]]

    output = output[:unpadded_h, :unpadded_w]

    return uint8_to_float_array(output)


SOBEL_BLUR_KERNEL_SIZE = 15
SOBEL_KERNEL_SIZE = (3, 3)

TURBIDITY_FACTOR_SOBEL_WEIGHT = 0.6  # w_1
TURBIDITY_FACTOR_SATURATION_WEIGHT = 1.0 - TURBIDITY_FACTOR_SOBEL_WEIGHT  # w_2

NUMERATOR_SAFETY = 1e-6

@dataclass
class MeanStdDev:
    mean: float
    std_dev: float

def sobel_mean_std_dev(img: OrderedImage[FloatArray]) -> MeanStdDev:
    gray = cv2.cvtColor(img.cl, cv2.COLOR_RGB2GRAY)

    # gray = cv2.GaussianBlur(gray, (SOBEL_BLUR_KERNEL_SIZE, SOBEL_BLUR_KERNEL_SIZE), 0)

    # Could use Scharr or Canny instead

    # ddepth is output precision
    # grad_sum = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=SOBEL_KERNEL_SIZE[0])
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=SOBEL_KERNEL_SIZE[0])

    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=SOBEL_KERNEL_SIZE[1])

    # Take abs first to normalize w.r.t highest magnitude val
    grad_x = np.abs(grad_x)
    grad_y = np.abs(grad_y)

    grad_x = grad_x / grad_x.max()
    grad_y = grad_y / grad_y.max()

    grad_sum = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    msd = cv2.meanStdDev(grad_sum)

    return MeanStdDev(msd[0].item(), msd[1].item())

def saturation_mean_std_dev(img: OrderedImage[FloatArray]) -> MeanStdDev:
    hsv = OrderedImage(cv2.cvtColor(img.cl, cv2.COLOR_BGR2HSV).astype(np.float32))
    msd = cv2.meanStdDev(hsv.cf[1])
    return MeanStdDev(msd[0].item(), msd[1].item())

def estimate_turbidity(img: OrderedImage[FloatArray]) -> float:
    sobel_msd = sobel_mean_std_dev(img)
    sat_msd = saturation_mean_std_dev(img)

    weighted_sobel = TURBIDITY_FACTOR_SOBEL_WEIGHT * (sobel_msd.std_dev / (sobel_msd.mean + NUMERATOR_SAFETY))
    weighted_sat = TURBIDITY_FACTOR_SATURATION_WEIGHT * (sat_msd.std_dev / (sat_msd.mean + NUMERATOR_SAFETY))
    turbidity_estimate = weighted_sobel + weighted_sat

    return turbidity_estimate

# def split_image_to_blocks(img: OrderedImage[FloatArray]) ->

MAX_CLIP_LIMIT = 0.05 * 256  # OpenCV normalizes clip limit by dividing by max int value
                                # int((clipLimit / 256) * (block_width * block_height))
                                # We want to specify raw fraction, so multiply first
CLIP_LIMIT_SAFETY = 1e-6  # TODO: the highest turbidity block always uses this, seems wrong

def tclahe(img: OrderedImage[FloatArray], n: int = 64, interpolate: bool = True) -> OrderedImage[FloatArray]:
    padded_w_in_blocks = img.w // n + 1
    padded_w = padded_w_in_blocks * n
    padded_h_in_blocks = img.h // n + 1
    padded_h = padded_h_in_blocks * n

    img_before_padding = img.clone()

    padding_w = padded_w - img.w
    padding_h = padded_h - img.h

    channels: list[FloatArray] = []
    for channel in img.cf:
        channels.append(cv2.copyMakeBorder(channel, 0, padding_h, 0, padding_w, cv2.BORDER_DEFAULT).astype(np.float32))

    img_padded = OrderedImage(cv2.merge(channels).astype(np.float32))

    blocks = []

    for y in range(padded_h_in_blocks):
        for x in range(padded_w_in_blocks):
            blocks.append(
                OrderedImage(img_padded.cl[y * n:y * n + n, x * n:x * n + n])
            )

    turbidities = []
    for block in blocks:
        turbidities.append(estimate_turbidity(block))

    clip_limits = []
    for turbidity in turbidities:
        clip_limits.append(MAX_CLIP_LIMIT * (1 - (turbidity - min(turbidities)) / (max(turbidities) - min(turbidities))) + CLIP_LIMIT_SAFETY)

    clip_limits_array = np.asarray(clip_limits)
    clip_limits_array = np.reshape(clip_limits, shape=(padded_h_in_blocks, padded_w_in_blocks))

    clahed3 = adaptive_clahe(img_before_padding, clip_limits_array, tile_size=n, interpolate=interpolate)

    return clahed3