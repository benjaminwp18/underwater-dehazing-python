# http://amandaduarte.com.br/turbid/
# https://ieeexplore.ieee.org/document/10770246

import time

import cv2
import numpy as np

from pathlib import Path

from common import OrderedImage, FloatArray, Uint8Array, cf_to_cl, float_to_uint8_array, uint8_to_float_array, Stats, StatType, MinMaxMean
from plotting import plot_channels, plot_img
from clahe import basic_CLAHE_float, basic_CLAHE_u8_cl, tclahe

import config

def load_img(path: Path) -> OrderedImage[FloatArray]:
    img_cl = cv2.imread(str(path))
    img_cl = cv2.resize(img_cl, (img_cl.shape[1] // 6, img_cl.shape[0] // 6)).astype(np.uint8)
    img_cl = uint8_to_float_array(img_cl.astype(np.uint8))

    img = OrderedImage(img_cl)

    return img

def color_loss(img: OrderedImage[FloatArray], c_s: int, c_m: int, c_l: int) -> float:
    stats = Stats.from_img(img)
    means = stats.stat_array(StatType.Mean)
    return np.abs(means[c_l] - means[c_m]) + np.abs(means[c_l] - means[c_s])

def channel_correction(img: OrderedImage[FloatArray]):
    og_stats = Stats.from_img(img)

    # Find large channel l, medium channel m, smallest channel s
    c_s, c_m, c_l = og_stats.channel_idxs_sorted_by(StatType.Mean)

    # Increase c_l dynamic range
    coef = (config.ColorCorrection.I_O.max - config.ColorCorrection.I_O.min) / (og_stats.channel(c_l).max - og_stats.channel(c_l).min)
    img.cf[c_l] = config.ColorCorrection.I_O.min + coef * (img.cf[c_l] - og_stats.channel(c_l).min)

    # Adjust c_m & c_s relative to c_l
    while config.ColorCorrection.COLOR_LOSS_EPSILON < color_loss(img, c_s, c_m, c_l):
        stats = Stats.from_img(img)
        for c in (c_s, c_m):
            coef = (stats.channel(c_l).mean - stats.channel(c).mean) / stats.channel(c_l).mean
            img.cf[c] = img.cf[c] + coef * img.cf[c_l]

def process_img(path: Path, display_intermediaries: bool = False):
    img = load_img(path)
    og_img = img.clone()
    if display_intermediaries:
        plot_channels(img)

    channel_correction(img)
    if display_intermediaries:
        plot_channels(img)

    # img_clahed_basic = basic_CLAHE_float(img)
    img_clahed = tclahe(img, n=64, interpolate=True)
    if display_intermediaries:
        plot_channels(img_clahed)
    plot_img(np.hstack((og_img.cl, img_clahed.cl)))
    # plot_img(np.hstack((
    #     cv2.copyMakeBorder(og_img.cl, 0, img_clahed.h - og_img.h, 0, img_clahed.w - og_img.w, cv2.BORDER_DEFAULT).astype(np.float32),
    #     img_clahed.cl
    # )))

def process_video(path: Path, side_by_side: bool = True, nth_frame: int = 3, num_seconds: int = -1):
    vid_reader = cv2.VideoCapture(str(path))

    fps = vid_reader.get(cv2.CAP_PROP_FPS)
    width = vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH)
    if side_by_side:
        width *= 2
    height = vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fourcc = vid_reader.get(cv2.CAP_PROP_FOURCC)  # Videos are encoded with h264, but that's under GPL
                                                    # Would need to compile from source
    fourcc = cv2.VideoWriter.fourcc(*'MJPG')
    vid_writer = cv2.VideoWriter('output.avi', int(fourcc), fps, (int(width), int(height)), True)

    i = 0
    while vid_reader.isOpened() and (num_seconds == -1 or fps * num_seconds < i):
        print(f'{i} ({i / np.round(fps, 2)} s)')
        i += 1
        ret, frame = vid_reader.read()

        if i % nth_frame == 0:
            if not ret:
                print('No more frames')
                break

            img = OrderedImage(uint8_to_float_array(frame))
            channel_correction(img)

            # clahed_u8_basic = basic_CLAHE_u8_cl(img, tile_size=128)
            clahed = tclahe(img, n=128, interpolate=True)
            clahed_u8 = clahed.to_uint8s()

            for n in range(nth_frame):
                if side_by_side:
                    vid_writer.write(np.hstack((frame, clahed_u8.cl)))
                else:
                    vid_writer.write(clahed_u8.cl)

    vid_writer.release()
    vid_reader.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # path = Path('crab') / '0.png'
    path = Path('milk') / 'a15.jpg'
    # path = Path('deepblue') / '8.jpg'

    start_time = time.time()
    # process_img(path)
    # process_video(Path('lake') / '2025.03.00' / 'blue.mp4', nth_frame=1)
    process_video(Path('lake') / '2025.03.00' / 'brown.2.mp4', nth_frame=1)
    print(f'--- {time.time() - start_time} seconds ---')
