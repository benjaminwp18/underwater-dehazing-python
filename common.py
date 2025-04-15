import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias
from dataclasses import dataclass
from enum import Enum
import cv2

FloatArray: TypeAlias = NDArray[np.float32]
Uint8Array: TypeAlias = NDArray[np.uint8]
type ImageArray = FloatArray | Uint8Array

class OrderedImage[T: (FloatArray, Uint8Array)]:
    def __init__(self, cl: T):
        self.cl: T = cl
        self.cf: T = cl_to_cf(cl)
        self.h = cl.shape[0]
        self.w = cl.shape[1]

    def set_cl(self, cl: T) -> None:
        self.cl = cl
        self.cf = cl_to_cf(cl)

    def set_cf(self, cf: T) -> None:
        self.cf = cf
        self.cl = cf_to_cl(cf)

    def to_floats(self) -> 'OrderedImage[FloatArray]':
        return OrderedImage(uint8_to_float_array(self.cl))

    def to_uint8s(self) -> 'OrderedImage[Uint8Array]':
        return OrderedImage(float_to_uint8_array(self.cl))

    def clone(self) -> 'OrderedImage':
        return OrderedImage(self.cl)

def float_to_uint8_array(float_array: FloatArray) -> Uint8Array:
    return (np.copy(float_array) * 255).astype(np.uint8)

def uint8_to_float_array(uint8_array: Uint8Array) -> FloatArray:
    return np.copy(uint8_array).astype(np.float32) / 255.0

def cf_to_cl[T: (FloatArray, Uint8Array)](img_cf: T) -> T:
    return np.transpose(img_cf, (1, 2, 0))

def cl_to_cf[T: (FloatArray, Uint8Array)](img_cl: T) -> T:
    return np.transpose(img_cl, (2, 0, 1))

# class BlockSet[T: (FloatArray, Uint8Array)]:
#     def __init__(self, img: OrderedImage[T]):

@dataclass
class MeanStdDev:
    mean: float
    std_dev: float

class StatType(Enum):
    Min = 0
    Max = 1
    Mean = 2

@dataclass
class MinMax:
    min: float
    max: float

@dataclass
class MinMaxMean:
    min: float
    max: float
    mean: float

class Stats:
    def __init__(self, stat_list: tuple[MinMaxMean, MinMaxMean, MinMaxMean]):
        self._stat_list = stat_list
        self._by_stat_type = {
            StatType.Min:  (self._stat_list[0].min,  self._stat_list[1].min,  self._stat_list[2].min),
            StatType.Max:  (self._stat_list[0].max,  self._stat_list[1].max,  self._stat_list[2].max),
            StatType.Mean: (self._stat_list[0].mean, self._stat_list[1].mean, self._stat_list[2].mean),
        }

    @staticmethod
    def from_img(img: OrderedImage[FloatArray]):
        means = cv2.mean(img.cl)[:-1]  # Remove fourth channel mean

        # [(min, max, (min_loc), (max_loc)), ..., ...]
        channel_min_max_locs = [cv2.minMaxLoc(c) for c in cv2.split(img.cl)]
        # [(min, max), ..., ...]
        extremes = [(locs[0], locs[1]) for locs in channel_min_max_locs]

        return Stats((
            MinMaxMean(extremes[0][0], extremes[0][1], means[0]),
            MinMaxMean(extremes[1][0], extremes[1][1], means[1]),
            MinMaxMean(extremes[2][0], extremes[2][1], means[2])
        ))

    def channel_stats(self, channel_index: int) -> MinMaxMean:
        return self._stat_list[channel_index]

    def stat_tuple(self, stat_type: StatType) -> tuple[float, float, float]:
        return self._by_stat_type[stat_type]

    def channel_idxs_sorted_by(self, stat_type: StatType) -> tuple[int, int, int]:
        array = np.argsort(self.stat_tuple(stat_type))
        return (array[0], array[1], array[2])

    def __str__(self) -> str:
        return (f'c0: {self._stat_list[0].min} < μ={self._stat_list[0].mean} < {self._stat_list[0].max}\n'
                f'c1: {self._stat_list[1].min} < μ={self._stat_list[1].mean} < {self._stat_list[1].max}\n'
                f'c2: {self._stat_list[2].min} < μ={self._stat_list[2].mean} < {self._stat_list[2].max}\n')