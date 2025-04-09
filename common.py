import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

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