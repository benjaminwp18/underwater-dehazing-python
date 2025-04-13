from common import MinMax

class ColorCorrection:
    COLOR_LOSS_EPSILON = 0.01  # ε in (6)
    I_O = MinMax(0.0, 1.0)  # Output intensity bounds, I_o^min & I_o^max in (3)

class TCLAHE:
    SOBEL_BLUR_KERNEL_SIZE = 15
    SOBEL_KERNEL_SIZE = (3, 3)

    TURBIDITY_FACTOR_SOBEL_WEIGHT = 0.6  # w_1
    TURBIDITY_FACTOR_SATURATION_WEIGHT = 1.0 - TURBIDITY_FACTOR_SOBEL_WEIGHT  # w_2

    NUMERATOR_SAFETY = 1e-6

    MAX_CLIP_LIMIT = 0.05 * 256  # OpenCV normalizes clip limit by dividing by max int value
                                # int((clipLimit / 256) * (block_width * block_height))
                                # We want to specify raw fraction, so multiply first
    CLIP_LIMIT_SAFETY = 1e-6  # TODO: the highest turbidity block always uses this, seems wrong