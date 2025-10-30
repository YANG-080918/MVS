import numpy as np
from scipy.ndimage import convolve


def gradient_matching_a(
    left_vertical_image: np.ndarray,
    left_horizontal_gradient: np.ndarray,
    right_vertical_gradient: np.ndarray,
    right_horizontal_gradient: np.ndarray,
    max_disparity: int,
    truncation: float,
):
    lv = np.asarray(left_vertical_image, dtype=np.float64)
    lh = np.asarray(left_horizontal_gradient, dtype=np.float64)
    rv = np.asarray(right_vertical_gradient, dtype=np.float64)
    rh = np.asarray(right_horizontal_gradient, dtype=np.float64)

    rows, cols = lv.shape
    left_cost = np.zeros((rows, cols, max_disparity), dtype=np.float64)
    right_cost = np.zeros((rows, cols, max_disparity), dtype=np.float64)

    for d in range(max_disparity):
        for i in range(cols):
            for j in range(rows):
                if i - d >= 0:
                    diff_v = abs(lv[j, i] - rv[j, i - d])
                    if diff_v > truncation:
                        lc = truncation
                    else:
                        lc = diff_v

                    diff_h = abs(lh[j, i] - rh[j, i - d])
                    if diff_h > truncation:
                        lc += truncation
                    else:
                        lc += diff_h
                else:
                    lc = truncation * 2.0
                left_cost[j, i, d] = lc

                if i + d < cols:
                    diff_v_r = abs(lv[j, i + d] - rv[j, i])
                    if diff_v_r > truncation:
                        rc = truncation
                    else:
                        rc = diff_v_r

                    diff_h_r = abs(lh[j, i + d] - rh[j, i])
                    if diff_h_r > truncation:
                        rc += truncation
                    else:
                        rc += diff_h_r
                else:
                    rc = truncation * 2.0
                right_cost[j, i, d] = rc

    return left_cost, right_cost


def gradient_matching(left_image, right_image, max_disparity, gradient_truncation):
    left_image = np.asarray(left_image, dtype=np.float64)
    right_image = np.asarray(right_image, dtype=np.float64)

    G1 = np.array([
        [1,  2,  0, -2, -1],
        [4,  8,  0, -8, -4],
        [6, 12,  0, -12, -6],
        [4,  8,  0, -8, -4],
        [1,  2,  0, -2, -1]
    ], dtype=np.float64) / 96.0
    G2 = G1.T

    left_vertical_gradient = convolve(left_image, G1, mode="reflect")
    left_horizontal_gradient = convolve(left_image, G2, mode="reflect")
    right_vertical_gradient = convolve(right_image, G1, mode="reflect")
    right_horizontal_gradient = convolve(right_image, G2, mode="reflect")

    left_cost, right_cost = gradient_matching_a(
        left_vertical_gradient,
        left_horizontal_gradient,
        right_vertical_gradient,
        right_horizontal_gradient,
        max_disparity,
        gradient_truncation,
    )

    return left_cost, right_cost


if __name__ == "__main__":
    h, w = 50, 60
    left_img = np.random.rand(h, w) * 255
    right_img = np.random.rand(h, w) * 255
    D = 8
    trunc = 10.0

    lcost, rcost = gradient_matching(left_img, right_img, D, trunc)

    print("left_cost shape:", lcost.shape)
    print("right_cost shape:", rcost.shape)
