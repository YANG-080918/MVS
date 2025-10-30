import numpy as np
from types import SimpleNamespace

try:
    from skimage.color import rgb2gray
    from skimage.feature import canny
except ImportError:
    rgb2gray = None
    canny = None

try:
    from scipy.ndimage import distance_transform_edt
    from scipy.signal import medfilt2d
except ImportError:
    distance_transform_edt = None
    medfilt2d = None


def fn_confidence_measure(image, dsiL, dsiR, max_disparity, min_disparity, confParam):
    image_gray = to_gray(image)
    dispL_idx = np.argmin(dsiL, axis=2)
    disparityL = dispL_idx + (min_disparity - 1)
    dispR_idx = np.argmin(dsiR, axis=2)
    disparityR = dispR_idx - min_disparity + 1
    H, W, D = dsiL.shape
    dsi = dsiL.reshape(H * W, D).T
    dsi_s = np.sort(dsi, axis=0)
    confidence = np.zeros((22, H * W), dtype=np.float64)
    methods = []
    cnt = 0
    c_1, c_2, c_hat_2, c_sum, NOI = compute_base_costs(dsi)
    sigma = getattr(confParam, "sigma", 1.0)
    pdf = np.exp(-dsi / (sigma ** 2))
    pdf = pdf / np.sum(pdf, axis=0, keepdims=True)
    if getattr(confParam, "useCost", False):
        confidence[cnt, :] = -c_1
        methods.append("MC")
        cnt += 1
    if getattr(confParam, "usePKR", False):
        confidence[cnt, :] = c_hat_2 / (c_1 + 1e-12)
        methods.append("PKR")
        cnt += 1
    if getattr(confParam, "usePKRN", False):
        confidence[cnt, :] = c_2 / (c_1 + 1e-12)
        methods.append("PKRN")
        cnt += 1
    if getattr(confParam, "useMM", False):
        confidence[cnt, :] = c_hat_2 - c_1
        methods.append("MM")
        cnt += 1
    if getattr(confParam, "useMMN", False):
        confidence[cnt, :] = c_2 - c_1
        methods.append("NMM")
        cnt += 1
    if getattr(confParam, "useWM", False):
        confidence[cnt, :] = (c_hat_2 - c_1) / (c_sum + 1e-12)
        methods.append("WM")
        cnt += 1
    if getattr(confParam, "useWMN", False):
        confidence[cnt, :] = (c_2 - c_1) / (c_sum + 1e-12)
        methods.append("NWM")
        cnt += 1
    if getattr(confParam, "useLRD", False):
        val = compute_LRD(disparityL, c_1, c_2, dsiR)
        confidence[cnt, :] = val
        methods.append("LRD")
        cnt += 1
    if getattr(confParam, "useLC", False):
        val = compute_LC(dsi)
        confidence[cnt, :] = val
        methods.append("LC")
        cnt += 1
    if getattr(confParam, "usePER", False):
        val = compute_perturbation(dsi_s)
        confidence[cnt, :] = val
        methods.append("PER")
        cnt += 1
    if getattr(confParam, "useAML", False):
        val = compute_AML(dsi_s)
        confidence[cnt, :] = val
        methods.append("AML")
        cnt += 1
    if getattr(confParam, "useNOI", False):
        confidence[cnt, :] = NOI
        methods.append("NOI")
        cnt += 1
    if getattr(confParam, "useMLM", False):
        val = np.max(pdf, axis=0)
        confidence[cnt, :] = val
        methods.append("MLM")
        cnt += 1
    if getattr(confParam, "useNEM", False):
        val = -np.sum(pdf * np.log(pdf + 1e-12), axis=0)
        confidence[cnt, :] = val
        methods.append("NEM")
        cnt += 1
    if getattr(confParam, "useLRC", False):
        val = compute_LRC(disparityL, disparityR)
        confidence[cnt, :] = val
        methods.append("LRC")
        cnt += 1
    if getattr(confParam, "useDispVar", False):
        radii = getattr(confParam, "radius_DVAR", [])
        for r in radii:
            val = -compute_Var(disparityL, r)
            confidence[cnt, :] = val
            methods.append(f"DVAR{r}")
            cnt += 1
    if getattr(confParam, "useMedDev", False):
        radii = getattr(confParam, "radius_MDD", [])
        for r in radii:
            val = -compute_MedDev(disparityL, r)
            confidence[cnt, :] = val
            methods.append(f"MDD{r}")
            cnt += 1
    if getattr(confParam, "useDTD", False):
        val = compute_DD(disparityL)
        confidence[cnt, :] = val
        methods.append("DTD")
        cnt += 1
    if getattr(confParam, "useGRAD", False):
        val = compute_ImGrad(image_gray)
        confidence[cnt, :] = val
        methods.append("GRAD")
        cnt += 1
    if getattr(confParam, "useDTE", False):
        val = compute_DD(image_gray)
        confidence[cnt, :] = val
        methods.append("DTE")
        cnt += 1
    if getattr(confParam, "useDistLeftBorder", False):
        val = compute_DistLeftBorder(disparityL, max_disparity)
        confidence[cnt, :] = val
        methods.append("DLB")
        cnt += 1
    if getattr(confParam, "useDistImgBorder", False):
        trunc_val = 5
        val = compute_DistBorder(disparityL, trunc_val)
        confidence[cnt, :] = val
        methods.append("DIB")
        cnt += 1
    confidence = confidence[:cnt, :]
    return confidence, methods


def to_gray(img):
    img = np.asarray(img)
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim == 3 and img.shape[2] == 3:
        if rgb2gray is not None:
            return rgb2gray(img).astype(np.float64)
        return (0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]).astype(np.float64)
    raise ValueError("image shape not supported")


def compute_base_costs(dsi):
    c_1 = np.min(dsi, axis=0)
    c_idx = np.argmin(dsi, axis=0)
    c_sum = np.sum(dsi, axis=0)
    D, N = dsi.shape
    up = np.vstack([dsi[0:1, :], dsi])
    down = np.vstack([dsi, dsi[-1:, :]])
    cond1 = (dsi - up[:-1, :]) < 0
    cond2 = (down[1:, :] - dsi) < 0
    inflections = cond1 & cond2
    dsi2 = dsi.copy()
    dsi2[c_idx, np.arange(N)] = np.inf
    c_2 = np.min(dsi2, axis=0)
    dsi3 = dsi.copy()
    dsi3[~inflections] = np.inf
    c_hat_2 = np.min(dsi3, axis=0)
    NOI = np.sum(inflections, axis=0)
    return c_1, c_2, c_hat_2, c_sum, NOI


def compute_perturbation(dsi_s):
    s = 0.1
    diff = dsi_s[1:, :] - dsi_s[0:1, :]
    val = -np.sum(np.exp(-(diff ** 2) / (s ** 2)), axis=0)
    return val


def compute_AML(dsi_s):
    sigma = 0.2
    diff = dsi_s[1:, :] - dsi_s[0:1, :]
    val = 1.0 / np.sum(np.exp(-(diff ** 2) / (2 * sigma ** 2)), axis=0)
    return val


def compute_LC(dsi):
    gamma = 480.0
    D, N = dsi.shape
    pad = np.full((D + 2, N), np.nan, dtype=np.float64)
    pad[1:-1, :] = dsi
    min_val = np.min(np.round(pad), axis=0)
    min_idx = np.argmin(np.round(pad), axis=0)
    out = np.zeros(N, dtype=np.float64)
    for k in range(N):
        idx = min_idx[k]
        up_v = pad[idx - 1, k]
        dw_v = pad[idx + 1, k]
        out[k] = (max(up_v, dw_v) - min_val[k]) / gamma
    return out


def compute_LRD(disparityL, c_1, c_2, dsiR):
    H, W = disparityL.shape
    _, _, D = dsiR.shape
    lrd_map = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            left_val = disparityL[i, j]
            offset = int(round(j - float(left_val)))
            if 0 <= offset < W:
                d = int(left_val)
                d = max(0, min(d, D - 1))
                cost_at_disp = dsiR[i, offset, d]
                min_cost = np.min(dsiR[i, offset, :])
                lrd_map[i, j] = abs(cost_at_disp - min_cost)
            else:
                lrd_map[i, j] = 0.0
    val = (c_2 - c_1) / (lrd_map.reshape(-1) + 1e-4)
    return val


def compute_LRC(disparityL, disparityR):
    H, W = disparityL.shape
    lrc_map = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            left_val = disparityL[i, j]
            offset = int(round(j - float(left_val)))
            if 0 <= offset < W:
                right_val = disparityR[i, offset]
                lrc_map[i, j] = abs(float(right_val) - float(left_val))
            else:
                lrc_map[i, j] = -1
    max_disp = np.max(disparityL)
    return (-lrc_map.reshape(-1) + max_disp)


def compute_Var(image, r):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3:
        image = to_gray(image)
    meanImg, denom = compute_mean(image, r)
    w = 2 * r + 1
    from numpy.lib.stride_tricks import sliding_window_view
    pad = image_padding(image, w)
    patches = sliding_window_view(pad, (w, w))
    diff2 = (patches - meanImg[..., None, None]) ** 2
    varImg = diff2.sum(axis=(2, 3)) / (denom - 1 + 1e-12)
    return varImg.reshape(-1)


def compute_MedDev(image, r):
    image = np.asarray(image, dtype=np.float64)
    w = 2 * r + 1
    if medfilt2d is None:
        raise ImportError("scipy is required")
    med = medfilt2d(image, kernel_size=w)
    val = np.abs(med - image)
    return val.reshape(-1)


def compute_DD(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3:
        image = to_gray(image)
    if canny is None or distance_transform_edt is None:
        raise ImportError("skimage and scipy are required")
    edge_map = canny(image)
    dist = distance_transform_edt(~edge_map)
    return dist.reshape(-1)


def compute_ImGrad(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        gx, gy = np.gradient(image)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        return mag.reshape(-1)
    else:
        mags = []
        for c in range(image.shape[2]):
            gx, gy = np.gradient(image[..., c])
            mags.append(np.sqrt(gx ** 2 + gy ** 2))
        mag = np.max(np.stack(mags, axis=2), axis=2)
        return mag.reshape(-1)


def compute_DistLeftBorder(disparity_map, max_disparity):
    H, W = disparity_map.shape
    x = np.arange(1, W + 1)
    x = np.minimum(x, max_disparity)
    val = np.tile(x, (H, 1))
    return val.reshape(-1)


def compute_DistBorder(disparity_map, trunc_val):
    disparity_map = np.asarray(disparity_map)
    H, W = disparity_map.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if distance_transform_edt is None:
        raise ImportError("scipy is required")
    dist = distance_transform_edt(~border)
    dist = np.minimum(dist, trunc_val)
    return dist.reshape(-1)


def compute_mean(image, r):
    H, W = image.shape[:2]
    ones = np.ones((H, W), dtype=np.float64)
    denom = boxfilter(ones, r)
    if image.ndim == 3:
        gray = to_gray(image)
        meanImg = boxfilter(gray, r) / (denom + 1e-12)
    else:
        meanImg = boxfilter(image, r) / (denom + 1e-12)
    return meanImg, denom


def boxfilter(imSrc, r):
    H, W = imSrc.shape
    imDst = np.zeros_like(imSrc, dtype=np.float64)
    imCum = np.cumsum(imSrc, axis=0)
    imDst[0:r + 1, :] = imCum[r:2 * r + 1, :]
    imDst[r + 1:H - r, :] = imCum[2 * r + 1:H, :] - imCum[0:H - 2 * r - 1, :]
    imDst[H - r:H, :] = (imCum[H - 1:H, :]) - imCum[H - 2 * r - 1:H - r - 1, :]
    imCum = np.cumsum(imDst, axis=1)
    out = np.zeros_like(imSrc, dtype=np.float64)
    out[:, 0:r + 1] = imCum[:, r:2 * r + 1]
    out[:, r + 1:W - r] = imCum[:, 2 * r + 1:W] - imCum[:, 0:W - 2 * r - 1]
    out[:, W - r:W] = imCum[:, W - 1:W] - imCum[:, W - 2 * r - 1:W - r - 1]
    return out


def image_padding(image, w_size):
    H, W = image.shape[:2]
    half = w_size // 2
    if image.ndim == 2:
        new_img = np.zeros((H + w_size - 1, W + w_size - 1), dtype=image.dtype)
        new_img[half:half + H, half:half + W] = image
    else:
        C = image.shape[2]
        new_img = np.zeros((H + w_size - 1, W + w_size - 1, C), dtype=image.dtype)
        new_img[half:half + H, half:half + W, :] = image
    return new_img
