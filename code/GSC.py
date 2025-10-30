import numpy as np

def GSC(left_image, right_image, max_disparity, tau_c, win, K):
    left_image  = np.asarray(left_image,  dtype=np.float64)
    right_image = np.asarray(right_image, dtype=np.float64)
    H, W = left_image.shape
    assert right_image.shape == (H, W)

    value_t1_L = [[None for _ in range(W)] for _ in range(H)]
    rank_t1_L  = [[None for _ in range(W)] for _ in range(H)]
    value_t2_R = [[None for _ in range(W)] for _ in range(H)]
    rank_t2_R  = [[None for _ in range(W)] for _ in range(H)]
    c1_L       = [[None for _ in range(W)] for _ in range(H)]
    c2_R       = [[None for _ in range(W)] for _ in range(H)]

    for i in range(win, H - win):
        for j in range(win, W - win):
            lwin = left_image[i - win:i + win + 1, j - win:j + win + 1]
            x1 = left_image[i, j]
            lwin_flat = lwin.ravel()
            c1 = np.abs(lwin_flat - x1)
            c1_L[i][j] = c1
            order = np.argsort(c1)
            value_t1 = c1[order]
            value_t1_L[i][j] = value_t1
            rank_t1_L[i][j]  = order

    for i in range(win, H - win):
        for j in range(win, W - win):
            rwin = right_image[i - win:i + win + 1, j - win:j + win + 1]
            y1 = right_image[i, j]
            rwin_flat = rwin.ravel()
            c2 = np.abs(rwin_flat - y1)
            c2_R[i][j] = c2
            order = np.argsort(c2)
            value_t2 = c2[order]
            value_t2_R[i][j] = value_t2
            rank_t2_R[i][j]  = order

    CL = np.full((H, W, max_disparity), 0.0, dtype=np.float64)
    CR = np.full((H, W, max_disparity), 0.0, dtype=np.float64)

    for k in range(1, max_disparity + 1):
        kk = k - 1
        for p in range(win, H - win):
            for q in range(win, W - win):
                if q - k >= win:
                    value_t1 = value_t1_L[p][q]
                    rank_t1  = rank_t1_L[p][q]
                    value_t2 = value_t2_R[p][q - k]
                    rank_t2  = rank_t2_R[p][q - k]
                    c1 = c1_L[p][q]
                    c2 = c2_R[p][q - k]
                    idx2 = slice(1, K)
                    f_x_distance = abs(np.mean(c1[rank_t2[idx2]]) - np.mean(value_t1[idx2]))
                    f_y_distance = abs(np.mean(c2[rank_t1[idx2]]) - np.mean(value_t2[idx2]))
                    f_x_ds = (f_x_distance + f_y_distance) / 2.0
                    CL[p, q, kk] = f_x_ds
                else:
                    CL[p, q, kk] = tau_c

    for k in range(1, max_disparity + 1):
        kk = k - 1
        for p in range(win, H - win):
            for q in range(win, W - win):
                if q + k <= (W - win - 1):
                    value_t1 = value_t1_L[p][q + k]
                    rank_t1  = rank_t1_L[p][q + k]
                    value_t2 = value_t2_R[p][q]
                    rank_t2  = rank_t2_R[p][q]
                    c1 = c1_L[p][q + k]
                    c2 = c2_R[p][q]
                    idx2 = slice(1, K)
                    f_x_distance = abs(np.mean(c1[rank_t2[idx2]]) - np.mean(value_t1[idx2]))
                    f_y_distance = abs(np.mean(c2[rank_t1[idx2]]) - np.mean(value_t2[idx2]))
                    f_x_ds = (f_x_distance + f_y_distance) / 2.0
                    CR[p, q, kk] = f_x_ds
                else:
                    CR[p, q, kk] = tau_c

    CL_flat = CL.reshape(H * W, max_disparity, order='F')
    CR_flat = CR.reshape(H * W, max_disparity, order='F')

    lcost_new = np.zeros((H * W, max_disparity), dtype=np.float64)
    rcost_new = np.zeros((H * W, max_disparity), dtype=np.float64)

    D = max_disparity
    idx_top = int(np.floor(D * 0.8))
    if idx_top < 1:
        idx_top = 1
    tan = 4.0
    disp_idx = np.arange(1, D + 1, dtype=np.float64)

    for i in range(H * W):
        col2 = CL_flat[i, :].astype(np.float64, copy=True)
        zz = np.stack([disp_idx, col2], axis=1)
        zz_sorted = zz[np.argsort(zz[:, 1]), :]
        vals = zz_sorted[:idx_top, 1]
        xmin = np.min(vals)
        xmax = np.max(vals)
        if xmax > xmin:
            OutImg = (vals - xmin) / (xmax - xmin)
        else:
            OutImg = np.zeros_like(vals)
        zz_sorted[:idx_top, 1] = (OutImg + 1.0) * 2.0
        zz_sorted[idx_top:, 1] = tan
        back_sorted = zz_sorted[np.argsort(zz_sorted[:, 0]), :]
        lcost_new[i, :] = back_sorted[:, 1]
    lcost_new[np.isnan(lcost_new)] = 4.0

    for i in range(H * W):
        col2 = CR_flat[i, :].astype(np.float64, copy=True)
        zz = np.stack([disp_idx, col2], axis=1)
        zz_sorted = zz[np.argsort(zz[:, 1]), :]
        vals = zz_sorted[:idx_top, 1]
        xmin = np.min(vals)
        xmax = np.max(vals)
        if xmax > xmin:
            OutImg = (vals - xmin) / (xmax - xmin)
        else:
            OutImg = np.zeros_like(vals)
        zz_sorted[:idx_top, 1] = (OutImg + 1.0) * 2.0
        zz_sorted[idx_top:, 1] = tan
        back_sorted = zz_sorted[np.argsort(zz_sorted[:, 0]), :]
        rcost_new[i, :] = back_sorted[:, 1]
    rcost_new[np.isnan(rcost_new)] = 4.0

    return lcost_new, rcost_new
