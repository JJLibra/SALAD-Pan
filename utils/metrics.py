import os
import math
import numpy as np
from torchvision.utils import make_grid
import torch
from numpy.linalg import norm
import cv2
import torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage.filters import sobel, convolve
from scipy.stats import pearsonr
import sewar as sewar_api


def tensor2img_4C(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    修改输入通道数量，由RGB到包含4个或8个通道的图像，但是只取其中可见光的三个通道[4,256,256]
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2 or n_dim == 1:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
        # 保存需要的sr结果

    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    # if mode == 'gray':
    #     cv2.imwrite(img_path, img)
    # else:
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


# numpy version
def SSIM_numpy(x_true, x_pred, data_range, sewar=False):
    r"""
    Args:
        x_true (np.ndarray): target image, shape like [H, W, C]
        x_pred (np.ndarray): predict image, shape like [H, W, C]
        data_range (int): max_value of the image
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: SSIM value
    """
    if sewar:
        return sewar_api.ssim(x_true, x_pred, MAX=data_range)[0]

    return structural_similarity(x_true, x_pred, data_range=data_range, channel_axis=-1)


def MPSNR_numpy(x_true, x_pred, data_range):
    r"""
    Args:
        x_true (np.ndarray): target image, shape like [H, W, C]
        x_pred (np.ndarray): predict image, shape like [H, W, C]
        data_range (int): max_value of the image
    Returns:
        float: Mean PSNR value
    """

    tmp = []
    for c in range(x_true.shape[-1]):
        tmp.append(peak_signal_noise_ratio(x_true[:, :, c], x_pred[:, :, c], data_range=data_range))
    return np.mean(tmp)


def SAM_numpy(x_true, x_pred, sewar=False, eps=1e-12):
    r"""
    Look at paper:
    `Discrimination among semiarid landscape endmembers using the spectral angle mapper (sam) algorithm` for details

    Args:
        x_true (np.ndarray): target image, shape like [H, W, C]
        x_pred (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: SAM value
    """
    if sewar:
        return sewar_api.sam(x_true, x_pred)

    x = np.asarray(x_true, dtype=np.float64)
    y = np.asarray(x_pred, dtype=np.float64)

    if x.ndim != 3 or x.shape != y.shape:
        raise ValueError(f"x_true/x_pred 需同形且为 [H,W,C]，但得到 {x.shape} 和 {y.shape}")

    # [H, W]
    dot = np.sum(x * y, axis=-1)
    nx = np.linalg.norm(x, axis=-1)
    ny = np.linalg.norm(y, axis=-1)

    valid = (nx > eps) & (ny > eps)
    if not np.any(valid):
        return 0.0

    denom = nx * ny
    cos = np.empty_like(dot, dtype=np.float64)
    cos[:] = 1.0
    cos[valid] = dot[valid] / np.maximum(denom[valid], eps)
    cos = np.clip(cos, -1.0, 1.0)

    ang = np.arccos(cos[valid])
    sam_deg = np.degrees(np.mean(ang))
    return float(sam_deg)


def SCC_numpy(ms, ps, sewar=False):
    r"""
    Look at paper:
    `A wavelet transform method to merge Landsat TM and SPOT panchromatic data` for details

    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: SCC value
    """
    if sewar:
        return sewar_api.scc(ms, ps)

    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')
    scc = 0.0
    for i in range(ms.shape[2]):
        a = (ps_sobel[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        b = (ms_sobel[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        scc += pearsonr(a, b)[0]
    return scc / ms.shape[2]


def CC_numpy(ms, ps, sewar=False):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
    Returns:
        float: CC value
    """
    if sewar:
        return sewar_api.scc(ms, ps)

    cc = 0.0
    for i in range(ms.shape[2]):
        a = (ps[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        b = (ms[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        cc += pearsonr(a, b)[0]
    return cc / ms.shape[2]


def Q4_numpy(ms, ps):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
    Returns:
        float: Q4 value
    """

    def conjugate(a):
        sign = -1 * np.ones(a.shape)
        sign[0, :] = 1
        return a * sign

    def product(a, b):
        a = a.reshape(a.shape[0], 1)
        b = b.reshape(b.shape[0], 1)
        R = np.dot(a, b.transpose())
        r = np.zeros(4)
        r[0] = R[0, 0] - R[1, 1] - R[2, 2] - R[3, 3]
        r[1] = R[0, 1] + R[1, 0] + R[2, 3] - R[3, 2]
        r[2] = R[0, 2] - R[1, 3] + R[2, 0] + R[3, 1]
        r[3] = R[0, 3] + R[1, 2] - R[2, 1] + R[3, 0]
        return r

    imps = np.copy(ps)
    imms = np.copy(ms)
    vec_ps = imps.reshape(imps.shape[1] * imps.shape[0], imps.shape[2])
    vec_ps = vec_ps.transpose(1, 0)
    vec_ms = imms.reshape(imms.shape[1] * imms.shape[0], imms.shape[2])
    vec_ms = vec_ms.transpose(1, 0)
    m1 = np.mean(vec_ps, axis=1)
    d1 = (vec_ps.transpose(1, 0) - m1).transpose(1, 0)
    s1 = np.mean(np.sum(d1 * d1, axis=0))
    m2 = np.mean(vec_ms, axis=1)
    d2 = (vec_ms.transpose(1, 0) - m2).transpose(1, 0)
    s2 = np.mean(np.sum(d2 * d2, axis=0))
    Sc = np.zeros(vec_ms.shape)
    d2 = conjugate(d2)
    for i in range(vec_ms.shape[1]):
        Sc[:, i] = product(d1[:, i], d2[:, i])
    C = np.mean(Sc, axis=1)
    Q4 = 4 * np.sqrt(np.sum(m1 * m1) * np.sum(m2 * m2) * np.sum(C * C)) / (s1 + s2) / (
            np.sum(m1 * m1) + np.sum(m2 * m2))
    return Q4


def Q8_numpy(ms, ps):
    r"""
    Args:
        ms (np.ndarray): target image, shape [H, W, C] with C=8, values in [0,1]
        ps (np.ndarray): predict image, shape [H, W, C] with C=8, values in [0,1]
    Returns:
        float: Q8 value in [0, 1], and Q8(ms, ms) == 1.0 (up to tiny eps)
    Notes:
        八元数乘法采用 Cayley–Dickson 构造：
        设 a=(a1,a2), b=(b1,b2)，其中 a1,a2,b1,b2 为四元数，
        则 a*b = ( a1*b1 - conj(b2)*a2 , b2*a1 + a2*conj(b1) )
    """
    import numpy as np

    def qconj(q):
        # q: (4,) -> (4,)
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)

    def qmul(a, b):
        # a,b: (4,) -> (4,)
        a0, a1, a2, a3 = a
        b0, b1, b2, b3 = b
        return np.array([
            a0*b0 - a1*b1 - a2*b2 - a3*b3,
            a0*b1 + a1*b0 + a2*b3 - a3*b2,
            a0*b2 - a1*b3 + a2*b0 + a3*b1,
            a0*b3 + a1*b2 - a2*b1 + a3*b0
        ], dtype=a.dtype)

    def oconj(o):
        out = o.copy()
        out[1:] = -out[1:]
        return out

    def omul(a, b):
        a1, a2 = a[:4], a[4:]
        b1, b2 = b[:4], b[4:]
        c1 = qmul(a1, b1) - qmul(qconj(b2), a2)    # a1*b1 - conj(b2)*a2
        c2 = qmul(b2, a1) + qmul(a2, qconj(b1))    # b2*a1 + a2*conj(b1)
        return np.concatenate([c1, c2], axis=0)

    imps = np.asarray(ps)
    imms = np.asarray(ms)
    H, W, C = imps.shape
    assert C == 8, f"Q8 requires C=8, got C={C}"

    vec_ps = imps.reshape(H*W, C).T   # (8, N)
    vec_ms = imms.reshape(H*W, C).T   # (8, N)

    m1 = np.mean(vec_ps, axis=1)      # (8,)
    d1 = (vec_ps.T - m1).T            # (8, N)
    s1 = np.mean(np.sum(d1 * d1, axis=0))

    m2 = np.mean(vec_ms, axis=1)      # (8,)
    d2 = (vec_ms.T - m2).T            # (8, N)
    s2 = np.mean(np.sum(d2 * d2, axis=0))

    d2 = oconj(d2)
    Sc = np.zeros_like(vec_ms)
    for i in range(vec_ms.shape[1]):
        Sc[:, i] = omul(d1[:, i], d2[:, i])

    Cvec = np.mean(Sc, axis=1)        # (8,)
    num_means = np.sum(m1*m1) * np.sum(m2*m2)
    num_cross = np.sum(Cvec*Cvec)
    denom_var  = (s1 + s2)
    denom_mean = (np.sum(m1*m1) + np.sum(m2*m2))

    eps = 1e-12
    Q8 = 4.0 * np.sqrt(max(num_means, 0.0) * max(num_cross, 0.0) + eps) / (denom_var + eps) / (denom_mean + eps)

    if not np.isfinite(Q8):
        Q8 = 0.0
    return float(min(Q8, 1.0))


def RMSE_numpy(ms, ps, sewar=False):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: RMSE value
    """
    if sewar:
        return sewar_api.rmse(ms, ps)

    d = (ms - ps) ** 2
    rmse = np.sqrt(np.sum(d) / (d.shape[0] * d.shape[1]))
    return rmse


def ERGAS_numpy(ms, ps, ratio=0.25, sewar=False):
    r"""
    Look at paper:
    `Quality of high resolution synthesised images: Is there a simple criterion?` for details

    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: ERGAS value
    """
    if sewar:
        return sewar_api.ergas(ms, ps)

    m, n, d = ms.shape
    summed = 0.0
    for i in range(d):
        summed += (RMSE_numpy(ms[:, :, i], ps[:, :, i])) ** 2 / np.mean(ps[:, :, i]) ** 2
    ergas = 100 * ratio * np.sqrt(summed / d)
    return ergas


def UIQC_numpy(ms, ps, sewar=False):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: UIQC value
    """
    if sewar:
        return sewar_api.uqi(ms, ps)

    l = ms.shape[2]
    uiqc = 0.0
    for i in range(l):
        uiqc += QIndex_numpy(ms[:, :, i], ps[:, :, i])
    return uiqc / l


def QIndex_numpy(a, b):
    r"""
    Look at paper:
    `A universal image quality index` for details

    Args:
        a (np.ndarray): one-channel image, shape like [H, W]
        b (np.ndarray): one-channel image, shape like [H, W]
    Returns:
        float: Q index value
    """
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    temp = np.cov(a, b)
    d1 = temp[0, 0]
    cov = temp[0, 1]
    d2 = temp[1, 1]
    m1 = np.mean(a)
    m2 = np.mean(b)
    Q = 4 * cov * m1 * m2 / (d1 + d2) / (m1 ** 2 + m2 ** 2)

    return Q


def D_lambda_numpy(l_ms, ps, sewar=False):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (np.ndarray): LR MS image, shape like [H, W, C]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: D_lambda value
    """
    if sewar:
        return sewar_api.d_lambda(l_ms, ps)

    L = ps.shape[2]
    sum = 0.0
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += np.abs(QIndex_numpy(ps[:, :, i], ps[:, :, j]) - QIndex_numpy(l_ms[:, :, i], l_ms[:, :, j]))
    return sum / L / (L - 1)


def D_s_numpy(l_ms, pan, ps, sewar=False):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (np.ndarray): LR MS image, shape like [H, W, C]
        pan (np.ndarray): pan image, shape like [H, W]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: D_s value
    """
    if sewar:
        return sewar_api.d_s(pan, l_ms, ps)

    L = ps.shape[2]
    l_pan = cv2.pyrDown(pan)
    l_pan = cv2.pyrDown(l_pan)
    sum = 0.0
    for i in range(L):
        sum += np.abs(QIndex_numpy(ps[:, :, i], pan) - QIndex_numpy(l_ms[:, :, i], l_pan))
    return sum / L


def FCC_numpy(pan, ps):
    r"""
    Look at paper:
    `A wavelet transform method to merge landsat TM and SPOT panchromatic data` for details

    Args:
        pan (np.ndarray): pan image, shape like [H, W]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
    Returns:
        float: FCC value
    """
    k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    fcc = []
    for i in range(ps.shape[2]):
        a = convolve(ps[:, :, i], k, mode='constant').reshape(-1)
        b = convolve(pan, k, mode='constant').reshape(-1)
        fcc.append(pearsonr(b, a)[0])
    return np.max(fcc)


def SF_numpy(ps):
    r"""
    Look at paper:
    `Review of pixel-level image fusion` for details

    Args:
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
    Returns:
        float: SF value
    """
    f_row = np.mean((ps[:, 1:] - ps[:, :-1]) * (ps[:, 1:] - ps[:, :-1]))
    f_col = np.mean((ps[1:, :] - ps[:-1, :]) * (ps[1:, :] - ps[:-1, :]))
    return np.sqrt(f_row + f_col)


def SD_numpy(ps):
    r"""
    Look at paper:
    `A novel metric approach evaluation for the spatial enhancement of pansharpened images` for details

    Args:
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
    Returns:
        float: SD value
    """
    SD = 0.0
    for i in range(ps.shape[2]):
        SD += np.std(ps[:, :, i].reshape(-1))
    return SD / ps.shape[2]


def HQNR_numpy(l_ms, pan, ps):
    return (1.0 - D_lambda_numpy(l_ms, ps)) * (1.0 - D_s_numpy(l_ms, pan, ps))


# torch version
def SAM_torch(x_true, x_pred):
    r"""
    Look at paper:
    `Discrimination among semiarid landscape endmembers using the spectral angle mapper (sam) algorithm` for details

    Args:
        x_true (torch.Tensor): target images, shape like [N, C, H, W]
        x_pred (torch.Tensor): predict images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean SAM value of n images
    """
    dot_sum = torch.sum(x_true * x_pred, dim=1)
    norm_true = torch.norm(x_true, dim=1)
    norm_pred = torch.norm(x_pred, dim=1)
    a = torch.Tensor([1]).to(x_true.device, dtype=x_true.dtype)
    b = torch.Tensor([-1]).to(x_true.device, dtype=x_true.dtype)
    res = dot_sum / norm_pred / norm_true
    res = torch.max(torch.min(res, a), b)
    res = torch.acos(res) * 180 / 3.1415926
    sam = torch.mean(res)
    return sam


def sobel_torch(im):
    r"""
    Args:
        im (torch.Tensor): images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: images after sobel filter
    """
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = torch.Tensor(sobel_kernel).to(im.device, dtype=im.dtype)
    return F.conv2d(im, weight)


def SCC_torch(x, y):
    r"""
    Args:
        x (torch.Tensor): target images, shape like [N, C, H, W]
        y (torch.Tensor): predict images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean SCC value of n images
    """
    x = sobel_torch(x)
    y = sobel_torch(y)

    vx = x - torch.mean(x, dim=(2, 3), keepdim=True)
    vy = y - torch.mean(y, dim=(2, 3), keepdim=True)
    scc = torch.sum(vx * vy, dim=(2, 3)) / torch.sqrt(torch.sum(vx * vx, dim=(2, 3))) / torch.sqrt(
        torch.sum(vy * vy, dim=(2, 3)))
    return torch.mean(scc)


def QIndex_torch(a, b, eps=1e-8):
    r"""
    Look at paper:
    `A universal image quality index` for details

    Args:
        a (torch.Tensor): one-channel images, shape like [N, H, W]
        b (torch.Tensor): one-channel images, shape like [N, H, W]
    Returns:
        torch.Tensor: Q index value of all images
    """
    E_a = torch.mean(a, dim=(1, 2))
    E_a2 = torch.mean(a * a, dim=(1, 2))
    E_b = torch.mean(b, dim=(1, 2))
    E_b2 = torch.mean(b * b, dim=(1, 2))
    E_ab = torch.mean(a * b, dim=(1, 2))
    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b
    return torch.mean(4 * cov_ab * E_a * E_b / ((var_a + var_b) * (E_a ** 2 + E_b ** 2) + eps))


def D_lambda_torch(l_ms, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_lambda value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += torch.abs(
                    QIndex_torch(ps[:, i, :, :], ps[:, j, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_ms[:, j, :, :]))
    return sum / L / (L - 1)


def D_s_torch(l_ms, pan, l_pan, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        pan (torch.Tensor): PAN images, shape like [N, C, H, W]
        l_pan (torch.Tensor): LR PAN images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_s value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        sum += torch.abs(
            QIndex_torch(ps[:, i, :, :], pan[:, 0, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_pan[:, 0, :, :]))
    return sum / L


def get_scores(target, sample, l_ms=None, pan=None):
    q4_score = Q4_numpy(target, sample)
    sam_score = SAM_numpy(target, sample)
    ergas_score = ERGAS_numpy(target, sample)
    scc_score = SCC_numpy(target, sample)
    rmse_score = RMSE_numpy(target, sample)
    ssim_score = SSIM_numpy(target, sample, data_range=1)
    mpsnr_score = MPSNR_numpy(target, sample, data_range=1)
    print(f"Q4: {q4_score}")
    print(f"SAM: {sam_score}")
    print(f"ERGAS: {ergas_score}")
    print(f"SCC: {scc_score}")
    print(f"RMSE: {rmse_score}")
    print(f"SSIM: {ssim_score}")
    print(f"MPSNR: {mpsnr_score}")
    if (l_ms is not None) and (pan is not None):
        d_lambda = D_lambda_numpy(l_ms, sample)
        d_s = D_s_numpy(l_ms, pan, sample)
        hqnr = (1 - d_lambda) * (1 - d_s)
        print(f"Dλ: {d_lambda}")
        print(f"Ds: {d_s}")
        print(f"HQNR: {hqnr}")
