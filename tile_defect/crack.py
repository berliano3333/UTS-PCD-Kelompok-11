"""Deteksi RETAK pada ubin bermotif (metode klasik, tanpa deep learning).

Ide utama: retak adalah garis tipis yang warnanya lebih PUDAR (abu-abu / putih) daripada
ubin di sekitarnya, sedangkan motif, serat kayu, dan nat ubin bukan.

Tahap yang dikerjakan di sini (dipetakan ke tahap UTS):
  Perbaikan kontras : peta ridge kroma (Hessian multi-skala) -> gerbang simetri sisi ->
                      normalisasi kepadatan lokal
  Segmentasi        : ambang histeresis (benih kuat + kandidat lemah, berbasis median + k*MAD)
  Pasca-proses      : closing, penekan tekstur garis-sejajar rapat, filter panjang dan
                      kelurusan komponen, penyambung celah antar-potongan retak
Hasil berupa mask retak saja (motif tidak ikut).
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import scratch_ops

_K_XX = np.array([[1, -2, 1]], np.float32)
_K_YY = _K_XX.T.copy()
_K_XY = 0.25 * np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], np.float32)


def _filter(img: np.ndarray, k: np.ndarray, use_scratch: bool) -> np.ndarray:
    if use_scratch and scratch_ops.available()["konvolusi"]:
        return scratch_ops.convolve2d(img.astype(np.float32), k.astype(np.float32))
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)


def hessian_valley(img: np.ndarray, sigma: float, use_scratch: bool = False):
    """Respons 'lembah' (garis yang nilainya lebih rendah dari sekitarnya) dan arah normalnya."""
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    s2 = sigma * sigma
    ixx = _filter(g, _K_XX, use_scratch) * s2
    iyy = _filter(g, _K_YY, use_scratch) * s2
    ixy = _filter(g, _K_XY, use_scratch) * s2
    half = 0.5 * (ixx + iyy)
    root = np.sqrt(0.25 * (ixx - iyy) ** 2 + ixy ** 2)
    l1, l2 = half + root, half - root              # l1 >= l2
    resp = np.maximum(l1 - np.maximum(l2, 0), 0)   # garis: l1 besar, l2 ~ 0 (bercak: keduanya besar -> 0)
    theta = 0.5 * np.arctan2(2 * ixy, ixx - iyy)   # sumbu lengkung terbesar = tegak lurus garis
    return resp, theta


def ridge_stage(im: np.ndarray, p, use_scratch: bool = False, cache: "dict | None" = None):
    """Peta retak ternormalisasi `Rn` dan arah normal `th` (disimpan float16 di cache)."""
    conv = bool(use_scratch and scratch_ops.available()["konvolusi"])
    key = ("ridge", tuple(p.sigmas), p.sym_d, p.sym_tau, p.dens_r, conv, im.shape)
    if cache is not None and key in cache:
        rn, th = cache[key]
        return rn.astype(np.float32), th.astype(np.float32), key

    lab = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    chroma = np.hypot(lab[..., 1], lab[..., 2])          # retak = lembah kroma (abu-abu di ubin berwarna)
    best = th = None
    for s in p.sigmas:
        r, t = hessian_valley(chroma, s, use_scratch)
        if best is None:
            best, th = r, t
        else:
            m = r > best
            best, th = np.where(m, r, best), np.where(m, t, th)

    # Gerbang simetri: warna di dua sisi retak harus mirip. Tepi ubin dan batas motif punya
    # warna berbeda di kiri-kanan (tidak simetris) sehingga ditekan.
    M = cv2.GaussianBlur(lab, (0, 0), 3)
    h, w = best.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    nx, ny = np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
    a = cv2.remap(M, xs + p.sym_d * nx, ys + p.sym_d * ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    b = cv2.remap(M, xs - p.sym_d * nx, ys - p.sym_d * ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    asym = np.sqrt(((a - b) ** 2).sum(-1))
    R = best * np.exp(-(asym / p.sym_tau) ** 2)

    # Normalisasi kepadatan lokal: daerah yang penuh garis (serat kayu) dilemahkan.
    dens = cv2.GaussianBlur(best, (0, 0), p.dens_r)
    rn = R / (dens + 0.25 * np.median(dens) + 1e-6)
    if cache is not None:
        cache[key] = (rn.astype(np.float16), th.astype(np.float16))
    return rn, th, key


def parallel_load(wskel: np.ndarray, th: np.ndarray, r: int, nbins: int = 12) -> np.ndarray:
    """Per piksel: banyaknya garis sejajar (orientasi sama +-1 bin) dalam jendela (2r+1)^2,
    dinyatakan dalam 'jumlah garis' (jumlah piksel kerangka dibagi 2r). Serat kayu: >3; retak tunggal: ~1."""
    ang = np.mod(th, np.pi)
    b = np.minimum((ang / np.pi * nbins).astype(int), nbins - 1)
    k = 2 * r + 1
    cnt = np.zeros((nbins,) + wskel.shape, np.float32)
    for i in range(nbins):
        cnt[i] = cv2.boxFilter(((b == i) & wskel).astype(np.float32), -1, (k, k),
                               normalize=False, borderType=cv2.BORDER_CONSTANT)
    tot = cnt + np.roll(cnt, 1, 0) + np.roll(cnt, -1, 0)
    return np.take_along_axis(tot, b[None], 0)[0] / (2.0 * r)


def _close(mask: np.ndarray, k: int, use_scratch: bool) -> np.ndarray:
    if k <= 1:
        return mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if use_scratch and scratch_ops.available()["erosi_dilasi"]:
        return scratch_ops.erode(scratch_ops.dilate(mask, ker), ker)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)


def _endpoints(skel: np.ndarray) -> np.ndarray:
    s = skel.astype(np.uint8)
    nb = cv2.filter2D(s, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - s
    ys, xs = np.nonzero(skel & (nb == 1))
    return np.column_stack([xs, ys])


def bridge_gaps(mask: np.ndarray, max_gap: int, max_angle_deg: float) -> np.ndarray:
    """Sambung ujung dua potongan retak yang berhadapan dan searah (celah <= max_gap piksel).
    Retak yang melewati motif bertekstur sering terputus; penyambungan ini meniru cara orang
    menelusuri retak sebagai satu garis kontinu."""
    if max_gap <= 0:
        return mask
    skel = skeletonize(mask > 0)
    ends = _endpoints(skel)
    if len(ends) < 2:
        return mask
    _, comp = cv2.connectedComponents(mask, connectivity=8)
    sk_pts = np.column_stack(np.nonzero(skel))[:, ::-1]
    sk_comp = comp[sk_pts[:, 1], sk_pts[:, 0]]
    info = []
    for x, y in ends:
        c = comp[y, x]
        d2 = ((sk_pts - (x, y)) ** 2).sum(1)
        sel = (sk_comp == c) & (d2 > 0) & (d2 <= 20 ** 2)
        if sel.sum() < 5:
            continue
        u = np.array([x, y], float) - sk_pts[sel].mean(0)
        nu = np.linalg.norm(u)
        if nu < 3:
            continue
        info.append((int(x), int(y), int(c), u / nu))
    cos_lim = np.cos(np.deg2rad(max_angle_deg))
    pairs = []
    for i in range(len(info)):
        for j in range(i + 1, len(info)):
            xi, yi, ci, ui = info[i]
            xj, yj, cj, uj = info[j]
            if ci == cj:
                continue
            v = np.array([xj - xi, yj - yi], float)
            dist = np.linalg.norm(v)
            if dist < 2 or dist > max_gap:
                continue
            v /= dist
            if ui @ v >= cos_lim and (-uj) @ v >= cos_lim:
                pairs.append((dist, i, j))
    out, used = mask.copy(), set()
    for _, i, j in sorted(pairs):
        if i in used or j in used:
            continue
        used.update((i, j))
        cv2.line(out, info[i][:2], info[j][:2], 255, 3)
    return out


def detect(im: np.ndarray, p, use_scratch: bool = False, cache: "dict | None" = None) -> dict:
    """Deteksi retak pada citra BGR `im` (sudah berukuran kerja). Mengembalikan dict:
    mask (uint8 0/255, hanya retak), hyst (sebelum penyaringan), ridge_vis (uint8 untuk tampilan)."""
    rn, th, rkey = ridge_stage(im, p, use_scratch, cache)
    med = float(np.median(rn))
    mad = float(np.median(np.abs(rn - med))) * 1.4826 + 1e-9
    strong = rn > med + p.hi_k * mad
    weak = rn > med + p.lo_k * mad

    # histeresis: ambil kandidat lemah yang tersambung ke benih kuat
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    hyst = keep[lab].astype(np.uint8) * 255
    hyst = _close(hyst, p.rclose_k, use_scratch)

    # penekan tekstur: buang bagian yang dikelilingi banyak garis sejajar (serat kayu)
    lkey = ("load", rkey, p.lo_k, p.par_r)
    if cache is not None and lkey in cache:
        load = cache[lkey].astype(np.float32)
    else:
        wskel = skeletonize(cv2.morphologyEx(weak.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                                             np.ones((3, 3), np.uint8)) > 0)
        load = parallel_load(wskel, th, p.par_r)
        if cache is not None:
            cache[lkey] = load.astype(np.float16)
    hyst_raw = hyst.copy()
    hyst[load > p.par_lines] = 0

    # filter komponen: cukup panjang dan cukup lurus
    skel = skeletonize(hyst > 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hyst, connectivity=8)
    mask = np.zeros_like(hyst)
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        comp = lab[y:y + h, x:x + w] == i
        sk = skel[y:y + h, x:x + w] & comp
        length = int(sk.sum())
        if length < p.min_len:
            continue
        pts = np.column_stack(np.nonzero(sk))[:, ::-1].astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        if max(rw, rh) / length < p.straight_min:
            continue
        mask[y:y + h, x:x + w][comp] = 255

    mask = bridge_gaps(mask, int(p.bridge_gap), float(p.bridge_angle))
    vis = np.clip(rn / max(float(np.percentile(rn, 99.7)), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return {"mask": mask, "hyst": hyst_raw, "ridge_vis": vis}
