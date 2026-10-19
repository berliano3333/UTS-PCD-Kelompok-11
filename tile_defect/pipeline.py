"""Pipeline klasik deteksi cacat ubin (tanpa deep learning).

Tahap berurutan:
  1. Pra-pemrosesan    : ubah ukuran, grayscale, Gaussian blur (reduksi derau)
  2. Perbaikan kontras : metode lama: koreksi iluminasi + CLAHE.
                         metode `ridge`: peta ridge kroma + gerbang simetri + normalisasi kepadatan
  3. Segmentasi        : salah satu METHODS (ridge | blackhat | otsu | adaptive | sigma | canny)
  4. Pasca-proses      : morfologi + penyaringan komponen (lihat crack.py untuk `ridge`)
  5. Pengukuran        : lokasi (kotak), panjang, lebar rata-rata, dan luas tiap cacat

Metode `ridge` dibuat khusus untuk RETAK pada ubin bermotif. Metode lain dipertahankan sebagai
pembanding (syarat UTS: bandingkan minimal dua metode berdasarkan angka).
"""
import ast
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import crack, scratch_ops


@dataclass
class Params:
    method: str = "ridge"
    size: int = 1280         # sisi terpanjang setelah resize (retak tipis butuh resolusi cukup)
    # --- metode lama ---
    blur_k: int = 5          # kernel Gaussian (ganjil)
    bg_k: int = 51           # kernel median untuk estimasi latar/iluminasi (ganjil)
    clahe_clip: float = 2.0  # 0 = tanpa CLAHE
    k_sigma: float = 3.0     # ambang metode sigma: median + k * 1.4826 * MAD
    adapt_block: int = 31
    adapt_c: float = 6.0
    canny_lo: int = 40
    canny_hi: int = 110
    bh_k: int = 21           # elemen penstruktur black-hat (ganjil)
    bh_thr: float = 6.0      # ambang black-hat: median + k * MAD
    open_k: int = 3          # elips untuk opening (0 = nonaktif)
    close_k: int = 7         # elips untuk closing (0 = nonaktif)
    min_area: int = 40       # komponen lebih kecil dari ini dibuang (piksel)
    grid_len: int = 0        # buang garis lurus panjang (nat) setelah segmentasi (0 = nonaktif)
    # --- metode ridge (retak) ---
    sigmas: tuple = (1.5, 2.5)   # skala garis (lebar retak sekitar 2*sigma piksel)
    sym_d: int = 7           # jarak sampel di kiri-kanan garis untuk uji simetri warna
    sym_tau: float = 10.0    # toleransi beda warna kiri-kanan (satuan Lab); lebih besar = lebih longgar
    dens_r: int = 25         # radius normalisasi kepadatan ridge lokal
    hi_k: float = 6.0        # ambang benih kuat: median + hi_k * MAD
    lo_k: float = 3.0        # ambang kandidat lemah
    rclose_k: int = 5        # closing penyambung retak putus-putus
    min_len: int = 90        # panjang kerangka minimal sebuah retak (piksel kerja)
    straight_min: float = 0.55   # panjang kotak/panjang kerangka minimal (membuang coretan berliku)
    par_r: int = 30          # setengah ukuran jendela penghitung garis sejajar
    par_lines: float = 2.1   # lebih dari ini garis sejajar dalam jendela = tekstur (serat kayu)
    bridge_gap: int = 110    # sambung potongan retak berhadapan bila celah <= nilai ini (0 = nonaktif)
    bridge_angle: float = 25.0   # toleransi sudut penyambungan (derajat)

    def as_dict(self):
        return asdict(self)

    def to_file(self, path):
        Path(path).write_text("\n".join(f"{k} = {v!r}" for k, v in self.as_dict().items()), encoding="utf-8")

    @classmethod
    def from_file(cls, path):
        """Baca parameter_final.txt (baris `nama = nilai`); nama tak dikenal diabaikan."""
        known = {f.name for f in fields(cls)}
        vals = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            k, v = (s.strip() for s in line.split("=", 1))
            if k in known:
                vals[k] = ast.literal_eval(v)
        return cls(**vals)


def _odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# ---------- Tahap 1: pra-pemrosesan ----------
def preprocess(bgr: np.ndarray, p: Params):
    h, w = bgr.shape[:2]
    s = min(1.0, p.size / max(h, w))
    img = bgr if s == 1.0 else cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))),
                                          interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if p.blur_k > 1:
        gray = cv2.GaussianBlur(gray, (_odd(p.blur_k),) * 2, 0)
    return img, gray


# ---------- Tahap 2: perbaikan kontras (metode lama) ----------
def enhance(gray: np.ndarray, p: Params):
    """Mengembalikan (residual_uint8, residual_float). Latar (pola ubin + gradasi cahaya)
    diperkirakan dengan median besar lalu dikurangkan."""
    bg = cv2.medianBlur(gray, _odd(p.bg_k))
    signed = gray.astype(np.float32) - bg.astype(np.float32)
    mag = np.abs(signed)
    hi = max(float(np.percentile(mag, 99.5)), 1.0)
    res8 = np.clip(mag / hi * 255.0, 0, 255).astype(np.uint8)
    if p.clahe_clip > 0:
        res8 = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(8, 8)).apply(res8)
    return res8, mag


# ---------- Tahap 3: segmentasi metode lama (pembanding) ----------
def seg_otsu(res8, mag, gray, p, use_scratch=False):
    if use_scratch and scratch_ops.available()["otsu"]:
        t = scratch_ops.otsu_threshold(res8)
        return ((res8 > t) * 255).astype(np.uint8)
    _, b = cv2.threshold(res8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def seg_adaptive(res8, mag, gray, p, use_scratch=False):
    a = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, _odd(p.adapt_block), p.adapt_c)
    b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, _odd(p.adapt_block), -p.adapt_c)
    return cv2.bitwise_or(a, b)  # cacat bisa lebih gelap atau lebih terang


def seg_sigma(res8, mag, gray, p, use_scratch=False):
    med = float(np.median(mag))
    mad = float(np.median(np.abs(mag - med))) * 1.4826 + 1e-6
    return ((mag > med + p.k_sigma * mad) * 255).astype(np.uint8)


def seg_canny(res8, mag, gray, p, use_scratch=False):
    e = cv2.Canny(res8, p.canny_lo, p.canny_hi)
    e = cv2.dilate(e, _ellipse(3))
    return cv2.morphologyEx(e, cv2.MORPH_CLOSE, _ellipse(5))


def seg_blackhat(res8, mag, gray, p, use_scratch=False):
    """Pembanding klasik untuk garis tipis gelap: black-hat lalu ambang median + k*MAD."""
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _ellipse(_odd(p.bh_k))).astype(np.float32)
    med = float(np.median(bh))
    mad = float(np.median(np.abs(bh - med))) * 1.4826 + 1e-6
    return ((bh > med + p.bh_thr * mad) * 255).astype(np.uint8)


METHODS = {"ridge": None, "blackhat": seg_blackhat, "otsu": seg_otsu, "adaptive": seg_adaptive,
           "sigma": seg_sigma, "canny": seg_canny}


# ---------- Tahap 4: pasca-proses morfologi (metode lama) ----------
def suppress_grid(binary: np.ndarray, length: int) -> np.ndarray:
    """Buang garis horizontal/vertikal panjang (nat antar-ubin)."""
    h_line = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    v_line = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    lines = cv2.dilate(cv2.bitwise_or(h_line, v_line), _ellipse(5))
    return cv2.bitwise_and(binary, cv2.bitwise_not(lines))


def postprocess(binary: np.ndarray, p: Params, use_scratch=False) -> np.ndarray:
    out = suppress_grid(binary, int(p.grid_len)) if p.grid_len > 1 else binary
    scratch_morph = use_scratch and scratch_ops.available()["erosi_dilasi"]
    if p.open_k > 1:
        k = _ellipse(_odd(p.open_k))
        out = (scratch_ops.dilate(scratch_ops.erode(out, k), k) if scratch_morph
               else cv2.morphologyEx(out, cv2.MORPH_OPEN, k))
    if p.close_k > 1:
        k = _ellipse(_odd(p.close_k))
        out = (scratch_ops.erode(scratch_ops.dilate(out, k), k) if scratch_morph
               else cv2.morphologyEx(out, cv2.MORPH_CLOSE, k))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= p.min_area
    return keep[lab].astype(np.uint8) * 255


# ---------- Tahap 5: pengukuran ----------
def measure(binary: np.ndarray) -> list:
    """Per komponen: kotak, pusat, luas (piksel), panjang kerangka (piksel), lebar rata-rata (piksel)."""
    m = (binary > 0).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return []
    skel = skeletonize(m > 0)
    lengths = np.bincount(lab[skel], minlength=n)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        ln = int(lengths[i])
        out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area_px": int(a),
                    "length_px": ln, "mean_width_px": round(float(a) / max(ln, 1), 2),
                    "cx": float(cent[i][0]), "cy": float(cent[i][1])})
    return sorted(out, key=lambda d: -d["length_px"])


def run(bgr: np.ndarray, p: Params, use_scratch: bool = False, cache: "dict | None" = None) -> dict:
    img, gray = preprocess(bgr, p)
    if p.method == "ridge":
        r = crack.detect(img, p, use_scratch, cache)
        res8, raw, mask = r["ridge_vis"], r["hyst"], r["mask"]
    else:
        res8, mag = enhance(gray, p)
        raw = METHODS[p.method](res8, mag, gray, p, use_scratch)
        mask = postprocess(raw, p, use_scratch)
    defects = measure(mask)
    return {"image": img, "gray": gray, "residual": res8, "raw": raw, "mask": mask,
            "defects": defects, "defect_area_px": int((mask > 0).sum()),
            "defect_ratio": float((mask > 0).mean())}


def _thick(m: np.ndarray, r: int = 1) -> np.ndarray:
    """Tebalkan garis tipis agar terlihat jelas di gambar (hanya untuk tampilan, bukan metrik)."""
    return cv2.dilate((m > 0).astype(np.uint8), _ellipse(2 * r + 1)) > 0


def highlight(img: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Sorot lokasi retak menurut MASK ground truth: piksel mask diwarnai kuning-hijau terang
    dan diberi kontur tipis."""
    vis = img.copy()
    m = _thick(gt, 1)
    vis[m] = (0.25 * vis[m] + 0.75 * np.array((0, 255, 255), np.float32)).astype(np.uint8)
    return vis


def overlay(img: np.ndarray, mask: np.ndarray, gt: "np.ndarray | None" = None) -> np.ndarray:
    """Perbandingan deteksi dengan mask.
       kuning = benar (deteksi tepat di mask)   hijau = retak di mask yang terlewat (FN)
       merah  = deteksi di luar mask (FP)
    Tanpa gt, deteksi saja digambar merah."""
    vis = img.copy()
    pred = mask > 0
    if gt is None:
        vis[_thick(pred)] = (0, 0, 255)
        return vis
    g = gt > 0
    tp, fn, fp = pred & g, g & ~pred, pred & ~g
    for m, col in ((fp, (0, 0, 255)), (fn, (0, 255, 0)), (tp, (0, 255, 255))):
        vis[_thick(m)] = col
    for i, (txt, col) in enumerate((("kuning: deteksi cocok mask (TP)", (0, 255, 255)),
                                    ("hijau: retak di mask, terlewat (FN)", (0, 255, 0)),
                                    ("merah: deteksi di luar mask (FP)", (0, 0, 255)))):
        y = 28 + 26 * i
        cv2.putText(vis, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return vis
