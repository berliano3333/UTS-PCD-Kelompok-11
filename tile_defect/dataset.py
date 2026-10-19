"""Pemuat dataset dari folder mana pun (path diberikan lewat --data).

Struktur folder yang didukung (pilih salah satu):

A) Per kondisi dengan subfolder
   <data>/images/<kondisi>/<nama>.jpg
   <data>/masks/<kondisi>/<nama>.png        (ground truth, putih = cacat)

B) Flat dengan awalan kondisi pada nama berkas
   <data>/images/<kondisi>_<nama>.jpg
   <data>/masks/<kondisi>_<nama>.png

Nama mask harus sama dengan citranya (ekstensi boleh beda). Citra tanpa mask
tetap diproses, tetapi tidak ikut dihitung metriknya.

GRUP UBIN (penting untuk pembagian dev/test)
  Foto yang diambil dari ubin/retak yang SAMA hampir kembar. Bila sebagian jatuh
  ke dev dan sebagian ke test, angka test menjadi terlalu bagus (kebocoran data).
  Beri awalan grup dengan dua garis bawah agar semua foto satu ubin masuk ke
  bagian yang sama:   ubinA__foto01.jpg, ubinA__foto02.jpg, ubinB__foto01.jpg ...
  Tanpa "__", tiap berkas dianggap grup sendiri.

Opsional: <data>/split.csv  (kolom: file,split  dengan split = dev|test)
Jika tidak ada, pembagian ~60% dev / 40% test dibuat deterministik (urutan hash nama grup,
hasil sama di semua komputer) dan DISEIMBANGKAN PER KONDISI, sehingga setiap kondisi punya
citra dev dan citra test (perlu agar tabel "per kondisi" terisi untuk semua kondisi).
"""
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
HEIF_EXT = {".heic", ".heif"}


@dataclass
class Sample:
    name: str          # path relatif tanpa ekstensi, jadi id unik
    condition: str
    group: str
    split: str         # "dev" | "test"
    image_path: Path
    mask_path: "Path | None"


def mask_path_for(data_dir, image_path) -> Path:
    """Lokasi mask (PNG) untuk sebuah citra: <data>/masks/<path relatif sama>.png"""
    root = Path(data_dir).expanduser().resolve()
    rel = Path(image_path).resolve().relative_to(root / "images")
    return (root / "masks" / rel).with_suffix(".png")


def load_samples(data_dir: str) -> list:
    root = Path(data_dir).expanduser().resolve()
    img_dir, mask_dir = root / "images", root / "masks"
    if not img_dir.is_dir():
        raise SystemExit(
            f"Folder '{img_dir}' tidak ditemukan. Taruh citra di <data>/images/ "
            f"(lihat README untuk struktur folder)."
        )

    masks = {}
    if mask_dir.is_dir():
        for p in mask_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                masks[p.relative_to(mask_dir).with_suffix("").as_posix()] = p

    forced = {}
    split_csv = root / "split.csv"
    if split_csv.exists():
        with open(split_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                forced[Path(row["file"]).with_suffix("").as_posix()] = row["split"].strip()

    samples = []
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        rel = p.relative_to(img_dir)
        key = rel.with_suffix("").as_posix()
        stem = rel.stem
        if "__" in stem:
            # grup = ubin yang sama, walau difoto pada kondisi (subfolder) berbeda
            group, base = stem.split("__", 1)
        else:
            group, base = key, stem
        if len(rel.parts) > 1:
            condition = rel.parts[0]
        elif "_" in base:
            condition = base.split("_")[0]
        else:
            condition = "tanpa_kondisi"
        samples.append(Sample(key, condition, group, "", p, masks.get(key)))
    if not samples:
        raise SystemExit(f"Tidak ada citra di {img_dir}")
    _assign_splits(samples, forced)
    return samples


def _assign_splits(samples: list, forced: dict, dev_percent: int = 60) -> None:
    """Isi Sample.split. Semua foto satu grup selalu masuk bagian yang sama. Grup diurutkan
    menurut hash namanya (acak tapi tetap), lalu tiap grup dimasukkan ke dev bila porsi dev
    pada kondisi utamanya masih di bawah dev_percent, jika tidak ke test."""
    by_group = {}
    for s in samples:
        by_group.setdefault(s.group, []).append(s)
    n_cond = {}
    for s in samples:
        n_cond[s.condition] = n_cond.get(s.condition, 0) + 1
    seen = {c: [0, 0] for c in n_cond}                  # kondisi -> [jumlah dev, jumlah total]
    order = sorted(by_group, key=lambda g: hashlib.md5(g.encode("utf-8")).hexdigest())
    for g in order:
        members = by_group[g]
        fixed = next((forced[k] for k in (members[0].name, g) if k in forced), None)
        if fixed is None:
            main = max({m.condition for m in members}, key=lambda c: sum(m.condition == c for m in members))
            dev_n, tot = seen[main]
            # target: dev menempati dev_percent dari SELURUH citra kondisi itu
            fixed = "dev" if dev_n < round(n_cond[main] * dev_percent / 100) else "test"
        for m in members:
            m.split = forced.get(m.name, fixed)
            seen[m.condition][1] += 1
            seen[m.condition][0] += m.split == "dev"


def _read_heif(path: Path) -> np.ndarray:
    """Baca HEIC/HEIF (foto iPhone) dan kembalikan array BGR. Orientasi EXIF diterapkan,
    sama seperti cv2.imread pada JPG, sehingga mask yang digambar dari foto yang sama cocok."""
    try:
        from PIL import Image, ImageOps
        import pillow_heif
    except ImportError:
        raise SystemExit("Berkas HEIC terdeteksi. Pasang dulu:  pip install pillow pillow-heif")
    pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in HEIF_EXT:
        return _read_heif(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Gagal membaca citra: {path}")
    return img


def read_image_scaled(path: Path, size: int):
    """Baca citra lalu perkecil agar sisi terpanjang = `size`. Mengembalikan (citra, (H, W) asli).
    Dipakai agar cache memori tidak menyimpan foto resolusi penuh."""
    img = read_image(path)
    h, w = img.shape[:2]
    s = size / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return img, (h, w)


def read_mask(path: Path, size_hw) -> np.ndarray:
    """Baca mask ground truth dan sesuaikan ke ukuran kerja. Garis tipis dipertahankan
    (INTER_AREA lalu ambang rendah) agar tidak hilang saat diperkecil."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Gagal membaca mask: {path}")
    h, w = size_hw
    if abs(m.shape[1] / m.shape[0] - w / h) > 0.02:
        raise ValueError(
            f"Rasio mask {path.name} ({m.shape[1]}x{m.shape[0]}) tidak cocok dengan fotonya "
            f"({w}x{h} pada ukuran kerja). Mask harus digambar pada foto dengan orientasi yang sama "
            f"(foto yang ditampilkan galeri), bukan foto yang diputar."
        )
    m = (m > 127).astype(np.uint8) * 255
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    return (m > 63).astype(np.uint8)
