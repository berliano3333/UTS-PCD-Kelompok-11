"""Operasi pengolahan citra BUATAN SENDIRI (hanya NumPy, tanpa OpenCV / scikit-image).

Operasi yang dikerjakan:
  1. konvolusi 2D        -> convolve2d(citra, kernel)
  2. erosi dan dilasi    -> erode(citra, ker), dilate(citra, ker)

Catatan:
- Cara kerja: untuk setiap elemen kernel kita geser seluruh citra sekali, lalu
  hasilnya dijumlahkan (konvolusi) atau diambil min/max-nya (erosi/dilasi).
  Hasilnya sama dengan loop per piksel, tetapi jauh lebih cepat karena
  dikerjakan NumPy sekaligus untuk seluruh citra.
- Dipakai oleh crack.py dan pipeline.py bila program dijalankan dengan --scratch.

CATATAN DESAIN (untuk laporan):
- convolve2d membalik kernel (konvolusi sejati), sedangkan jalur OpenCV di crack.py
  memakai cv2.filter2D yang berupa korelasi (kernel tidak dibalik). Ketiga kernel Hessian
  (_K_XX, _K_YY, _K_XY) simetris terhadap putaran 180 derajat, jadi hasil kedua jalur sama.
  Bila suatu saat dipakai kernel yang tidak simetris (misalnya Sobel), kernelnya harus
  dibalik dulu pada salah satu jalur, kalau tidak hasilnya akan berbeda.
- available() menjalankan uji mandiri kecil (self-test) pada saat pertama dipanggil. Bila
  sebuah operasi gagal uji, operasi itu dimatikan otomatis dan sebuah peringatan dicetak.
"""
import numpy as np


# ---------------------------------------------------------------
# 1. Konvolusi 2D
# ---------------------------------------------------------------
def convolve2d(citra, kernel):
    """Konvolusi 2D citra grayscale dengan kernel.

    - Tepi citra diisi dengan menyalin piksel tepi (replicate).
    - Kernel dibalik dulu (konvolusi sejati). cv2.filter2D sebenarnya
      korelasi (kernel tidak dibalik), jadi untuk membandingkan dengan OpenCV
      kernelnya harus dibalik juga (lihat verify_scratch.py).
    - Ukuran keluaran sama dengan ukuran masukan, tipe float32.
    """
    citra = np.asarray(citra, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    if citra.ndim != 2 or k.ndim != 2:
        raise ValueError("convolve2d hanya untuk citra 2D (grayscale) dan kernel 2D")

    k = k[::-1, ::-1]                 # balik kernel (flip vertikal + horizontal)
    kh, kw = k.shape
    h, w = citra.shape
    ah, aw = kh // 2, kw // 2         # titik pusat kernel

    # tambah bingkai di sekeliling citra supaya piksel tepi tetap punya tetangga
    padded = np.pad(citra, ((ah, kh - 1 - ah), (aw, kw - 1 - aw)), mode="edge")

    hasil = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            if k[i, j] == 0:          # elemen nol tidak perlu dihitung
                continue
            hasil += k[i, j] * padded[i:i + h, j:j + w]
    return hasil


# ---------------------------------------------------------------
# 2. Erosi dan dilasi
# ---------------------------------------------------------------
def _morfologi(citra, ker, dilasi):
    """Fungsi dasar erosi/dilasi. dilasi=True -> ambil MAKS, dilasi=False -> ambil MIN.
    Hanya elemen kernel yang bernilai > 0 yang dipakai."""
    citra = np.asarray(citra, dtype=np.uint8)
    ker = np.asarray(ker) > 0
    kh, kw = ker.shape
    h, w = citra.shape
    ah, aw = kh // 2, kw // 2

    # Nilai di luar citra: dilasi pakai 0 (tidak menambah putih dari luar),
    # erosi pakai 255 (tepi citra tidak ikut terkikis).
    tepi = 0 if dilasi else 255
    padded = np.pad(citra, ((ah, kh - 1 - ah), (aw, kw - 1 - aw)),
                    mode="constant", constant_values=tepi)

    hasil = np.zeros((h, w), np.uint8) if dilasi else np.full((h, w), 255, np.uint8)
    for i in range(kh):
        for j in range(kw):
            if not ker[i, j]:
                continue
            geser = padded[i:i + h, j:j + w]
            hasil = np.maximum(hasil, geser) if dilasi else np.minimum(hasil, geser)
    return hasil


def dilate(citra, ker):
    """Dilasi: objek putih (255) membesar. Untuk menyambung garis yang putus."""
    return _morfologi(citra, ker, dilasi=True)


def erode(citra, ker):
    """Erosi: objek putih (255) mengecil. Untuk membuang bintik kecil."""
    return _morfologi(citra, ker, dilasi=False)


# ---------------------------------------------------------------
# Daftar operasi yang sudah selesai (dibaca oleh pipeline & run_all)
# ---------------------------------------------------------------
def _uji_konvolusi():
    img = np.arange(25, dtype=np.float32).reshape(5, 5)
    identitas = np.zeros((3, 3), np.float32)
    identitas[1, 1] = 1
    geser = np.zeros((3, 3), np.float32)
    geser[1, 2] = 1            # konvolusi sejati: hasil(y, x) = citra(y, x-1)
    return (np.array_equal(convolve2d(img, identitas), img)
            and convolve2d(img, geser)[2, 2] == img[2, 1]
            and convolve2d(img, geser)[2, 0] == img[2, 0])      # tepi: salinan piksel tepi


def _uji_morfologi():
    img = np.zeros((7, 7), np.uint8)
    img[3, 3] = 255
    ker = np.ones((3, 3), np.uint8)
    besar = dilate(img, ker)                       # titik -> blok 3x3
    kecil = erode(besar, ker)                      # blok 3x3 -> titik semula
    return (besar.sum() == 9 * 255 and besar[2:5, 2:5].all()
            and np.array_equal(kecil, img))


# Saklar manual: ubah ke False untuk mematikan satu operasi.
_AKTIF = {"konvolusi": True, "erosi_dilasi": True, "otsu": False}
_UJI = {"konvolusi": _uji_konvolusi, "erosi_dilasi": _uji_morfologi}
_hasil_uji = {}


def available():
    """True = operasi sudah diisi DAN lolos uji mandiri."""
    hasil = {}
    for nama, aktif in _AKTIF.items():
        ok = aktif
        if aktif and nama in _UJI:
            if nama not in _hasil_uji:
                try:
                    _hasil_uji[nama] = bool(_UJI[nama]())
                except Exception as e:
                    print(f"PERINGATAN scratch_ops: uji '{nama}' error: {e}")
                    _hasil_uji[nama] = False
                if not _hasil_uji[nama]:
                    print(f"PERINGATAN scratch_ops: operasi '{nama}' GAGAL uji mandiri "
                          f"dan dinonaktifkan. Periksa kodenya.")
            ok = _hasil_uji[nama]
        hasil[nama] = ok
    return hasil


def active_ops(use_scratch=True):
    if not use_scratch:
        return []
    return [nama for nama, ada in available().items() if ada]
