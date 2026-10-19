"""Verifikasi operasi buatan sendiri (scratch_ops.py) terhadap OpenCV.

Untuk >= 10 citra dicatat: waktu eksekusi buatan sendiri vs OpenCV, selisih piksel
maksimum, selisih rata-rata, dan banyaknya piksel yang berbeda.
Keluaran: <out>/verifikasi_scratch.csv (per citra) dan <out>/verifikasi_ringkasan.csv.

Bisa dijalankan sendiri (dari folder yang berisi run_all.py dan tile_defect/):
    python verify_scratch.py --data dataset
atau otomatis sebagai langkah [5] di run_all.py.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from tile_defect import crack, dataset, pipeline, scratch_ops

TOL_FLOAT = 1e-3    # selisih konvolusi (float) di bawah ini dianggap sama


def _waktu_ms(fungsi, *args):
    t0 = time.perf_counter()
    hasil = fungsi(*args)
    return hasil, (time.perf_counter() - t0) * 1000.0


def _pilih_citra(samples, n):
    """Ambil n citra bergiliran dari tiap kondisi supaya semua kondisi terwakili."""
    per_kondisi = {}
    for s in samples:
        per_kondisi.setdefault(s.condition, []).append(s)
    antrean = [list(v) for _, v in sorted(per_kondisi.items())]
    terpilih = []
    while len(terpilih) < n and any(antrean):
        for a in antrean:
            if a and len(terpilih) < n:
                terpilih.append(a.pop(0))
    return terpilih


def verify(samples, out, n=10):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pilih = _pilih_citra(samples, n)
    if len(pilih) < n:
        print(f"  PERINGATAN: hanya {len(pilih)} citra tersedia (syarat UTS: minimal 10).")

    p = pipeline.Params()
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernels = {"konvolusi Kxx": crack._K_XX, "konvolusi Kyy": crack._K_YY,
               "konvolusi Kxy": crack._K_XY}
    rows = []

    for s in pilih:
        img, _ = dataset.read_image_scaled(s.image_path, p.size)
        _, gray = pipeline.preprocess(img, p)
        gf = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)     # masukan konvolusi
        res8, _ = pipeline.enhance(gray, p)
        _, biner = cv2.threshold(res8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # masukan morfologi

        # --- konvolusi ---
        for nama, k in kernels.items():
            k = k.astype(np.float32)
            a, t_a = _waktu_ms(scratch_ops.convolve2d, gf, k)
            # filter2D itu korelasi, jadi kernel dibalik agar setara konvolusi
            kb = np.ascontiguousarray(k[::-1, ::-1])
            b, t_b = _waktu_ms(lambda: cv2.filter2D(gf, -1, kb, borderType=cv2.BORDER_REPLICATE))
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            rows.append(dict(citra=s.name, kondisi=s.condition, operasi=nama,
                             ms_buatan=t_a, ms_opencv=t_b, selisih_maks=float(d.max()),
                             selisih_rata2=float(d.mean()), piksel_beda=int((d > TOL_FLOAT).sum()),
                             total_piksel=d.size))

        # --- erosi, dilasi, closing ---
        tes = {
            "erosi 7x7": (lambda: scratch_ops.erode(biner, ker), lambda: cv2.erode(biner, ker)),
            "dilasi 7x7": (lambda: scratch_ops.dilate(biner, ker), lambda: cv2.dilate(biner, ker)),
            "closing 7x7": (lambda: scratch_ops.erode(scratch_ops.dilate(biner, ker), ker),
                            lambda: cv2.morphologyEx(biner, cv2.MORPH_CLOSE, ker)),
        }
        for nama, (f_buatan, f_cv) in tes.items():
            a, t_a = _waktu_ms(f_buatan)
            b, t_b = _waktu_ms(f_cv)
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            rows.append(dict(citra=s.name, kondisi=s.condition, operasi=nama,
                             ms_buatan=t_a, ms_opencv=t_b, selisih_maks=float(d.max()),
                             selisih_rata2=float(d.mean()), piksel_beda=int((d > 0).sum()),
                             total_piksel=d.size))

    fields = ["citra", "kondisi", "operasi", "ms_buatan", "ms_opencv", "selisih_maks",
              "selisih_rata2", "piksel_beda", "total_piksel"]
    with open(out / "verifikasi_scratch.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    # --- ringkasan per operasi ---
    ringkas = []
    print(f"  {len(pilih)} citra | {'operasi':14s} {'ms buatan':>10s} {'ms OpenCV':>10s} "
          f"{'selisih maks':>13s} {'piksel beda':>12s}")
    for nama in dict.fromkeys(r["operasi"] for r in rows):
        rr = [r for r in rows if r["operasi"] == nama]
        item = dict(operasi=nama, n_citra=len(rr),
                    ms_buatan=round(float(np.mean([r["ms_buatan"] for r in rr])), 2),
                    ms_opencv=round(float(np.mean([r["ms_opencv"] for r in rr])), 3),
                    selisih_maks=max(r["selisih_maks"] for r in rr),
                    selisih_rata2=round(float(np.mean([r["selisih_rata2"] for r in rr])), 8),
                    piksel_beda_total=sum(r["piksel_beda"] for r in rr),
                    persen_piksel_beda=round(100.0 * sum(r["piksel_beda"] for r in rr)
                                             / sum(r["total_piksel"] for r in rr), 6))
        ringkas.append(item)
        print(f"  {'':{len(str(len(pilih))) + 8}s}{nama:14s} {item['ms_buatan']:10.2f} {item['ms_opencv']:10.3f} "
              f"{item['selisih_maks']:13.6f} {item['piksel_beda_total']:12d}")
    with open(out / "verifikasi_ringkasan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ringkas[0]))
        w.writeheader()
        w.writerows(ringkas)
    print(f"  tersimpan: {out / 'verifikasi_scratch.csv'}")
    return len(pilih) >= 1


def main():
    ap = argparse.ArgumentParser(description="Verifikasi operasi buatan sendiri terhadap OpenCV")
    ap.add_argument("--data", required=True, help="folder dataset (berisi images/ dan masks/)")
    ap.add_argument("--out", default="results", help="folder keluaran (default: results)")
    ap.add_argument("--n", type=int, default=10, help="jumlah citra yang diverifikasi (default 10)")
    args = ap.parse_args()

    aktif = scratch_ops.active_ops(True)
    print("Operasi buatan sendiri yang aktif:", ", ".join(aktif) if aktif else "(belum ada)")
    if not aktif:
        raise SystemExit("Belum ada operasi di scratch_ops.py yang aktif/lolos uji.")
    samples = dataset.load_samples(args.data)
    print(f"Memverifikasi {min(args.n, len(samples))} dari {len(samples)} citra...")
    verify(samples, args.out, args.n)


if __name__ == "__main__":
    main()
