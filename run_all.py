#!/usr/bin/env python3
"""Satu perintah untuk menjalankan seluruh pipeline dan mencetak tabel metrik.

    python run_all.py --data /path/ke/folder_dataset

Folder --data hanya perlu berisi  images/<kondisi>/...  dan  masks/<kondisi>/...

Langkah:
  1. Muat dataset (images/ dan masks/), bagi ~60% dev / ~40% test per kondisi.
  2. Bandingkan metode segmentasi pada citra DEV -> pilih terbaik (metrik --select).
  3. Setel parameter metode terpilih dengan grid kecil pada citra DEV saja.
  4. Laporkan angka akhir HANYA dari citra TEST (keseluruhan dan per kondisi).
  5. Simpan highlight retak menurut mask, overlay deteksi-vs-mask, tabel CSV,
     lokasi/luas retak, grafik, dan >= 3 kasus gagal terburuk.
  6. Bila scratch_ops.py sudah diisi: verifikasi terhadap OpenCV pada 10 citra.
Semua keluaran ada di folder --out (default: results/).
"""
import argparse
import csv
import itertools
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from tile_defect import dataset, metrics, pipeline, scratch_ops
from verify_scratch import verify

GRIDS = {
    "ridge": {"hi_k": [5.0, 6.0, 8.0], "par_lines": [1.7, 2.1, 2.5], "min_len": [70, 100, 140]},
    "blackhat": {"bh_k": [15, 21, 31], "bh_thr": [4.0, 6.0, 8.0], "min_area": [20, 40, 100]},
    "otsu": {"bg_k": [31, 51, 81], "close_k": [5, 7, 11], "min_area": [20, 40, 100], "grid_len": [0, 81, 161]},
    "sigma": {"bg_k": [31, 51, 81], "k_sigma": [2.5, 3.0, 4.0], "min_area": [20, 40, 100], "grid_len": [0, 81, 161]},
    "adaptive": {"bg_k": [31, 51, 81], "adapt_c": [4.0, 6.0, 8.0], "min_area": [20, 40, 100]},
    "canny": {"bg_k": [31, 51, 81], "canny_hi": [90, 110, 150], "min_area": [20, 40, 100]},
}


def item(s, p, cache):
    c = cache.setdefault(s.name, {})
    if "img" not in c:
        c["img"], c["orig_hw"] = dataset.read_image_scaled(s.image_path, p.size)
    return c


def evaluate(samples, p, use_scratch, cache, tol):
    rows = []
    for s in samples:
        if s.mask_path is None:
            continue
        c = item(s, p, cache)
        res = pipeline.run(c["img"], p, use_scratch, cache=c)
        if "gt" not in c:
            c["gt"] = dataset.read_mask(s.mask_path, res["mask"].shape)
        sc = metrics.scores((res["mask"] > 0).astype(np.uint8), c["gt"], tol)
        sc.update(name=s.name, condition=s.condition)
        rows.append(sc)
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(d):
    return (f"IoU={d['iou']:.3f} Dice={d['dice']:.3f} Prec={d['precision']:.3f} Rec={d['recall']:.3f} | "
            f"toleransi: P={d['precision_tol']:.3f} R={d['recall_tol']:.3f} F1={d['f1_tol']:.3f} (n={d['n']})")


METRIC_FIELDS = ["iou", "dice", "precision", "recall", "precision_tol", "recall_tol", "f1_tol", "micro_iou", "n"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="folder dataset (berisi images/ dan masks/)")
    ap.add_argument("--out", default="results", help="folder keluaran")
    ap.add_argument("--methods", default=",".join(pipeline.METHODS),
                    help="metode yang dibandingkan, dipisah koma (default: semua)")
    ap.add_argument("--select", choices=["dice", "f1_tol"], default="dice",
                    help="metrik pemilih metode/parameter pada data DEV (default: dice)")
    ap.add_argument("--tol", type=int, default=metrics.TOL_PX, help="toleransi piksel untuk metrik *_tol")
    ap.add_argument("--scratch", action="store_true",
                    help="pakai operasi buatan sendiri (scratch_ops.py) yang sudah diisi")
    ap.add_argument("--no-tune", action="store_true", help="lewati penyetelan parameter (lebih cepat)")
    ap.add_argument("--size", type=int, default=None, help="sisi terpanjang citra kerja (default 1280)")
    args = ap.parse_args()

    if args.scratch:
        act = scratch_ops.active_ops(True)
        print("Operasi buatan sendiri yang aktif:", ", ".join(act) if act else "(belum ada yang diisi)")
        if not act:
            raise SystemExit("scratch_ops.py belum diisi. Tulis minimal dua operasi dulu, "
                             "atau jalankan tanpa --scratch.")

    out = Path(args.out)
    for d in ("overlay", "highlight", "failures"):
        (out / d).mkdir(parents=True, exist_ok=True)

    samples = dataset.load_samples(args.data)
    dev = [s for s in samples if s.split == "dev" and s.mask_path]
    test = [s for s in samples if s.split == "test" and s.mask_path]
    n_nomask = sum(1 for s in samples if s.mask_path is None)
    print(f"Total citra: {len(samples)} | dev bermask: {len(dev)} | test bermask: {len(test)} "
          f"| tanpa mask: {n_nomask}")
    g_dev, g_test = {s.group for s in dev}, {s.group for s in test}
    print(f"Grup ubin: dev={len(g_dev)} test={len(g_test)} (foto satu grup tidak pernah terbagi ke dev dan test)")
    if g_dev & g_test:
        print("PERINGATAN: grup yang sama ada di dev dan test (dari split.csv?). Angka test bisa terlalu bagus.")
    conds = {}
    for s in samples:
        conds.setdefault(s.condition, []).append(s)
    print("Kondisi: " + ", ".join(
        f"{c}={len(v)} (dev {sum(x.split == 'dev' for x in v)}/test {sum(x.split == 'test' for x in v)})"
        for c, v in sorted(conds.items())))
    if len(samples) < 50:
        print("PERINGATAN: syarat UTS minimal 50 citra (30 di antaranya dipotret sendiri).")
    if len(conds) < 3:
        print("PERINGATAN: syarat UTS minimal 3 kondisi pengambilan citra (subfolder di images/).")
    if len(dev) + len(test) < 15:
        print("PERINGATAN: syarat UTS minimal 15 citra bermask.")
    t_nomask = [s.name for s in samples if s.split == "test" and s.mask_path is None]
    if t_nomask:
        print(f"PERINGATAN: {len(t_nomask)} citra test tidak punya mask (UTS: ground truth untuk seluruh citra uji): "
              + ", ".join(t_nomask[:5]) + (" ..." if len(t_nomask) > 5 else ""))
    if not dev or not test:
        raise SystemExit("Butuh citra bermask di dev DAN test. Tambah data/mask, beri awalan grup "
                         "(ubinA__foto.jpg) pada lebih dari satu ubin, atau isi split.csv.")

    cache = {}
    base = pipeline.Params() if args.size is None else pipeline.Params(size=args.size)
    methods = [m for m in args.methods.split(",") if m in pipeline.METHODS]

    # --- 1) perbandingan metode pada DEV ---
    print(f"\n[1] Perbandingan metode segmentasi (DEV, dipilih menurut {args.select})")
    comp = []
    for m in methods:
        t0 = time.perf_counter()
        agg = metrics.aggregate(evaluate(dev, replace(base, method=m), args.scratch, cache, args.tol))
        agg.update(method=m, seconds=round(time.perf_counter() - t0, 2))
        comp.append(agg)
        print(f"  {m:9s} {fmt(agg)}  {agg['seconds']}s")
    write_csv(out / "perbandingan_metode_dev.csv", comp, ["method"] + METRIC_FIELDS + ["seconds"])
    best_method = max(comp, key=lambda r: r[args.select])["method"]
    print(f"  -> metode terpilih ({args.select} DEV tertinggi): {best_method}")

    # --- 2) penyetelan parameter pada DEV ---
    best = replace(base, method=best_method)
    if not args.no_tune and best_method in GRIDS:
        print("\n[2] Penyetelan parameter (DEV)")
        grid = GRIDS[best_method]
        keys = list(grid)
        trials = []
        for vals in itertools.product(*grid.values()):
            cand = replace(best, **dict(zip(keys, vals)))
            agg = metrics.aggregate(evaluate(dev, cand, args.scratch, cache, args.tol))
            trials.append({**dict(zip(keys, vals)), "dice": agg["dice"], "iou": agg["iou"],
                           "f1_tol": agg["f1_tol"]})
        trials.sort(key=lambda r: -r[args.select])
        write_csv(out / "penyetelan_parameter_dev.csv", trials, keys + ["dice", "iou", "f1_tol"])
        best = replace(best, **{k: trials[0][k] for k in keys})
        print(f"  parameter terbaik: { {k: trials[0][k] for k in keys} }  "
              f"{args.select} DEV={trials[0][args.select]:.3f}")
    best.to_file(out / "parameter_final.txt")

    # --- 3) evaluasi akhir pada TEST ---
    print("\n[3] HASIL AKHIR (TEST - tidak dipakai saat penyetelan)")
    rows = evaluate(test, best, args.scratch, cache, args.tol)
    overall = metrics.aggregate(rows)
    print(f"  Keseluruhan : {fmt(overall)}")
    table = [{"kondisi": "SEMUA", **overall}]
    for c in sorted({r["condition"] for r in rows}):
        agg = metrics.aggregate([r for r in rows if r["condition"] == c])
        table.append({"kondisi": c, **agg})
        print(f"  {c:12s}: {fmt(agg)}")
    write_csv(out / "hasil_test_per_kondisi.csv", table, ["kondisi"] + METRIC_FIELDS)
    write_csv(out / "hasil_test_per_citra.csv", rows,
              ["name", "condition", "iou", "dice", "precision", "recall", "precision_tol", "recall_tol",
               "f1_tol", "tp", "fp", "fn"])

    # --- 4) overlay + lokasi/panjang/luas retak ---
    by_name = {s.name: s for s in samples}
    defect_rows = []
    for s in samples:
        c = item(s, best, cache)
        res = pipeline.run(c["img"], best, args.scratch, cache=c)
        gt = c.get("gt")
        if gt is None and s.mask_path is not None:
            gt = dataset.read_mask(s.mask_path, res["mask"].shape)
        nm = s.name.replace('/', '__')
        cv2.imwrite(str(out / "overlay" / f"{nm}.png"), pipeline.overlay(res["image"], res["mask"], gt))
        if gt is not None:     # highlight retak menurut mask: foto asli | foto dengan retak disorot
            cv2.imwrite(str(out / "highlight" / f"{nm}.png"),
                        np.hstack([res["image"], pipeline.highlight(res["image"], gt)]))
        f = max(c["orig_hw"]) / max(res["image"].shape[:2])      # skala ke piksel foto asli
        for i, d in enumerate(res["defects"]):
            defect_rows.append({"citra": s.name, "split": s.split, "id": i,
                                "x_asli": round(d["x"] * f), "y_asli": round(d["y"] * f),
                                "lebar_kotak_asli": round(d["w"] * f), "tinggi_kotak_asli": round(d["h"] * f),
                                "panjang_px_asli": round(d["length_px"] * f),
                                "lebar_rata2_px_asli": round(d["mean_width_px"] * f, 1),
                                "luas_px_asli": round(d["area_px"] * f * f),
                                "cx": round(d["cx"] * f), "cy": round(d["cy"] * f)})
    write_csv(out / "lokasi_luas_cacat.csv", defect_rows,
              ["citra", "split", "id", "x_asli", "y_asli", "lebar_kotak_asli", "tinggi_kotak_asli",
               "panjang_px_asli", "lebar_rata2_px_asli", "luas_px_asli", "cx", "cy"])

    # --- 5) kasus gagal terburuk (analisis kegagalan) ---
    worst = sorted(rows, key=lambda r: r["dice"])[:max(3, min(5, len(rows)))]
    print("\n[4] Kasus gagal terburuk di TEST (untuk analisis kegagalan di laporan):")
    for r in worst:
        s = by_name[r["name"]]
        c = cache[s.name]
        res = pipeline.run(c["img"], best, args.scratch, cache=c)
        panel = np.hstack([res["image"], cv2.cvtColor(res["residual"], cv2.COLOR_GRAY2BGR),
                           pipeline.overlay(res["image"], res["mask"], c["gt"])])
        # kolom: (1) foto, (2) hasil tahap perbaikan kontras, (3) hasil akhir vs mask
        cv2.imwrite(str(out / "failures" / f"{r['name'].replace('/', '__')}.png"), panel)
        why = ("banyak positif palsu (FP>FN): bagian non-retak ikut terdeteksi" if r["fp"] > r["fn"]
               else "banyak retak terlewat (FN>FP): retak tidak terdeteksi/terputus")
        print(f"  {r['name']:30s} Dice={r['dice']:.3f} F1tol={r['f1_tol']:.3f} kondisi={r['condition']}  -> {why}")
    if len(rows) < 3:
        print("  (kurang dari 3 citra test bermask: analisis kegagalan UTS butuh minimal 3 citra)")

    # --- 6) grafik per kondisi ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [t["kondisi"] for t in table]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (k, lab) in enumerate((("iou", "IoU"), ("dice", "Dice"), ("precision", "Precision"),
                                      ("recall", "Recall"))):
            ax.bar(x + (i - 1.5) * 0.2, [t[k] for t in table], 0.2, label=lab)
        ax.set_xticks(x, names)
        ax.set_ylim(0, 1)
        ax.set_ylabel("rata-rata per citra (TEST)")
        ax.set_title("Metrik pada data uji menurut kondisi")
        ax.legend(ncol=4, fontsize=8, loc="upper center")
        fig.tight_layout()
        fig.savefig(out / "grafik_per_kondisi.png", dpi=150)
        plt.close(fig)
    except Exception as e:                        # grafik hanya pelengkap
        print(f"  (grafik dilewati: {e})")

    # --- 7) verifikasi operasi buatan sendiri (syarat UTS no. 2-3) ---
    print("\n[5] Verifikasi operasi buatan sendiri terhadap OpenCV (10 citra)")
    verified = verify(samples, out, 10)

    print("\nCek ketentuan UTS:")
    print(f"  [{'x' if len(samples) >= 50 else ' '}] >= 50 citra          ({len(samples)})")
    print(f"  [{'x' if len(conds) >= 3 else ' '}] >= 3 kondisi           ({len(conds)})")
    print(f"  [{'x' if len(dev) + len(test) >= 15 else ' '}] >= 15 citra bermask   ({len(dev) + len(test)})")
    print(f"  [{'x' if len(rows) >= 3 else ' '}] >= 3 citra untuk analisis kegagalan")
    print(f"  [x] perbandingan metode ({len(methods)} metode, dipilih dari angka DEV)")
    print(f"  [{'x' if len(scratch_ops.active_ops(True)) >= 2 else ' '}] >= 2 operasi dari nol terisi "
          f"({', '.join(scratch_ops.active_ops(True)) or 'belum ada'})")
    print(f"  [{'x' if verified else ' '}] verifikasi vs OpenCV tersimpan")
    print(f"\nSelesai. Semua keluaran ada di: {out.resolve()}")


if __name__ == "__main__":
    main()
