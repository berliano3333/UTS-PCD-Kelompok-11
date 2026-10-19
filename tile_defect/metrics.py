"""Metrik piksel: IoU, Dice, precision, recall (mask biner 0/1).

Retak adalah garis tipis (beberapa piksel), sehingga selisih 1-2 piksel antara
prediksi dan ground truth sudah menurunkan IoU/Dice tajam walaupun lokasinya benar.
Karena itu ada juga versi BERTOLERANSI (`*_tol`): prediksi dihitung benar bila berada
dalam jarak TOL_PX piksel dari ground truth (dan sebaliknya untuk recall). Laporkan
keduanya dan jelaskan alasannya di laporan.
"""
import cv2
import numpy as np

TOL_PX = 4   # toleransi pada resolusi kerja (size=1280 di pipeline)


def confusion(pred: np.ndarray, gt: np.ndarray):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    return tp, fp, fn


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def scores(pred: np.ndarray, gt: np.ndarray, tol: int = TOL_PX) -> dict:
    tp, fp, fn = confusion(pred, gt)
    # Jika gt dan pred sama-sama kosong (ubin mulus, tidak ada deteksi) => sempurna.
    iou = 1.0 if (tp + fp + fn) == 0 else tp / (tp + fp + fn)
    dice = 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    precision = 1.0 if (tp + fp) == 0 and fn == 0 else (tp / (tp + fp) if (tp + fp) else 0.0)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)

    p, g = pred.astype(bool), gt.astype(bool)
    n_p, n_g = int(p.sum()), int(g.sum())
    g_d, p_d = _dilate(g, tol), _dilate(p, tol)
    prec_tol = 1.0 if n_p == 0 and n_g == 0 else (float((p & g_d).sum()) / n_p if n_p else 0.0)
    rec_tol = 1.0 if n_g == 0 else float((g & p_d).sum()) / n_g
    f1_tol = 0.0 if (prec_tol + rec_tol) == 0 else 2 * prec_tol * rec_tol / (prec_tol + rec_tol)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall,
            "precision_tol": prec_tol, "recall_tol": rec_tol, "f1_tol": f1_tol,
            "tp": tp, "fp": fp, "fn": fn}


KEYS = ("iou", "dice", "precision", "recall", "precision_tol", "recall_tol", "f1_tol")


def aggregate(rows: list) -> dict:
    """Rata-rata per citra + IoU piksel-gabungan (micro)."""
    if not rows:
        return {}
    out = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    out["micro_iou"] = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    out["n"] = len(rows)
    return out
