# Deteksi Retak Permukaan Ubin (Metode Klasik)

Deteksi retak pada ubin bermotif dengan pengolahan citra klasik (tanpa deep learning).
Dua operasi dibuat dari nol memakai NumPy saja: **konvolusi 2D** dan **erosi + dilasi**
(`tile_defect/scratch_ops.py`).

## Struktur proyek
```
deteksi_cacat_ubin/
├── run_all.py            # satu perintah menjalankan seluruh pipeline
├── verify_scratch.py     # verifikasi operasi buatan sendiri vs OpenCV
├── requirements.txt
└── tile_defect/
    ├── dataset.py        # pemuat data + pembagian dev/test
    ├── pipeline.py       # tahap pra-proses ... pengukuran + 5 metode pembanding
    ├── crack.py          # metode utama `ridge` (khusus retak)
    ├── metrics.py        # IoU, Dice, precision, recall (+ versi bertoleransi)
    └── scratch_ops.py    # konvolusi 2D, erosi, dilasi (NumPy saja)
```

## Pemasangan
```
pip install -r requirements.txt

## Struktur dataset
```
dataset/
├── images/
│   ├── normal/   foto.jpg ...
│   ├── terang/   ...
│   └── redup/    ...
└── masks/                  # nama sama dengan citra (.png atau .jpg), putih = retak
    ├── normal/   foto1.jpg ...
    ├── terang/
    └── redup/
```

## Cara Menjalankan
```
python run_all.py --data dataset --scratch            # lengkap (pakai operasi buatan sendiri)
```

## Tahap pipeline
1. Pra-pemrosesan: ubah ukuran (sisi terpanjang 1280), konversi grayscale/Lab, Gaussian blur.
2. Perbaikan kontras: peta ridge kroma (Hessian multi-skala), gerbang simetri, normalisasi kepadatan lokal.
3. Segmentasi: ambang histeresis (median + k*MAD). Pembanding: blackhat, otsu, adaptive, sigma, canny.
4. Pasca-proses morfologi: closing, penekan tekstur garis sejajar, filter panjang/kelurusan, penyambung celah.
5. Pengukuran: lokasi (kotak), panjang, lebar rata-rata, luas tiap retak.

## Pembagian data dan evaluasi
- ~60% dev / ~40% test, deterministik (hash nama grup), seimbang per kondisi.
- Metode dan parameter dipilih **hanya** dari dev; angka akhir **hanya** dari test.
- Metrik dilaporkan keseluruhan dan per kondisi. Ada versi bertoleransi 4 px (`*_tol`)
  karena retak hanya beberapa piksel lebarnya.

## Keluaran (folder `results/`)
| Berkas | Isi |
|---|---|
| `perbandingan_metode_dev.csv` | perbandingan metode pada dev |
| `penyetelan_parameter_dev.csv`, `parameter_final.txt` | penyetelan dan parameter akhir |
| `hasil_test_per_kondisi.csv`, `hasil_test_per_citra.csv` | angka akhir (test) |
| `grafik_per_kondisi.png` | grafik metrik per kondisi |
| `overlay/`, `highlight/` | deteksi vs mask, dan retak menurut mask |
| `lokasi_luas_cacat.csv` | lokasi, panjang, lebar, luas tiap retak (piksel foto asli) |
| `failures/` | panel kasus gagal terburuk (untuk analisis kegagalan) |
| `verifikasi_scratch.csv`, `verifikasi_ringkasan.csv` | waktu dan selisih piksel vs OpenCV |

## Dua Operasi Pilihan:
Konvolusi 2D dengan erosi dan
dilasi

## Dataset publik (bila dipakai)
Tidak menggunakan dataset publik

## Privasi
Citra tidak menggunakan wajah atau gambar pribadi
