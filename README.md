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
```
(Foto HEIC iPhone: `pip install pillow pillow-heif`)

## Struktur dataset
```
dataset/
├── images/
│   ├── normal/   foto001.jpg ...
│   ├── terang/   ...
│   └── redup/    ...
└── masks/                  # nama sama dengan citra (.png atau .jpg), putih = retak
    ├── normal/   foto001.png ...
    ├── terang/
    └── redup/
```
- Dataset: 20 citra normal, 15 terang, 15 redup (50 citra, semua bermask).
- Foto dari ubin yang sama diberi awalan grup: `ubinA__foto01.jpg`, agar tidak terbagi ke dev dan test.
- Mask harus digambar pada foto dengan orientasi yang sama.
- Citra dan mask boleh JPG. PNG lebih disarankan untuk mask karena garis retak tipis bisa buram akibat kompresi JPG.

## Menjalankan
```
python run_all.py --data dataset --scratch            # lengkap (pakai operasi buatan sendiri)
python run_all.py --data dataset --scratch --no-tune  # lebih cepat, tanpa penyetelan parameter
python run_all.py --data dataset                      # tanpa operasi buatan sendiri (OpenCV)
python verify_scratch.py --data dataset               # hanya verifikasi vs OpenCV
```
Opsi lain: `--select dice|f1_tol`, `--methods ridge,otsu,...`, `--tol 4`, `--size 1280`, `--out results`.

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

## Catatan desain operasi buatan sendiri
- `convolve2d` membalik kernel (konvolusi sejati), sedangkan `cv2.filter2D` berupa korelasi.
  Ketiga kernel Hessian di `crack.py` simetris 180 derajat, sehingga hasilnya sama. Untuk kernel
  tidak simetris (mis. Sobel), kernel harus dibalik pada salah satu jalur.
- `available()` menjalankan uji mandiri kecil; operasi yang gagal uji dimatikan otomatis.

## Dataset publik (bila dipakai)
Isi sumber dan lisensi di sini: _(belum ada)_

## Privasi
Citra yang memuat wajah atau plat nomor orang lain hanya dipakai dengan izin dan disamarkan pada laporan.
