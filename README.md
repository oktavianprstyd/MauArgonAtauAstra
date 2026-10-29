# Prediksi Niat Pembelian Pengunjung Toko Online Berdasarkan Perilaku Sesi
### Proyek Akhir Praktikum Data Science (Semester Ganjil 2026/2027)
**Program Studi Informatika - Fakultas Teknik Industri**  
**Universitas Pembangunan Nasional "Veteran" Yogyakarta**

---

## Informasi Kelompok

- **Nomor Kelompok**: 61
- **Nama Kelompok**: Mau ARGON atau ASTRA?
- **Kelas**: IF-H

### Anggota Kelompok

| No | NIM | Nama Mahasiswa | Kelas |
|:--:|:---:|:---------------|:-----:|
| 1 | 123240135 | Oktavian Prasetya Adi | IF-H |
| 2 | 123240019 | Marvel Valensiano | IF-H |
| 3 | 123240069 | Adhafa Joan Putranto | IF-H |
| 4 | 123240160 | Dzaki Ghatfaan Abhipraya | IF-H |

---

## Overview Proyek

Dalam ekosistem toko online (*e-commerce*), memahami intensi atau niat belanja pengunjung selama berselancar di situs web merupakan faktor penting untuk meningkatkan rasio konversi penjualan. Sebagian besar pengunjung sering kali hanya melihat-lihat katalog produk tanpa melakukan transaksi pembelian.

Proyek ini bertujuan untuk mengolah dan menganalisis data sesi penjelajahan web guna memprediksi apakah suatu sesi kunjungan pengguna akan berujung pada transaksi pembelian (`Revenue = True`) atau tidak (`Revenue = False`). Melalui pendekatan Data Science, pola perilaku navigasi pengunjung diharapkan dapat memberikan pemahaman yang jelas bagi pemilik platform dalam mengenali karakteristik pengunjung yang berpotensi melakukan pembelian.

---

## Tautan Repository & Notebook Utama

- **Repository GitHub**: [https://github.com/oktavianprstyd/MauArgonAtauAstra](https://github.com/oktavianprstyd/MauArgonAtauAstra)
- **Jupyter Notebook Proyek**: [`notebook.ipynb`](notebook.ipynb)

---

## Struktur Direktori Repository

```text
MauArgonAtauAstra/
│
├── data/
│   └── online_shoppers_intention.csv         # Dataset mentah asli dari UCI Machine Learning Repository
│
├── notebook.ipynb                            # Jupyter Notebook utama (Business Understanding, Data Gathering, EDA, Cleaning, Preprocessing)
├── .gitignore                                # Konfigurasi file yang diabaikan Git
└── README.md                                 # Dokumentasi laporan proyek
```

---

## Alur Kerja Notebook (*Workflow*)

Notebook [`notebook.ipynb`](notebook.ipynb) dirancang secara sistematis sesuai dengan panduan praktikum Data Science:

1. **Business Understanding**
   - Menjelaskan latar belakang bisnis pada platform *e-commerce*, tantangan *conversion rate*, rumusan masalah, tujuan analisis, serta metrik evaluasi Machine Learning (*Precision*, *Recall*, *F1-Score*, *ROC-AUC* karena kondisi *class imbalance*).
2. **Data Gathering**
   - Memuat pustaka standar praktikum (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
   - Melakukan pemuatan dataset dan inspeksi awal struktur, bentuk dimensi (12.330 baris $\times$ 18 kolom), dan tipe data.
3. **Exploratory Data Analysis (EDA)**
   - Statistik deskriptif & analisis *skewness* (Right-skewed pada sebagian besar metrik durasi dan nilai halaman).
   - Analisis ketimpangan kelas target `Revenue` (84,53% `False` vs 15,47% `True`).
   - Visualisasi distribusi univariat & bivariat terhadap `Revenue`.
   - Analisis korelasi (*heatmap*): Mengidentifikasi multikolinearitas antara `BounceRates` dan `ExitRates` ($r \approx 0.91$) serta pengaruh kuat `PageValues` terhadap konversi ($r \approx 0.49$).
4. **Data Cleaning**
   - Pemeriksaan *missing values*: Ditemukan 0 nilai kosong (dataset lengkap).
   - Penanganan data duplikat: Mengidentifikasi dan menghapus 125 baris duplikat untuk mencegah distorsi bobot data.
   - Deteksi & analisis *outliers*: Menghitung batas *outlier* menggunakan metode Interquartile Range (IQR) serta memberikan justifikasi domain bahwa nilai ekstrem durasi/belanja merupakan perilaku nyata konsumen antusias (*heavy users*), sehingga dipertahankan dan distandarisasi pada tahap *preprocessing*.
5. **Data Preprocessing**
   - *Feature Engineering*: Membentuk fitur ringkasan sesi `Total_Pages` dan `Total_Duration`.
   - *Categorical Encoding*: Konversi fitur *boolean* (`Weekend`, `Revenue`) ke nilai biner integer (0 dan 1), serta penerapan *One-Hot Encoding* (`pd.get_dummies`) untuk fitur kategorikal nominal (`Month`, `VisitorType`).
   - *Train-Test Split*: Membagi data menjadi 80% data latih dan 20% data uji menggunakan *stratified sampling* (`stratify=y`).
   - *Feature Scaling*: Menstandarisasi fitur numerik dengan `StandardScaler` (Mean = 0, Std = 1) untuk mencegah bias jarak/gradien, dengan penerapan `fit` hanya pada data latih guna mencegah *data leakage*.
   - Seluruh data bersih dan matriks fitur tersimpan langsung di dalam memori DataFrame (`df`, `X_train_scaled`, `X_test_scaled`, `y_train`, `y_test`) yang siap digunakan untuk pemodelan Machine Learning.

---

## Informasi Dataset

Dataset yang digunakan dalam proyek ini bersumber dari repositori publik:
- **Nama Dataset**: Online Shoppers Purchasing Intention Dataset
- **Sumber**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Sitasi**: C. Sakar & S. Polat (2019), *Online Shoppers Purchasing Intention Dataset*, UCI Machine Learning Repository.
- **Lisensi Data**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Jumlah Data Awal**: 12.330 baris (sesi kunjungan)
- **Jumlah Kolom**: 18 atribut (10 fitur numerik, 8 fitur kategorikal/boolean termasuk kolom target `Revenue`)

### Kamus Data

| No | Nama Kolom | Tipe Data | Deskripsi |
|:--:|:-----------|:---------:|:----------|
| 1 | `Administrative` | Integer | Jumlah halaman administratif akun/profil yang dikunjungi pengguna dalam sesi |
| 2 | `Administrative_Duration` | Float | Total durasi waktu (detik) yang dihabiskan pada halaman administratif |
| 3 | `Informational` | Integer | Jumlah halaman informasi atau kebijakan situs yang dikunjungi |
| 4 | `Informational_Duration` | Float | Total durasi waktu (detik) yang dihabiskan pada halaman informasi |
| 5 | `ProductRelated` | Integer | Jumlah halaman katalog produk yang dibuka pengunjung |
| 6 | `ProductRelated_Duration` | Float | Total durasi waktu (detik) melihat halaman katalog produk |
| 7 | `BounceRates` | Float | Persentase pengunjung yang masuk dan langsung keluar tanpa membuka halaman lain |
| 8 | `ExitRates` | Float | Persentase tampilan halaman yang menjadi halaman terakhir sebelum sesi berakhir |
| 9 | `PageValues` | Float | Rata-rata nilai halaman sebelum pengguna menyelesaikan transaksi pembelian |
| 10 | `SpecialDay` | Float | Tingkat kedekatan tanggal sesi dengan hari perayaan tertentu (rentang 0.0 - 1.0) |
| 11 | `Month` | Object | Bulan sesi kunjungan dilakukan |
| 12 | `OperatingSystems` | Integer | Kode identifikasi sistem operasi perangkat pengguna |
| 13 | `Browser` | Integer | Kode identifikasi peramban (browser) web yang digunakan |
| 14 | `Region` | Integer | Kode wilayah geografis lokasi pengguna |
| 15 | `TrafficType` | Integer | Kode saluran rujukan atau sumber lalu lintas pengunjung |
| 16 | `VisitorType` | Object | Tipe pengunjung (`Returning_Visitor`, `New_Visitor`, atau `Other`) |
| 17 | `Weekend` | Boolean | Menunjukkan apakah sesi berlangsung pada akhir pekan (`True`/`False`) |
| 18 | `Revenue` | Boolean | Kolom target: menunjukkan apakah sesi menghasilkan transaksi pembelian (`True`/`False`) |

---

## Petunjuk Menjalankan Notebook

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/oktavianprstyd/MauArgonAtauAstra.git
   cd MauArgonAtauAstra
   ```

2. **Pastikan library Python yang dibutuhkan terinstal:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Jalankan Jupyter Notebook:**
   ```bash
   jupyter notebook notebook.ipynb
   ```
