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

Proyek ini bertujuan untuk mengolah dan menganalisis data sesi penjelajahan web guna memprediksi apakah suatu sesi kunjungan pengguna akan berujung pada transaksi pembelian (`Revenue = True`) atau tidak (`Revenue = False`). Melalui pendekatan Data Science, pola perilaku navigasi pengunjung diharapkan dapat memberikan pemahaman yang jelas bagi pemilik platform dalam mengenali karakteristik pengunjung yang berpotensi melakukan pembelian serta mendukung keputusan intervensi pemasaran yang tepat sasaran.

---

## Tautan Repository & Notebook Utama

- **Repository GitHub**: [https://github.com/oktavianprstyd/MauArgonAtauAstra](https://github.com/oktavianprstyd/MauArgonAtauAstra)
- **Jupyter Notebook Proyek**: [Buka di Google Colab](https://colab.research.google.com/drive/1WsRG95iQbLxMq5VFP_db974yDQUhhh8m?usp=sharing)

---

## Struktur Direktori Repository

```text
MauArgonAtauAstra/
│
├── data/
│   └── online_shoppers_intention.csv         # Dataset mentah asli dari UCI Machine Learning Repository
│
├── notebook.ipynb                            # Jupyter Notebook utama dari Business Understanding hingga Rekomendasi Bisnis
├── requirements.txt                          # Daftar dependensi library Python untuk reproduksibilitas
├── .gitignore                                # Konfigurasi file yang diabaikan Git
└── README.md                                 # Dokumentasi laporan proyek
```

---

## Alur Kerja Notebook (*Workflow*)

Notebook [notebook.ipynb](https://colab.research.google.com/drive/1WsRG95iQbLxMq5VFP_db974yDQUhhh8m?usp=sharing) dirancang secara sistematis sesuai dengan panduan praktikum Data Science:

1. **Business Understanding**
   - Menjelaskan latar belakang bisnis pada platform *e-commerce*, tantangan *conversion rate*, rumusan masalah, tujuan analisis, serta metrik evaluasi Machine Learning (*Precision*, *Recall*, *F1-Score*, *ROC-AUC* karena kondisi *class imbalance*).
2. **Data Gathering & Inspeksi Awal**
   - Memuat pustaka standar praktikum (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
   - Melakukan pemuatan dataset dan inspeksi awal struktur, bentuk dimensi (12.330 baris $\times$ 18 kolom), dan tipe data.
3. **Exploratory Data Analysis (EDA)**
   - Statistik deskriptif & analisis *skewness* (*right-skewed* pada sebagian besar metrik durasi dan nilai halaman).
   - Analisis ketimpangan kelas target `Revenue` (84,53% `False` vs 15,47% `True` pada data mentah).
   - Visualisasi distribusi univariat, bivariat terhadap `Revenue`, serta analisis *Conversion Rate (%)* berdasarkan `Month`, `VisitorType`, `Weekend`, dan `SpecialDay` (puncak konversi di bulan November sebesar 25,35%).
   - Analisis korelasi (*heatmap*): Mengidentifikasi multikolinearitas antara `BounceRates` dan `ExitRates` ($r \approx 0.91$) serta pengaruh kuat `PageValues` terhadap konversi ($r \approx 0.49$).
4. **Data Cleaning**
   - Pemeriksaan *missing values*: Ditemukan 0 nilai kosong (dataset lengkap).
   - Penanganan data duplikat: Mengidentifikasi dan menghapus 125 baris duplikat (menyisakan 12.205 baris dengan proporsi `Revenue = True` sebesar 15,63%) untuk mencegah distorsi bobot data.
   - Deteksi & analisis *outliers*: Menghitung batas *outlier* menggunakan metode Interquartile Range (IQR) serta memberikan justifikasi domain bahwa nilai ekstrem durasi/belanja merupakan perilaku nyata konsumen antusias (*heavy users*), sehingga dipertahankan dan distandarisasi pada tahap *preprocessing*.
5. **Audit Data, Preprocessing, dan Ringkasan *Before vs. After***
   - Membuat fitur agregat `Total_Pages` dan `Total_Duration`, memisahkan data secara terstratifikasi 80:20 (9.764 data latih & 2.441 data uji), dan mempelajari *encoding* kategori serta *scaling* di dalam `ColumnTransformer` + `Pipeline` guna mencegah *data leakage*.
   - Menyajikan tabel ringkasan *Before vs. After Cleaning & Preprocessing*.
6. **Modelling dan Tuning**
   - Membandingkan SVM kernel RBF dan Random Forest dengan pemilihan konfigurasi serta kalibrasi ambang prediksi berdasarkan F1 kelas pembelian pada validasi silang terstratifikasi 3-fold.
7. **Evaluasi Data Uji**
   - Membandingkan *Precision*, *Recall*, *F1-Score*, dan *ROC-AUC* dilengkapi *confusion matrix* dan kurva ROC.
8. **Konklusi Awal**
   - Merangkum performa model dasar dan arah peningkatan eksperimen lanjutan.
9. **Eksperimen Lanjutan Dengan dan Tanpa `PageValues`**
   - Membandingkan performa SVM dan Random Forest pada skenario dengan dan tanpa `PageValues` untuk mengantisipasi ketersediaan fitur tersebut sebelum *checkout*.
10. **Pengujian Fitur Perilaku dan Tuning Lanjutan**
    - Menambahkan fitur perilaku sesi (`Ada_PageValue`, `Durasi_per_Halaman`, `Proporsi_Halaman_Produk`) serta *tuning* hiperparameter dan ambang batas lanjutan.
11. **Uji Kontribusi Kelompok Fitur**
    - Validasi silang bersarang (*nested CV* 6 fold luar dengan kalibrasi ambang internal) membandingkan fitur lengkap, fitur terpilih, dan `PageValues` saja.
12. **Uji Lanjutan Menuju F1 0,80 & Analisis Diagnostik Model**
    - Menguji 6 konfigurasi Random Forest tambahan, dilanjutkan evaluasi **Kurva Precision-Recall (PR Curve)**, **Permutation Feature Importance**, dan **Error Analysis** (karakteristik *False Positive* dan *False Negative*).
13. **Evaluasi Lengkap per Model & Terjemahan ke Dampak Bisnis (13.1)**
    - Menyajikan perbandingan metrik lengkap (Akurasi, Precision, Recall, F1 pembelian, Macro F1, Weighted F1, ROC-AUC), laporan per kelas, *confusion matrix*, serta strategi intervensi berbasis probabilitas ($P \ge 0.70$, $0.40 \le P < 0.70$, $P < 0.40$) dan kalibrasi ambang batas (*threshold calibration*).

---

## Ringkasan Hasil Eksperimen

### 1. Eksperimen Skenario Dengan & Tanpa `PageValues` (Data Uji)

| Skenario | Model | Precision | Recall | F1 | ROC-AUC |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Dengan `PageValues` | SVM | 0,628 | 0,743 | 0,681 | 0,897 |
| Dengan `PageValues` | Random Forest | 0,662 | 0,702 | 0,681 | 0,931 |
| Tanpa `PageValues` | SVM | 0,324 | 0,555 | 0,409 | 0,769 |
| Tanpa `PageValues` | Random Forest | 0,350 | 0,568 | 0,433 | 0,788 |

SVM terpilih pada skenario awal dengan `PageValues` berdasarkan F1 validasi 0,689 (mendeteksi 284 dari 382 sesi pembelian dengan 168 *false positive*). Tanpa `PageValues`, Random Forest unggul dengan F1 uji 0,433. Penggunaan model dengan `PageValues` pada sistem produksi *real-time* memerlukan verifikasi bahwa fitur sudah tersedia sebelum *checkout*.

### 2. Evaluasi Akhir Lengkap per Model (Setelah Tuning Fitur Perilaku)

| Model (dengan `PageValues`) | Akurasi | Precision Pembelian | Recall Pembelian | F1 Pembelian | Macro F1 (mF1) | Weighted F1 | ROC-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Random Forest (Model Utama)** | **90,17%** | **67,84%** | 70,68% | **69,23%** | **81,69%** | **90,25%** | **93,26%** |
| SVM | 89,35% | 63,80% | **73,82%** | 68,45% | 81,02% | 89,66% | 89,58% |

- **Model Utama**: Random Forest (ambang optimal `0.3989`) mendeteksi **270 dari 382** sesi pembelian (*True Positive*), menghasilkan **128** *False Positive*, **112** *False Negative*, dan **1.931** *True Negative*.
- **Uji Kontribusi Fitur**: Pada 6 fold luar dengan kalibrasi ambang internal, Random Forest dengan fitur lengkap unggul di semua fold (rata-rata F1 **0,688**), dibandingkan `PageValues` saja (0,660) dan fitur terpilih (0,663).
- **Uji Lanjutan Menuju F1 0,80**: Enam konfigurasi Random Forest tambahan menghasilkan F1 validasi di rentang 0,684–0,690, sehingga model pembanding tetap dipertahankan.
- **Rekomendasi Bisnis**: Segmentasi intervensi berbasis probabilitas—fokus pemberian promo/insentif pada segmen moderat ($0.40 \le P < 0.70$), *frictionless checkout* tanpa diskon pada segmen tinggi ($P \ge 0.70$), dan *retargeting* non-invasif pada segmen rendah ($P < 0.40$).

---

## Informasi Dataset

Dataset yang digunakan dalam proyek ini bersumber dari repositori publik:
- **Nama Dataset**: Online Shoppers Purchasing Intention Dataset
- **Sumber**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Sitasi**: C. Sakar & S. Polat (2019), *Online Shoppers Purchasing Intention Dataset*, UCI Machine Learning Repository.
- **Lisensi Data**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Jumlah Data Awal**: 12.330 baris (sesi kunjungan) → 12.205 baris setelah penghapusan 125 duplikat
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
| 13 | `Browser` | Integer | Kode identifikasi peramban (*browser*) web yang digunakan |
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

2. **Instal seluruh dependensi Python:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Jupyter Notebook:**
   ```bash
   jupyter notebook notebook.ipynb
   ```
