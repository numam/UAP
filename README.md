# Analisis Penampakan UFO untuk Strategi Konten/Film


Aplikasi web berbasis Streamlit untuk menganalisis data penampakan UFO guna memberikan rekomendasi strategi konten atau film. Aplikasi ini memungkinkan pengguna untuk menanyakan bentuk UFO yang paling cocok berdasarkan lokasi, melakukan analisis mendalam berdasarkan lokasi, melatih model machine learning, serta mengevaluasi performa model. Tema utama adalah membantu pembuat konten dan sineas dalam menciptakan cerita yang autentik berdasarkan data penampakan UFO nyata.

## Deskripsi Proyek

Proyek ini menganalisis data historis penampakan UFO untuk mengidentifikasi pola bentuk, lokasi, dan timeline. Dirancang khusus untuk pembuat konten/film, aplikasi memberikan rekomendasi bentuk UFO yang paling sesuai untuk cerita dengan latar lokasi tertentu, berdasarkan data penampakan asli. Fitur utama meliputi:

- **Analisis Query Berbasis Bahasa Alami**: Ajukan pertanyaan seperti "Bentuk UFO apa yang cocok untuk film di Anchorage?" untuk mendapatkan rekomendasi yang disesuaikan.
- **Analisis Berdasarkan Lokasi**: Pilih kota, negara bagian, atau negara untuk melihat distribusi bentuk UFO dan tren waktu.
- **Analisis Berdasarkan Bentuk**: Pilih bentuk UFO tertentu untuk melihat distribusi lokasi, persentase total kemunculan, dan ranking lokasi lain dengan kemunculan bentuk yang sama.
- **Pelatihan Model**: Latih tiga model neural network secara otomatis, dengan penyimpanan ke lokal.
- **Evaluasi Model**: Bandingkan performa tiga model dengan metrik lengkap, termasuk perbandingan dan visualisasi.
- **EDA Dataset**: Halaman Home menampilkan eksplorasi data awal (EDA) seperti distribusi bentuk dan tren tahunan.

Antarmuka interaktif dibangun dengan Streamlit, visualisasi Plotly, dan PyTorch untuk pelatihan/inferensi model.

## Dataset

Dataset bersumber dari [Corgis UFO Sightings Dataset](https://corgis-edu.github.io/corgis/csv/ufo_sightings/), yang berisi lebih dari 88.000 laporan penampakan UFO dari tahun 1906 hingga 2014, terutama dari Amerika Serikat.

### Kolom Utama yang Digunakan
| Nama Kolom                | Deskripsi                           |
|---------------------------|-------------------------------------|
| `Location.City`          | Kota penampakan (misalnya: Anchorage) |
| `Location.State`         | Kode negara bagian (misalnya: AK)   |
| `Location.Country`       | Negara (kebanyakan AS)              |
| `Data.Shape`             | Bentuk UFO yang dilaporkan (misalnya: light, triangle) – **Variabel Target** |
| `Data.Description excerpt` | Cuplikan deskripsi penampakan      |
| `Dates.Sighted.Year`     | Tahun penampakan                    |

### Langkah Preprocessing
1. **Pembersihan Kolom**: Hilangkan spasi ekstra dan tangani variasi nama kolom (misalnya: ganti titik dengan spasi).
2. **Pemfilteran Data**: Buang baris dengan nilai hilang pada `Data.Shape`. Batasi bentuk pada yang memiliki minimal 10 penampakan untuk keseimbangan pelatihan.
3. **Encoding**:
   - Kategorikal: LabelEncoder untuk kota, negara bagian, negara, dan bentuk.
   - Numerik: StandardScaler untuk fitur tahun.
4. **Rekayasa Fitur**: Encoding label untuk lokasi dan tahun, menghasilkan vektor input 4-dimensi (kota, negara bagian, negara, tahun).
5. **Pembagian Data**: Split 80/20 dengan stratifikasi pada bentuk.
6. **Penanganan Ketidakseimbangan Kelas**: Model dilatih pada 21 bentuk umum (misalnya: light, triangle, circle) untuk menghindari kelas langka yang memengaruhi hasil.

Dataset yang dibersihkan memiliki ~80.000 rekaman.

## Model

Tiga arsitektur neural network diimplementasikan untuk klasifikasi multi-kelas (memprediksi bentuk UFO dari fitur lokasi dan tahun). Semua model menggunakan CrossEntropyLoss, optimizer Adam dengan penjadwalan learning rate, dan early stopping (patience=15 epoch). Pelatihan dilakukan di GPU/CPU dengan batch size 32.

### 1. MLP (Multilayer Perceptron)
Jaringan feedforward dengan 3 lapisan tersembunyi (128, 64, 32 neuron), LayerNorm, aktivasi ReLU, dan dropout 30%. Model baseline sederhana untuk klasifikasi data tabular.

### 2. TabNet (Terinspirasi Pretrained)
Arsitektur TabNet sederhana dengan blok transformer fitur, mekanisme perhatian (sigmoid-gated), dan klasifier akhir. Terinspirasi dari model tabular yang interpretable, menggunakan seleksi fitur berurutan melalui bobot perhatian.

### 3. FT-Transformer (Feature Tokenizer Transformer)
Model berbasis transformer dengan embedding fitur, self-attention multi-head (4 head, 2 lapisan), dan proyeksi feedforward. Memperlakukan setiap fitur sebagai token dalam urutan untuk menangkap interaksi.

**Hyperparameter Pelatihan** (per model):
| Model          | Num Classes | Best Val Acc | Epochs | Learning Rate |
|----------------|-------------|--------------|--------|---------------|
| MLP           | 21         | 22.10%      | 50    | 0.001        |
| TabNet        | 21         | 22.32%      | 70    | 0.0005       |
| FT-Transformer| 21         | 22.10%      | 80    | 0.0001       |

Model disimpan di `saved_models/` dengan metadata, bobot, encoder, dan riwayat pelatihan.

## Hasil Evaluasi dan Analisis Perbandingan

Model dievaluasi pada test set terpisah menggunakan akurasi, F1-score, dan metrik loss. Karena ketidakseimbangan kelas tinggi (misalnya: "light" mendominasi ~40%), akurasi baseline sekitar 22%. TabNet menunjukkan keunggulan tipis pada macro F1 karena penanganan bentuk langka yang lebih baik.

### Ringkasan Perbandingan Model
| Model                          | Test Accuracy | Best Val Acc | Final Train Acc | Final Val Acc | Train Loss | Val Loss | Overfitting | Macro F1 | Weighted F1 |
|--------------------------------|---------------|--------------|-----------------|---------------|------------|----------|-------------|----------|-------------|
| MLP (Multilayer Perceptron)    | 22.49%       | 22.10%      | 21.97%         | 22.10%       | 2.589     | 2.5838  | -0.13%     | 2.87%   | 9.34%      |
| TabNet (Pretrained-inspired)   | 22.60%       | 22.32%      | 22.04%         | 22.28%       | 2.5830    | 2.5806  | -0.24%     | 3.22%   | 10.08%     |
| FT-Transformer (Pretrained)    | 22.50%       | 22.10%      | 21.96%         | 22.06%       | 2.5976    | 2.5856  | -0.10%     | 2.89%   | 9.35%      |

### Wawasan Analisis
- **Model Terbaik**: TabNet unggul sedikit pada akurasi test (22.60%) dan weighted F1 (10.08%), kemungkinan karena mekanisme perhatiannya yang menangkap interaksi lokasi-bentuk lebih baik.
- **Overfitting**: Semua model menunjukkan overfitting minimal (gap negatif menandakan underfitting ringan, yang diinginkan untuk generalisasi). Tidak ada model melebihi 5% gap.
- **Macro vs. Weighted F1**: Macro F1 rendah (~3%) menyoroti performa buruk pada bentuk langka; weighted F1 lebih tinggi karena memfavoritkan kelas dominan.
- **Dinamika Pelatihan**: TabNet memerlukan lebih banyak epoch (70) tapi konvergen lebih cepat dengan LR lebih rendah. Semua model stabil setelah ~30 epoch.
- **Keterbatasan**: Dataset tidak seimbang membatasi potensi; pekerjaan selanjutnya bisa termasuk oversampling SMOTE atau focal loss.

Matriks kebingungan (divisualisasikan di app) menunjukkan "light" dan "triangle" paling mudah diprediksi, dengan bentuk langka seperti "cigar" sering salah diklasifikasikan sebagai "unknown."

## Panduan Menjalankan Sistem Website Secara Lokal

### Prasyarat
- Python 3.8+ (diuji pada 3.12)
- Git

### Langkah-langkah
1. **Klon Repository**:
   ```
   git clone [<url-repo-anda>](https://github.com/numam/UAP.git)
   cd UAP
   ```

2. **Unduh Dataset**:
   - Unduh `ufo_sightings.csv` dari [Corgis UFO Sightings](https://corgis-edu.github.io/corgis/csv/ufo_sightings/).
   - Letakkan di root proyek (folder sama dengan `app.py`).

3. **Instal Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   (Buat `requirements.txt` jika diperlukan: `pip freeze > requirements.txt`. Paket utama: `streamlit`, `pandas`, `numpy`, `plotly`, `torch`, `scikit-learn`.)

4. **Jalankan Aplikasi**:
   ```
   streamlit run app.py
   ```
   - Dibuka di `http://localhost:8501`.
   - Navigasi sidebar: Home (EDA Dataset), Analisis Query (cari dengan query inputan), Analisis Lokasi (select berdasarkan lokasi), Evaluasi Model (perbandingan 3 model), Manajemen Model (train & simpan otomatis).

5. **Pelatihan/Load Model** (Opsional):
   - Model dilatih langsung di UI (otomatis tersimpan ke `saved_models/`). Gunakan halaman Manajemen Model untuk load/hapus.
   - Pastikan direktori `saved_models/` ada (dibuat otomatis).

### Mekanisme Pencarian di Web
1. **Query Inputan**: Masukkan pertanyaan bahasa alami di halaman "Analisis Query" (misalnya: "Bentuk UFO apa yang cocok untuk film di Anchorage AK?"). Sistem ekstrak lokasi dan berikan rekomendasi bentuk teratas beserta ciri-ciri (warna, ukuran, gerakan).
2. **Select Berdasarkan Lokasi**: Di halaman "Analisis Lokasi", pilih tipe (Kota/State/Negara), lalu analisis distribusi bentuk, pie chart, dan tren timeline.

### Troubleshooting
- **Dataset Tidak Ditemukan**: Pastikan `ufo_sightings.csv` di root.
- **Masalah CUDA**: App fallback ke CPU; instal PyTorch dengan CUDA jika diperlukan.
- **Konflik Port**: Gunakan `streamlit run app.py --server.port 8502`.

## Fitur Detail
- **Home**: EDA dataset – overview metrik, distribusi bentuk, tren tahunan.
- **Analisis Query**: Ekstraksi NLP untuk kota/negara bagian; rekomendasi dengan karakteristik (misalnya: "cahaya oranye dalam formasi").
- **Analisis Lokasi**: Pilih lokasi; lihat bentuk teratas, pie, timeline.
- **Evaluasi Model**: Bandingkan metrik, matriks kebingungan, kurva loss/akurasi.
- **Manajemen Model**: Load/hapus model tersimpan; ekspor folder untuk backup.

## Kontribusi
Fork repo, buat branch, dan submit PR. Issue diterima!

## Lisensi
MIT License – bebas digunakan dan dimodifikasi.

---

*Terakhir Diperbarui: 24 Desember 2025*
