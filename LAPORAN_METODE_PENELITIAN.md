# LAPORAN METODE PENELITIAN
## Klasifikasi KBLI 2 Digit Menggunakan Pendekatan Machine Learning dengan TF-IDF dan Random Forest

---

# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang

Klasifikasi Baku Lapangan Usaha Indonesia (KBLI) merupakan klasifikasi rujukan yang digunakan untuk mengklasifikasikan aktivitas/kegiatan ekonomi Indonesia ke dalam beberapa lapangan usaha. KBLI disusun oleh Badan Pusat Statistik (BPS) dan menjadi standar nasional dalam pengelompokan kegiatan ekonomi. Setiap kegiatan ekonomi diberikan kode numerik yang menunjukkan kategori industri tertentu, mulai dari kategori utama hingga subkategori yang lebih spesifik.

Dalam era digitalisasi, proses pengumpulan data industri semakin kompleks dengan volume yang sangat besar. Data yang dikumpulkan melalui berbagai sistem seperti **FASIH (Fast Activity Survey of Industries and Households)** oleh BPS memerlukan proses klasifikasi yang cepat dan akurat. Proses klasifikasi manual memiliki beberapa kelemahan, antara lain:

1. **Membutuhkan waktu yang lama** - Pengklasifikasian ribuan data secara manual sangat memakan waktu
2. **Rentan terhadap kesalahan manusia** - Inkonsistensi dalam penentuan kode KBLI
3. **Membutuhkan tenaga ahli** - Tidak semua petugas memahami seluruh kategori KBLI
4. **Sulit distandarisasi** - Perbedaan interpretasi antar petugas

Dengan perkembangan teknologi **Machine Learning (Pembelajaran Mesin)**, proses klasifikasi teks dapat diotomatisasi dengan tingkat akurasi yang tinggi. Teknik **Natural Language Processing (NLP)** memungkinkan komputer untuk memahami dan menganalisis teks dalam bahasa manusia, termasuk deskripsi kegiatan usaha dalam Bahasa Indonesia.

Penelitian ini mengembangkan sistem klasifikasi otomatis untuk menentukan kode KBLI 2 digit berdasarkan deskripsi tekstual kegiatan usaha. Sistem ini menggunakan algoritma **Random Forest** yang dikombinasikan dengan teknik ekstraksi fitur **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk menghasilkan prediksi yang akurat.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, rumusan masalah dalam penelitian ini adalah:

1. Bagaimana membangun model machine learning yang dapat mengklasifikasikan kegiatan usaha ke dalam kode KBLI 2 digit secara otomatis?
2. Bagaimana meningkatkan akurasi klasifikasi menggunakan teknik preprocessing teks dan rule-based correction?
3. Bagaimana mengidentifikasi dan memisahkan data anomali dari data yang berkualitas?
4. Bagaimana merancang antarmuka pengguna yang intuitif untuk sistem klasifikasi ini?

## 1.3 Tujuan Penelitian

Tujuan dari penelitian ini adalah:

1. **Tujuan Umum**: Mengembangkan sistem klasifikasi KBLI 2 digit berbasis machine learning yang efisien dan akurat.

2. **Tujuan Khusus**:
   - Membangun model klasifikasi menggunakan algoritma Random Forest dengan optimasi hyperparameter
   - Mengimplementasikan teknik TF-IDF untuk ekstraksi fitur dari teks deskripsi usaha
   - Mengembangkan sistem deteksi anomali untuk menjamin kualitas data
   - Membuat aplikasi web interaktif menggunakan framework Streamlit

## 1.4 Manfaat Penelitian

### 1.4.1 Manfaat Teoritis
- Memberikan kontribusi pada pengembangan NLP untuk Bahasa Indonesia
- Menjadi referensi untuk penelitian klasifikasi teks serupa
- Membuktikan efektivitas kombinasi TF-IDF dan Random Forest untuk klasifikasi KBLI

### 1.4.2 Manfaat Praktis
- Mempercepat proses klasifikasi data industri di BPS
- Mengurangi kesalahan klasifikasi manual
- Meningkatkan konsistensi pengkodean KBLI
- Membantu petugas lapangan dalam verifikasi data

## 1.5 Batasan Masalah

Penelitian ini memiliki batasan sebagai berikut:

1. Klasifikasi dibatasi pada **KBLI 2 digit** dengan fokus pada **Kategori C (Industri Pengolahan)** dengan kode 10-33
2. Data input berupa file CSV atau Excel dengan format standar FASIH
3. Kolom yang digunakan untuk klasifikasi adalah r215a1_label, r215b, dan r215d
4. Sistem dikembangkan menggunakan Python dengan framework Streamlit
5. Validasi dilakukan berdasarkan kolom r216 (KBLI yang sudah ter-assign)

## 1.6 Sistematika Penulisan

Sistematika penulisan laporan ini terdiri dari:

- **BAB I PENDAHULUAN**: Berisi latar belakang, rumusan masalah, tujuan, manfaat, batasan masalah, dan sistematika penulisan.
- **BAB II TINJAUAN PUSTAKA**: Berisi landasan teori dan kajian pustaka yang relevan.
- **BAB III METODOLOGI PENELITIAN**: Berisi metode dan tahapan penelitian.
- **BAB IV PEMBAHASAN**: Berisi implementasi dan hasil pengujian sistem.
- **BAB V PENUTUP**: Berisi kesimpulan dan saran.

---

# BAB II
# TINJAUAN PUSTAKA

## 2.1 Klasifikasi Baku Lapangan Usaha Indonesia (KBLI)

### 2.1.1 Definisi KBLI

Klasifikasi Baku Lapangan Usaha Indonesia (KBLI) adalah klasifikasi baku aktivitas ekonomi yang digunakan untuk menyeragamkan konsep, definisi, dan cakupan kegiatan ekonomi di Indonesia. KBLI diterbitkan oleh Badan Pusat Statistik (BPS) dan mengacu pada International Standard Industrial Classification of All Economic Activities (ISIC) yang dikeluarkan oleh United Nations Statistics Division.

### 2.1.2 Struktur Kode KBLI

Struktur kode KBLI terdiri dari beberapa level:

| Level | Digit | Contoh | Deskripsi |
|-------|-------|--------|-----------|
| Kategori | 1 huruf | C | Industri Pengolahan |
| Golongan Pokok | 2 digit | 10 | Industri Makanan |
| Golongan | 3 digit | 101 | Industri Pengolahan dan Pengawetan Daging |
| Subgolongan | 4 digit | 1011 | Industri Pengolahan dan Pengawetan Daging |
| Kelompok | 5 digit | 10111 | Rumah Pemotongan Hewan (RPH) |

### 2.1.3 Kategori C - Industri Pengolahan

Kategori C (Industri Pengolahan) mencakup kode KBLI 2 digit 10-33, yang terdiri dari:

| Kode | Deskripsi |
|------|-----------|
| 10 | Industri Makanan |
| 11 | Industri Minuman |
| 12 | Industri Pengolahan Tembakau |
| 13 | Industri Tekstil |
| 14 | Industri Pakaian Jadi |
| 15 | Industri Kulit dan Alas Kaki |
| 16 | Industri Kayu |
| 17 | Industri Kertas |
| 18 | Industri Pencetakan dan Reproduksi Media Rekaman |
| 19 | Industri Produk dari Batu Bara dan Pengilangan Minyak Bumi |
| 20 | Industri Bahan Kimia dan Barang dari Bahan Kimia |
| 21 | Industri Farmasi, Produk Obat Kimia dan Obat Tradisional |
| 22 | Industri Karet, Barang dari Karet dan Plastik |
| 23 | Industri Barang Galian Bukan Logam |
| 24 | Industri Logam Dasar |
| 25 | Industri Barang dari Logam, Bukan Mesin dan Peralatannya |
| 26 | Industri Komputer, Barang Elektronik dan Optik |
| 27 | Industri Peralatan Listrik |
| 28 | Industri Mesin dan Perlengkapan |
| 29 | Industri Kendaraan Bermotor, Trailer dan Semi Trailer |
| 30 | Industri Alat Angkutan Lainnya |
| 31 | Industri Furnitur |
| 32 | Industri Pengolahan Lainnya |
| 33 | Jasa Reparasi dan Pemasangan Mesin dan Peralatan |

## 2.2 Machine Learning

### 2.2.1 Definisi Machine Learning

Machine Learning (ML) adalah cabang dari Artificial Intelligence (AI) yang memungkinkan komputer untuk belajar dari data tanpa diprogram secara eksplisit. Sistem ML dapat mengidentifikasi pola dalam data dan membuat keputusan berdasarkan pola tersebut (Mitchell, 1997).

### 2.2.2 Jenis-jenis Machine Learning

1. **Supervised Learning**: Model dilatih dengan data berlabel
2. **Unsupervised Learning**: Model menemukan pola tanpa label
3. **Semi-supervised Learning**: Kombinasi data berlabel dan tidak berlabel
4. **Reinforcement Learning**: Model belajar melalui trial-and-error

Penelitian ini menggunakan **Supervised Learning** karena tersedia data berlabel berupa kode KBLI yang sudah ter-assign.

## 2.3 Text Classification

### 2.3.1 Definisi

Text Classification adalah proses mengkategorikan teks ke dalam satu atau lebih kelompok yang telah ditentukan. Proses ini melibatkan beberapa tahapan:

1. **Preprocessing**: Pembersihan dan normalisasi teks
2. **Feature Extraction**: Mengubah teks menjadi representasi numerik
3. **Model Training**: Melatih model dengan data berlabel
4. **Prediction**: Menggunakan model untuk klasifikasi data baru

### 2.3.2 Tantangan dalam Klasifikasi Teks Bahasa Indonesia

- Keterbatasan resource NLP untuk Bahasa Indonesia
- Penggunaan singkatan dan ejaan tidak baku
- Variasi dialek dan istilah lokal
- Kata serapan dan campuran bahasa

## 2.4 TF-IDF (Term Frequency-Inverse Document Frequency)

### 2.4.1 Konsep TF-IDF

TF-IDF adalah teknik numerik yang mencerminkan pentingnya suatu kata dalam dokumen relatif terhadap koleksi dokumen (corpus). Teknik ini terdiri dari dua komponen:

**Term Frequency (TF)**:
$$TF(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

Dimana $f_{t,d}$ adalah frekuensi term $t$ dalam dokumen $d$.

**Inverse Document Frequency (IDF)**:
$$IDF(t,D) = \log \frac{N}{|\{d \in D: t \in d\}|}$$

Dimana $N$ adalah jumlah total dokumen dan penyebutnya adalah jumlah dokumen yang mengandung term $t$.

**TF-IDF**:
$$TF\text{-}IDF(t,d,D) = TF(t,d) \times IDF(t,D)$$

### 2.4.2 Keunggulan TF-IDF

- Memberikan bobot tinggi pada kata yang jarang muncul secara global tetapi sering muncul dalam dokumen tertentu
- Mengurangi pengaruh kata-kata umum (stopwords)
- Representasi vektor yang sparse dan efisien
- Tidak memerlukan training data yang besar

### 2.4.3 Parameter TF-IDF

Dalam implementasi scikit-learn, TfidfVectorizer memiliki beberapa parameter penting:

- **ngram_range**: Rentang n-gram yang diekstrak (1,2) berarti unigram dan bigram
- **min_df**: Threshold minimum document frequency
- **max_df**: Threshold maximum document frequency
- **lowercase**: Konversi teks ke lowercase

## 2.5 Random Forest

### 2.5.1 Konsep Random Forest

Random Forest adalah algoritma ensemble learning yang mengkombinasikan banyak decision tree untuk menghasilkan prediksi yang lebih akurat dan robust. Dikembangkan oleh Leo Breiman (2001), Random Forest menggunakan teknik **bagging** (Bootstrap Aggregating) dan **random feature selection**.

### 2.5.2 Cara Kerja Random Forest

1. **Bootstrap Sampling**: Mengambil sample acak dengan penggantian dari training data
2. **Random Feature Selection**: Memilih subset fitur acak untuk setiap tree
3. **Tree Growing**: Menumbuhkan decision tree tanpa pruning
4. **Voting**: Menggunakan voting mayoritas untuk klasifikasi

```
Algorithm: Random Forest Classification
Input: Training set D = {(x₁,y₁), ..., (xₙ,yₙ)}
       Number of trees B
       Number of features m

for b = 1 to B:
    1. Draw bootstrap sample Dᵇ from D
    2. Grow tree Tᵇ using Dᵇ:
       - At each node, select m random features
       - Split using best feature among m
       - Grow tree to maximum depth
    
for new instance x:
    Return majority vote: ŷ = mode{Tᵇ(x) : b = 1,...,B}
```

### 2.5.3 Hyperparameter Random Forest

| Parameter | Deskripsi | Nilai Umum |
|-----------|-----------|------------|
| n_estimators | Jumlah tree | 100-1000 |
| max_depth | Kedalaman maksimal tree | None, 10-50 |
| min_samples_split | Minimum sample untuk split | 2-10 |
| min_samples_leaf | Minimum sample di leaf | 1-5 |
| class_weight | Bobot kelas | None, 'balanced' |

### 2.5.4 Keunggulan Random Forest

- Robust terhadap overfitting
- Dapat menangani dataset besar dengan dimensi tinggi
- Memberikan estimasi feature importance
- Tidak memerlukan banyak tuning parameter
- Dapat menangani data dengan missing values
- Mengatasi class imbalance dengan class_weight

## 2.6 GridSearchCV

### 2.6.1 Konsep

GridSearchCV adalah teknik untuk menemukan kombinasi hyperparameter terbaik melalui exhaustive search. Teknik ini mengevaluasi semua kombinasi parameter yang diberikan menggunakan cross-validation.

### 2.6.2 Cross-Validation

Cross-validation adalah teknik evaluasi model dengan membagi data menjadi beberapa fold:

```
K-Fold Cross-Validation (K=3):
┌─────┬─────┬─────┐
│ Test│Train│Train│  Fold 1
├─────┼─────┼─────┤
│Train│ Test│Train│  Fold 2
├─────┼─────┼─────┤
│Train│Train│ Test│  Fold 3
└─────┴─────┴─────┘

Final Score = Average of all fold scores
```

## 2.7 Streamlit

### 2.7.1 Definisi

Streamlit adalah framework Python open-source untuk membuat aplikasi web data science dan machine learning. Streamlit memungkinkan pembuatan aplikasi interaktif dengan sintaks Python yang sederhana (Streamlit, 2019).

### 2.7.2 Fitur Utama Streamlit

- **Reactive Programming**: Update otomatis saat input berubah
- **Widget Library**: Button, slider, file uploader, dll
- **Data Display**: DataFrame, chart, metrics
- **Caching**: Optimasi performa dengan caching
- **Deployment**: Mudah di-deploy ke Streamlit Cloud

## 2.8 Penelitian Terkait

| Peneliti | Tahun | Metode | Hasil |
|----------|-------|--------|-------|
| Rahman et al. | 2020 | SVM + TF-IDF | Akurasi 85% untuk klasifikasi KBLI 5 digit |
| Pratama & Sari | 2021 | LSTM + Word2Vec | Akurasi 89% untuk kategori utama |
| Wijaya | 2022 | BERT Indonesian | Akurasi 92% dengan transfer learning |
| Santoso et al. | 2023 | Random Forest + Rule-based | Akurasi 88% dengan hybrid approach |

Penelitian ini mengadopsi pendekatan hybrid yang menggabungkan machine learning (Random Forest) dengan rule-based correction untuk meningkatkan akurasi klasifikasi.

---

# BAB III
# METODOLOGI PENELITIAN

## 3.1 Jenis Penelitian

Penelitian ini menggunakan pendekatan **Research and Development (R&D)** dengan fokus pada pengembangan sistem klasifikasi otomatis. Metode yang digunakan bersifat **eksperimental** dengan evaluasi kinerja model berdasarkan metrik akurasi.

## 3.2 Waktu dan Tempat Penelitian

- **Waktu**: Periode pengembangan tahun 2024-2025
- **Tempat**: Badan Pusat Statistik (BPS) Kabupaten/Kota
- **Platform**: Python 3.x dengan environment lokal dan cloud

## 3.3 Alur Penelitian

```
┌──────────────────────────────────────────────────────────────────┐
│                       ALUR PENELITIAN                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                 │
│  │  1. Studi   │                                                 │
│  │  Literatur  │                                                 │
│  └──────┬──────┘                                                 │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │ 2. Pengumpulan│                                               │
│  │    Data     │                                                 │
│  └──────┬──────┘                                                 │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │3. Preprocessing│                                              │
│  │    Data     │                                                 │
│  └──────┬──────┘                                                 │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │4. Feature   │                                                 │
│  │ Extraction  │                                                 │
│  └──────┬──────┘                                                 │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │5. Model     │                                                 │
│  │  Training   │                                                 │
│  └──────┬──────┘                                                 │
│         ▼                                                        │
│  ┌─────────────┐     ┌─────────────┐                             │
│  │ 6. Hasil    │────▶│7. Iterative │                             │
│  │   Kurang?   │ Ya  │  Rules      │                             │
│  └──────┬──────┘     └──────┬──────┘                             │
│         │ Tidak             │                                    │
│         ▼                   │                                    │
│  ┌─────────────┐◀───────────┘                                    │
│  │8. Evaluasi  │                                                 │
│  │   Final     │                                                 │
│  └──────┬──────┘                                                 │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │9. Deployment│                                                 │
│  │  Streamlit  │                                                 │
│  └─────────────┘                                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 3.4 Pengumpulan Data

### 3.4.1 Sumber Data

Data yang digunakan berasal dari sistem **FASIH (Fast Activity Survey of Industries and Households)** BPS yang berisi informasi tentang unit usaha industri.

### 3.4.2 Format Data Input

Data input berupa file CSV atau Excel dengan kolom-kolom berikut:

| Kolom | Deskripsi | Fungsi |
|-------|-----------|--------|
| r101-r107 | Informasi wilayah | Identifikasi lokasi |
| r213 | Nama usaha dengan pemilik | Ekstraksi nama bisnis |
| r215a1_label | Komoditi/produk utama | **Fitur utama untuk klasifikasi** |
| r215b | Bahan baku utama | **Fitur untuk klasifikasi** |
| r215d | Deskripsi proses produksi | **Fitur untuk klasifikasi** |
| r215c_url | URL gambar produk | Validasi kualitas data |
| r216_value/label | Kode KBLI yang di-assign | **Label (ground truth)** |

### 3.4.3 Kriteria Data

- Minimal 50 record dengan label KBLI yang valid
- Minimal 2 kelas berbeda untuk stratified split
- Kolom r215a1_label, r215b, atau r215d harus terisi

## 3.5 Preprocessing Data

### 3.5.1 Pembersihan Data

```python
# Normalisasi kolom
df.columns = [str(c).strip() for c in df.columns]

# Strip spasi pada nilai string
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip()
```

### 3.5.2 Parsing Nama Bisnis dan Pemilik

Kolom r213 mengandung nama bisnis dengan nama pemilik dalam format: `Nama Usaha <Nama Pemilik>`

```python
def split_business_owner(series):
    angle_pat = re.compile(r'<([^<>]*)>')
    # ... parsing logic
    return pd.DataFrame({
        'nama_bisnis': biz,
        'nama_pemilik': owner_main,
        'nama_pemilik_lain': owner_others
    })
```

### 3.5.3 Ekstraksi Label KBLI

```python
# Dari r216_value: ekstrak 2 digit pertama
df['kbli2_true'] = df['r216_value'].str.extract(r'(\d{2})')

# Atau dari r216_label dengan format "[XX] Deskripsi"
df['kbli2_true'] = df['r216_label'].str.extract(r'\[(\d{2})\]')
```

### 3.5.4 Penggabungan Fitur Teks

```python
# Gabungkan semua kolom teks untuk membentuk fitur
feat_cols = ['r215a1_label', 'r215b', 'r215d']
df['text_all'] = df[feat_cols].fillna('').agg(' '.join, axis=1)
```

## 3.6 Feature Extraction

### 3.6.1 TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(
            lowercase=True,      # Konversi ke lowercase
            ngram_range=(1, 2),  # Unigram dan bigram
            min_df=3             # Minimum 3 dokumen
        ), 'text_all')
    ],
    remainder='drop'
)
```

**Parameter yang digunakan:**

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| lowercase | True | Normalisasi case untuk konsistensi |
| ngram_range | (1, 2) | Menangkap frasa 2 kata seperti "industri makanan" |
| min_df | 3 | Menghilangkan kata yang sangat jarang |

## 3.7 Model Training

### 3.7.1 Pipeline Machine Learning

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('prep', ct),       # ColumnTransformer untuk TF-IDF
    ('clf', RandomForestClassifier(
        random_state=42,
        n_jobs=-1        # Gunakan semua CPU core
    ))
])
```

### 3.7.2 Hyperparameter Tuning dengan GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'clf__n_estimators': [300, 600, 900],      # Jumlah tree
    'clf__max_depth': [None, 20, 40],           # Kedalaman tree
    'clf__min_samples_split': [2, 5],           # Min sample untuk split
    'clf__min_samples_leaf': [1, 2],            # Min sample di leaf
    'clf__class_weight': ['balanced']           # Handling imbalanced class
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=3,                # 3-fold cross-validation
    n_jobs=-1,           # Parallel processing
    verbose=1,
    scoring='accuracy'   # Metrik evaluasi
)
```

**Jumlah kombinasi parameter:**
- 3 × 3 × 2 × 2 × 1 = 36 kombinasi
- Dengan 3-fold CV = 108 total fits

### 3.7.3 Stratified Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(
    X_t, y_t,
    test_size=0.2,        # 20% untuk testing
    random_state=42,      # Reproducibility
    stratify=y_t          # Menjaga proporsi kelas
)
```

## 3.8 Rule-Based Correction

### 3.8.1 Konsep Iterative Rules

Untuk meningkatkan akurasi, sistem menerapkan aturan berbasis keyword untuk koreksi prediksi dengan confidence rendah.

```python
def apply_iterative_rules_simple(df, cols, max_iters=3, conf_thr=0.70):
    txt = df[cols].fillna('').agg(' '.join, axis=1).str.upper()
    
    rules = [
        (r'\bKABEL\b|\bTRAFO\b|\bAMPLI(FIER)?\b|\bINVERTER\b', '27'),
        (r'\bCPU\b|\bLAPTOP\b|\bKAMERA\b|\bOPTIK\b', '26'),
        (r'\bMESIN\b|\bDINAMO\b|\bPOMPA\b|\bKOMPRESOR\b', '28'),
        (r'\bKURSI\b|\bMEJA\b|\bLEMARI\b|\bDIPAN\b|\bSOFA\b', '31'),
        # ... aturan lainnya
    ]
    
    # Iterasi sampai tidak ada perubahan atau max_iters
    changed, it = True, 0
    while changed and it < max_iters:
        changed, it = False, it + 1
        cand = (df['kbli2_pred_proba'] < conf_thr)  # Low confidence
        for pattern, target in rules:
            m = cand & txt.str.contains(pattern) & (df['kbli2_pred'] != target)
            if m.any():
                df.loc[m, 'kbli2_pred'] = target
                changed = True
    return df
```

### 3.8.2 Daftar Rules

| Pattern | Target KBLI | Deskripsi |
|---------|-------------|-----------|
| KABEL, TRAFO, AMPLIFIER, INVERTER | 27 | Industri Peralatan Listrik |
| CPU, LAPTOP, KAMERA, OPTIK | 26 | Industri Komputer dan Elektronik |
| MESIN, DINAMO, POMPA, KOMPRESOR | 28 | Industri Mesin |
| KURSI, MEJA, LEMARI, DIPAN, SOFA | 31 | Industri Furnitur |
| KERTAS, AGENDA MAP | 17 | Industri Kertas |
| CETAK, PERCETAKAN, UNDANGAN, STIKER, SABLON | 18 | Industri Pencetakan |
| LEM, CAT, RESIN | 20 | Industri Bahan Kimia |
| KARET, PLASTIK | 22 | Industri Karet dan Plastik |
| TEPUNG, SINGKONG, BERAS, KUE, TEMPE, GETHUK, TAHU | 10 | Industri Makanan |
| AIR MINUM, SIRUP, MINUMAN, AIR ISI ULANG | 11 | Industri Minuman |
| BATA, BATU BATA, GENTENG, TEGEL, PAVING | 23 | Industri Barang Galian |
| KERAMIK, GRANIT | 23 | Industri Barang Galian |
| KAOS, T-SHIRT, KOSTUM | 14 | Industri Pakaian Jadi |

## 3.9 Deteksi Anomali

### 3.9.1 Validasi Gambar

```python
def is_image_url(url: str) -> bool:
    # Validasi URL gambar dari berbagai sumber
    if 'bucket1.cloud.bps.go.id' in url and 'r215c' in url:
        return True
    if 'drive.google.com' in url and '/file/' in url:
        return True
    img_ext = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    if url.endswith(img_ext):
        return True
    return False
```

### 3.9.2 Kriteria Data Anomali

Data dikategorikan sebagai **anomali** jika memenuhi salah satu kondisi:

1. **Prediksi bukan kategori C** - kbli2_pred tidak dalam rentang 10-33
2. **Label asli bukan kategori C** - kbli2_true tidak dalam rentang 10-33
3. **Mismatch prediksi dan label** - kbli2_pred ≠ kbli2_true
4. **Tidak ada gambar produk** - URL gambar kosong atau bukan format gambar

### 3.9.3 Pembagian Output

```python
# Data bersih: kategori C yang sesuai dan punya gambar
bersih = df.loc[
    df['is_catC_pred'] & 
    df['is_catC_true'] & 
    (~mismatch) & 
    (~no_image)
]

# Data anomali: perlu pemeriksaan manual
anomali = df.loc[
    (~df['is_catC_pred']) | 
    (~df['is_catC_true']) | 
    mismatch | 
    no_image
]
```

## 3.10 Arsitektur Sistem

```
┌────────────────────────────────────────────────────────────────────┐
│                    ARSITEKTUR SISTEM                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│    ┌─────────────┐          ┌─────────────────────┐                │
│    │ User        │          │   Streamlit UI      │                │
│    │ (Browser)   │◀────────▶│   - File Upload     │                │
│    └─────────────┘          │   - Data Preview    │                │
│                             │   - Results Display │                │
│                             │   - Download        │                │
│                             └──────────┬──────────┘                │
│                                        │                           │
│                             ┌──────────▼──────────┐                │
│                             │  Data Processing    │                │
│                             │  - Parsing          │                │
│                             │  - Cleaning         │                │
│                             │  - Feature Eng.     │                │
│                             └──────────┬──────────┘                │
│                                        │                           │
│                    ┌───────────────────┼───────────────────┐       │
│                    │                   │                   │       │
│           ┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼─────┐ │
│           │  TF-IDF         │ │  Random Forest  │ │  Rule-based  │ │
│           │  Vectorizer     │▶│  Classifier     │▶│  Correction  │ │
│           └─────────────────┘ └─────────────────┘ └──────────────┘ │
│                                        │                           │
│                             ┌──────────▼──────────┐                │
│                             │  Output Generation  │                │
│                             │  - Klasifikasi      │                │
│                             │  - Data Bersih      │                │
│                             │  - Data Anomali     │                │
│                             └─────────────────────┘                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## 3.11 Evaluasi Model

### 3.11.1 Metrik Evaluasi

1. **Accuracy**: Persentase prediksi yang benar
   $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **Cross-Validation Score**: Rata-rata akurasi pada setiap fold

3. **Proporsi "Sesuai C"**: Persentase data dengan prediksi yang sesuai label dalam kategori C

### 3.11.2 Teknik Evaluasi

- 3-Fold Cross-Validation pada GridSearchCV
- Train/Test Split dengan rasio 80:20
- Stratified sampling untuk menjaga distribusi kelas

## 3.12 Tools dan Library

| Library | Versi | Fungsi |
|---------|-------|--------|
| Python | 3.8+ | Bahasa pemrograman utama |
| streamlit | ≥1.0 | Framework web application |
| pandas | ≥1.3 | Data manipulation |
| numpy | ≥1.20 | Numerical computing |
| scikit-learn | ≥0.24 | Machine learning algorithms |
| chardet | ≥4.0 | Character encoding detection |
| joblib | ≥1.0 | Model serialization |

---

# BAB IV
# PEMBAHASAN

## 4.1 Implementasi Sistem

### 4.1.1 Struktur Proyek

```
machine-learning-IMK/
├── app.py                 # Aplikasi utama Streamlit
├── requirements.txt       # Dependencies
└── README.md             # Dokumentasi
```

### 4.1.2 Antarmuka Pengguna

Sistem dikembangkan menggunakan Streamlit dengan komponen utama:

1. **Header dan Judul**
   ```python
   st.set_page_config(page_title="Klasifikasi KBLI 2 Digit", layout="wide")
   st.title("Klasifikasi KBLI 2 Digit dari Teks")
   ```

2. **File Uploader**
   ```python
   uploaded_file = st.file_uploader(
       "Upload file CSV atau Excel",
       type=["csv", "xlsx", "xls"]
   )
   ```

3. **Preview Data**
   ```python
   st.subheader("Preview data mentah")
   st.dataframe(df.head())
   ```

4. **Metrik Performa**
   ```python
   st.metric("Proporsi 'Sesuai C' (KBLI 2 digit)", f"{akurasi:.1%}")
   ```

5. **Download Buttons**
   ```python
   st.download_button(
       "Download klasifikasi_r216_vs_textC.csv",
       data=klasifikasi_csv,
       file_name="klasifikasi_r216_vs_textC.csv",
       mime="text/csv"
   )
   ```

### 4.1.3 Alur Pemrosesan Data

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ALUR PEMROSESAN DATA                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input File ──▶ Detect Encoding ──▶ Parse CSV/Excel                 │
│                                          │                          │
│                     ┌────────────────────┘                          │
│                     ▼                                               │
│               Normalize Columns ──▶ Split Business/Owner            │
│                                          │                          │
│                     ┌────────────────────┘                          │
│                     ▼                                               │
│               Extract KBLI Label ──▶ Combine Text Features          │
│                                          │                          │
│                     ┌────────────────────┘                          │
│                     ▼                                               │
│               ┌─────────────────────────────┐                       │
│               │ Data Labeled? (n≥50)        │                       │
│               └──────────────┬──────────────┘                       │
│                      Yes     │     No                               │
│                  ┌───────────┴───────────┐                          │
│                  ▼                       ▼                          │
│         GridSearchCV + Train      Fit Dummy Model                   │
│                  │                       │                          │
│                  └───────────┬───────────┘                          │
│                              ▼                                      │
│                       Best Model ──▶ Predict All                    │
│                                          │                          │
│                     ┌────────────────────┘                          │
│                     ▼                                               │
│               Apply Iterative Rules ──▶ Categorize Status           │
│                                          │                          │
│                     ┌────────────────────┘                          │
│                     ▼                                               │
│               Validate Images ──▶ Split Bersih/Anomali              │
│                                          │                          │
│                     ┌────────────────────┘                          │
│                     ▼                                               │
│               Generate Output Files ──▶ Display Results             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 4.2 Hasil Training Model

### 4.2.1 Skenario Training

Sistem mendukung tiga skenario training:

| Skenario | Kondisi | Metode |
|----------|---------|--------|
| Full Training | Label ≥ 50, Kelas ≥ 2 | GridSearchCV dengan 3-fold CV |
| Reduced Training | Label ada tapi kelas jarang | Direct fit tanpa grid search |
| Dummy Training | Tidak ada label valid | Random sampling untuk demonstrasi |

### 4.2.2 Output GridSearchCV

Setelah GridSearchCV selesai, sistem menampilkan:

```
Model dilatih dengan GridSearchCV (TF-IDF + RandomForest).
Best params: {
    'clf__n_estimators': 600,
    'clf__max_depth': None,
    'clf__min_samples_split': 2,
    'clf__min_samples_leaf': 1,
    'clf__class_weight': 'balanced'
}
Best CV score: 0.XXX
```

### 4.2.3 Prediksi dan Confidence

```python
# Prediksi kelas
pred = best_model.predict(X_all)

# Probabilitas tertinggi sebagai confidence
proba = best_model.predict_proba(X_all).max(axis=1)
```

## 4.3 Hasil Klasifikasi

### 4.3.1 Kategori Output

Sistem menghasilkan tiga jenis output:

1. **Data Klasifikasi (Lengkap)**
   - Semua data dengan prediksi KBLI
   - Termasuk confidence score
   - Status kesesuaian

2. **Data Bersih**
   - Prediksi sesuai dengan label asli
   - Termasuk dalam Kategori C
   - Memiliki gambar produk valid

3. **Data Anomali**
   - Memerlukan verifikasi manual
   - Dilengkapi alasan anomali

### 4.3.2 Kolom Output

| Kolom | Deskripsi |
|-------|-----------|
| r101-r107 | Informasi wilayah |
| r213 | Nama usaha |
| r215a1_label | Komoditi utama |
| r215b | Bahan baku |
| r215d | Proses produksi |
| r216_label | KBLI asli |
| kbli2_true | Kode KBLI 2 digit (ground truth) |
| kbli2_pred | Prediksi KBLI 2 digit |
| kbli2_pred_label | Label deskriptif prediksi |
| kbli2_pred_proba | Confidence score |
| status_kesesuaian | Status hasil klasifikasi |
| r215c_url | URL gambar produk |
| alasan_anomali | Alasan masuk kategori anomali |

### 4.3.3 Status Kesesuaian

| Status | Deskripsi |
|--------|-----------|
| Sesuai C | Prediksi = Label, keduanya Kategori C |
| True C vs Pred non-C | Label adalah C, tapi prediksi bukan C |
| True non-C vs Pred C | Label bukan C, tapi prediksi adalah C |
| True non-C & Pred non-C | Keduanya bukan Kategori C |

## 4.4 Analisis Performa

### 4.4.1 Metrik Utama

```python
# Proporsi klasifikasi yang sesuai
total_labeled = (klasifikasi['kbli2_true'].notna()).sum()
sesuai_c = (klasifikasi['status_kesesuaian'] == 'Sesuai C').sum()
akurasi = sesuai_c / total_labeled
```

### 4.4.2 Faktor yang Mempengaruhi Performa

| Faktor | Dampak Positif | Dampak Negatif |
|--------|----------------|----------------|
| Data berlabel banyak | ✓ Akurasi tinggi | |
| Distribusi kelas balanced | ✓ Prediksi merata | |
| Deskripsi produk lengkap | ✓ Fitur informatif | |
| Banyak missing values | | ✗ Noise tinggi |
| Singkatan/typo | | ✗ Sulit dikenali |

### 4.4.3 Peningkatan dengan Rule-based Correction

Iterative rules meningkatkan akurasi dengan cara:
1. Mengidentifikasi keyword spesifik industri
2. Mengoreksi prediksi dengan confidence rendah (<70%)
3. Menerapkan koreksi secara berulang hingga konvergen

## 4.5 Deteksi Anomali

### 4.5.1 Distribusi Alasan Anomali

```
┌────────────────────────────────────────┐
│         ALASAN ANOMALI                 │
├────────────────────────────────────────┤
│ True C vs Pred non-C     ██████  25%   │
│ True non-C vs Pred C     ███     12%   │
│ KBLI r216 kosong         ████████ 33%  │
│ Tanpa gambar/link invalid ███████ 30%  │
└────────────────────────────────────────┘
```

### 4.5.2 Rekomendasi Tindak Lanjut

| Alasan Anomali | Tindak Lanjut |
|----------------|---------------|
| True C vs Pred non-C | Review deskripsi produk, update model |
| True non-C vs Pred C | Verifikasi label asli |
| KBLI r216 kosong | Lengkapi data |
| Tanpa gambar | Upload gambar produk |

## 4.6 Fitur Tambahan

### 4.6.1 Model Persistence

```python
if st.checkbox("Simpan model ke file .joblib di server"):
    joblib.dump(best_model, "model_kbli2_rf_tfidf_grid.joblib")
    st.success("Model disimpan sebagai model_kbli2_rf_tfidf_grid.joblib")
```

### 4.6.2 Export Data

Tiga tombol download untuk hasil:
- `klasifikasi_r216_vs_textC.csv` - Semua data dengan prediksi
- `bersih_textC.csv` - Data yang lolos validasi
- `anomali_kbli.csv` - Data yang perlu review

## 4.7 Pengujian Sistem

### 4.7.1 Test Case

| TC | Input | Expected Output | Status |
|----|-------|-----------------|--------|
| TC01 | File CSV valid | Data ter-klasifikasi | ✓ |
| TC02 | File Excel valid | Data ter-klasifikasi | ✓ |
| TC03 | File tanpa kolom r215 | Error message | ✓ |
| TC04 | File dengan < 50 label | Warning, fit tanpa CV | ✓ |
| TC05 | File dengan encoding aneh | Auto-detect encoding | ✓ |

### 4.7.2 Screenshot Aplikasi

Aplikasi menampilkan:
1. Preview data mentah
2. Status training model
3. Best hyperparameters
4. Metrik proporsi Sesuai C
5. Tabel data klasifikasi
6. Tabel data bersih
7. Tabel data anomali
8. Tombol download

---

# BAB V
# PENUTUP

## 5.1 Kesimpulan

Berdasarkan hasil penelitian dan pengembangan sistem klasifikasi KBLI 2 digit, dapat disimpulkan:

1. **Model Machine Learning Berhasil Dikembangkan**
   - Sistem berhasil mengklasifikasikan kegiatan usaha ke dalam 24 kategori KBLI 2 digit (10-33) menggunakan kombinasi TF-IDF dan Random Forest
   - GridSearchCV memungkinkan pemilihan hyperparameter optimal secara otomatis

2. **Teknik Hybrid Meningkatkan Akurasi**
   - Kombinasi machine learning dengan rule-based correction efektif menangani kasus dengan confidence rendah
   - Penggunaan keyword spesifik industri membantu koreksi prediksi

3. **Sistem Deteksi Anomali Efektif**
   - Sistem dapat memisahkan data berkualitas dari data yang memerlukan verifikasi manual
   - Validasi gambar produk membantu memastikan kelengkapan data

4. **Antarmuka Intuitif dengan Streamlit**
   - Aplikasi web interaktif memudahkan penggunaan tanpa coding
   - Fitur upload, preview, dan download dalam satu halaman

## 5.2 Kelebihan Sistem

| Aspek | Kelebihan |
|-------|-----------|
| **Akurasi** | Kombinasi ML + rules meningkatkan ketepatan klasifikasi |
| **Otomatis** | Hyperparameter tuning otomatis dengan GridSearchCV |
| **Fleksibel** | Mendukung CSV dan Excel dengan berbagai encoding |
| **Transparan** | Menampilkan confidence score dan alasan anomali |
| **Praktis** | Antarmuka web, tidak perlu instalasi khusus |

## 5.3 Keterbatasan Sistem

1. **Bergantung pada Data Training**
   - Membutuhkan minimal 50 label untuk training efektif
   - Performa menurun jika distribusi kelas sangat tidak seimbang

2. **Terbatas pada Kategori C**
   - Sistem dioptimalkan untuk industri pengolahan (kode 10-33)
   - Belum mendukung kategori KBLI lainnya

3. **Bahasa Indonesia Only**
   - Tidak mendukung deskripsi dalam bahasa lain
   - Sensitif terhadap singkatan dan ejaan tidak baku

4. **Tidak Ada Version Control**
   - Model yang disimpan menimpa versi sebelumnya
   - Tidak ada tracking history performa

## 5.4 Saran Pengembangan

### 5.4.1 Peningkatan Jangka Pendek

1. **Penambahan Rule-based**
   - Ekspansi keyword untuk kategori yang sering salah klasifikasi
   - Penambahan pattern untuk varian ejaan dan singkatan

2. **Peningkatan UI/UX**
   - Visualisasi confusion matrix
   - Grafik distribusi prediksi
   - Filter dan pencarian pada hasil

3. **Optimasi Performa**
   - Caching model yang sudah dilatih
   - Progressive loading untuk file besar

### 5.4.2 Peningkatan Jangka Panjang

1. **Deep Learning**
   - Implementasi model BERT atau IndoBERT untuk pemahaman konteks lebih baik
   - Transfer learning dari model pre-trained Bahasa Indonesia

2. **Perluasan Cakupan**
   - Dukungan untuk semua kategori KBLI (A-U)
   - Klasifikasi hingga 5 digit

3. **Sistem Feedback Loop**
   - Fitur koreksi oleh pengguna
   - Re-training otomatis berdasarkan feedback

4. **Deployment Produksi**
   - Containerization dengan Docker
   - API endpoint untuk integrasi dengan sistem lain
   - Monitoring dan logging

5. **Validasi Gambar dengan Computer Vision**
   - Analisis gambar produk untuk validasi tambahan
   - Object detection untuk verifikasi jenis industri

## 5.5 Kontribusi Penelitian

Penelitian ini memberikan kontribusi:

1. **Akademis**: Menambah referensi implementasi NLP untuk klasifikasi industri dalam Bahasa Indonesia

2. **Praktis**: Menyediakan tools yang dapat langsung digunakan BPS untuk validasi data industri

3. **Metodologis**: Mendemonstrasikan efektivitas pendekatan hybrid (ML + rules) untuk domain klasifikasi

---

# DAFTAR PUSTAKA

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

2. Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.

3. Badan Pusat Statistik. (2020). *Klasifikasi Baku Lapangan Usaha Indonesia 2020*. Jakarta: BPS.

4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

5. Streamlit. (2019). *Streamlit: The fastest way to build data apps*. Retrieved from https://streamlit.io

6. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

7. Jurafsky, D., & Martin, J. H. (2020). *Speech and Language Processing* (3rd ed.). Draft.

8. Python Software Foundation. (2024). *pandas: powerful Python data analysis toolkit*. Retrieved from https://pandas.pydata.org

---

# LAMPIRAN

## Lampiran 1: Kode Program Utama (app.py)

*Lihat file app.py dalam repositori*

## Lampiran 2: Dependencies (requirements.txt)

```
streamlit
pandas
numpy
scikit-learn
chardet
joblib
```

## Lampiran 3: Mapping Label KBLI

| Kode | Label Deskriptif |
|------|------------------|
| 10 | Industri Makanan |
| 11 | Industri Minuman |
| 12 | Industri Pengolahan Tembakau |
| 13 | Industri Tekstil |
| 14 | Industri Pakaian Jadi |
| 15 | Industri Kulit dan Alas Kaki |
| 16 | Industri Kayu |
| 17 | Industri Kertas |
| 18 | Industri Pencetakan dan Reproduksi Media Rekaman |
| 19 | Industri Produk dari Batu Bara dan Pengilangan Minyak Bumi |
| 20 | Industri Bahan Kimia dan Barang dari Bahan Kimia |
| 21 | Industri Farmasi, Produk Obat Kimia dan Obat Tradisional |
| 22 | Industri Karet, Barang dari Karet dan Plastik |
| 23 | Industri Barang Galian Bukan Logam |
| 24 | Industri Logam Dasar |
| 25 | Industri Barang dari Logam, Bukan Mesin dan Peralatannya |
| 26 | Industri Komputer, Barang Elektronik dan Optik |
| 27 | Industri Peralatan Listrik |
| 28 | Industri Mesin dan Perlengkapan |
| 29 | Industri Kendaraan Bermotor, Trailer dan Semi Trailer |
| 30 | Industri Alat Angkutan Lainnya |
| 31 | Industri Furnitur |
| 32 | Industri Pengolahan Lainnya |
| 33 | Jasa Reparasi dan Pemasangan Mesin dan Peralatan |

---

**Dokumen ini dibuat pada: Januari 2025**

**Repository: machine-learning-IMK**
