# Sequence Diagram Sederhana - Klasifikasi KBLI 2 Digit

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant UI as Streamlit UI
    participant Prep as Data Preprocessing
    participant ML as ML Pipeline
    participant Output as Output Handler

    User->>UI: Upload file CSV/Excel
    UI->>Prep: Baca & parsing file
    Prep->>Prep: Normalisasi kolom
    Prep->>Prep: Split nama bisnis & pemilik (r213)
    Prep->>Prep: Extract kbli2_true dari r216
    Prep->>Prep: Gabungkan fitur teks (r215a1_label, r215b, r215d)
    Prep-->>UI: Return DataFrame terproses

    UI-->>User: Tampilkan Preview Data Mentah

    UI->>ML: Build Pipeline (TF-IDF + RandomForest)
    
    alt Cukup label (≥50 samples)
        ML->>ML: GridSearchCV (train/test split 80:20)
        ML->>ML: Cross-validation 3 folds
        ML-->>UI: Return best_model + best_params
        UI-->>User: Tampilkan Best Params & CV Score
    else Tidak cukup label
        ML->>ML: Train dengan parameter default
        ML-->>UI: Return trained model
    end

    UI->>ML: Predict kbli2_pred & confidence
    ML-->>UI: Return prediksi + probabilitas

    UI->>Prep: Apply aturan iteratif (keyword rules)
    Prep-->>UI: Return prediksi yang dikoreksi

    UI->>Prep: Validasi kategori C & cek URL gambar
    Prep-->>UI: Return status validasi

    UI->>Output: Split data
    Output->>Output: klasifikasi = semua data
    Output->>Output: bersih = data valid (catC + match + ada gambar)
    Output->>Output: anomali = data bermasalah + alasan
    Output-->>UI: Return 3 DataFrames

    UI-->>User: Tampilkan Metrik Akurasi
    UI-->>User: Tampilkan Tabel Klasifikasi
    UI-->>User: Tampilkan Tabel Data Bersih
    UI-->>User: Tampilkan Tabel Data Anomali

    User->>UI: Klik Download
    UI->>Output: Generate CSV
    Output-->>User: Download klasifikasi_r216_vs_textC.csv
    Output-->>User: Download bersih_textC.csv
    Output-->>User: Download anomali_kbli.csv

    opt Simpan Model
        User->>UI: Centang "Simpan model"
        UI->>Output: joblib.dump(model)
        Output-->>User: File .joblib tersimpan
    end
```

## Ringkasan Alur

| No | Tahap | Deskripsi |
|----|-------|-----------|
| 1 | Upload | User upload file CSV/Excel |
| 2 | Preprocessing | Parsing, normalisasi, ekstrak fitur |
| 3 | Training | GridSearchCV dengan TF-IDF + RandomForest |
| 4 | Prediksi | Klasifikasi KBLI 2 digit + confidence |
| 5 | Koreksi | Aturan keyword untuk confidence rendah |
| 6 | Validasi | Cek kategori C & URL gambar |
| 7 | Split | Bagi ke klasifikasi/bersih/anomali |
| 8 | Download | Export hasil ke CSV |
