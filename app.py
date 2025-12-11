# =========================
# KBLI 2 DIGIT + SMOTE TOMEK (FORMAT KOLOM BERSIH_textC)
# =========================

# ---- Unit 1: Library & Upload ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)

from google.colab import files
import re
import chardet
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
import joblib
from collections import Counter


# ---- Fungsi baca file KBLI ----
def read_kbli_file(path):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        with open(path, "rb") as f:
            raw_bytes = f.read()
        enc = (chardet.detect(raw_bytes)['encoding'] or 'utf-8')
        text = raw_bytes.decode(enc, errors='replace')
        text = text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
        lines = text.split('\n')
        while lines and (
            lines[0].strip().startswith('**')
            or lines[0].strip().lower().startswith('mohon')
            or lines[0].strip().lower().startswith('catatan')
        ):
            lines.pop(0)
        df = pd.read_csv(StringIO('\n'.join(lines)))
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


# ---- Fungsi util: split pemilik, label map, aturan iteratif ----
def split_business_owner(series):
    angle_pat = re.compile(r'<([^<>]*)>')
    invalid_tokens = {'', '-', '—', '.', '..', '...'}
    biz, owner_main, owner_others = [], [], []
    for val in series.fillna(''):
        s = str(val).strip()
        s = re.sub(r'\s*<\s*', '<', s)
        s = re.sub(r'\s*>\s*', '>', s)
        raw_owners = angle_pat.findall(s)
        owners = []
        for o in raw_owners:
            oc = re.sub(r'\s+', ' ', o).strip(' <>-_./|')
            if oc.upper() not in invalid_tokens and oc != '':
                owners.append(oc)
        name_raw = angle_pat.sub('', s).strip()
        name_clean = re.sub(r'\s{2,}', ' ', name_raw).strip(' -_/|')
        if not name_clean and '<' in s:
            name_clean = s.split('<', 1)[0].strip()
        biz.append(name_clean)
        owner_main.append(owners[0] if owners else '')
        owner_others.append(', '.join(owners[1:]) if len(owners) > 1 else '')
    return pd.DataFrame(
        {
            'nama_bisnis': biz,
            'nama_pemilik': owner_main,
            'nama_pemilik_lain': owner_others
        }
    )

label_map = {
 '10':'Industri Makanan','11':'Industri Minuman','12':'Industri Pengolahan Tembakau','13':'Industri Tekstil',
 '14':'Industri Pakaian Jadi','15':'Industri Kulit dan Alas Kaki','16':'Industri Kayu','17':'Industri Kertas',
 '18':'Industri Pencetakan dan Reproduksi Media Rekaman','19':'Industri Produk dari Batu Bara dan Pengilangan Minyak Bumi',
 '20':'Industri Bahan Kimia dan Barang dari Bahan Kimia','21':'Industri Farmasi, Produk Obat Kimia dan Obat Tradisional',
 '22':'Industri Karet, Barang dari Karet dan Plastik','23':'Industri Barang Galian Bukan Logam','24':'Industri Logam Dasar',
 '25':'Industri Barang dari Logam, Bukan Mesin dan Peralatannya','26':'Industri Komputer, Barang Elektronik dan Optik',
 '27':'Industri Peralatan Listrik','28':'Industri Mesin dan Perlengkapan','29':'Industri Kendaraan Bermotor, Trailer dan Semi Trailer',
 '30':'Industri Alat Angkutan Lainnya','31':'Industri Furnitur','32':'Industri Pengolahan Lainnya',
 '33':'Jasa Reparasi dan Pemasangan Mesin dan Peralatan'
}

def apply_iterative_rules_simple(df, cols, max_iters=3, conf_thr=0.70):
    txt = df[cols].fillna('').agg(' '.join, axis=1).str.upper()
    rules = [
        (r'\bKABEL\b|\bTRAFO\b|\bAMPLI(FIER)?\b|\bINVERTER\b', '27'),
        (r'\bCPU\b|\bLAPTOP\b|\bKAMERA\b|\bOPTIK\b', '26'),
        (r'\bMESIN\b|\bDINAMO\b|\bPOMPA\b|\bKOMPRESOR\b', '28'),
        (r'\bKURSI\b|\bMEJA\b|\bLEMARI\b', '31'),
        (r'\bKERTAS\b|\bAGENDA MAP\b', '17'),
        (r'\bCETAK\b|\bPERCETAKAN\b|\bUNDANGAN\b|\bSTIKER\b', '18'),
        (r'\bLEM\b|\bCAT\b|\bRESIN\b', '20'),
        (r'\bKARET\b|\bPLASTIK\b', '22'),
        (r'\bTEPUNG\b|\bSINGKONG\b|\bBERAS\b|\bKUE\b|\bTEMPE\b|\bGETHUK\b|\bTAHU\b', '10'),
        (r'\bAIR MINUM\b|\bSIRUP\b|\bMINUMAN\b', '11'),
    ]
    changed, it = True, 0
    out2 = df.copy()
    while changed and it < max_iters:
        changed, it = False, it + 1
        cand = (out2['kbli2_pred_proba'] < conf_thr)
        for pattern, target in rules:
            m = cand & txt.str.contains(pattern, regex=True, na=False) & (out2['kbli2_pred'] != target)
            if m.any():
                out2.loc[m, 'kbli2_pred'] = target
                out2.loc[m, 'kbli2_pred_label'] = out2.loc[m, 'kbli2_pred'].map(label_map)
                changed = True
    return out2


# ---- Upload file mentah KBLI ----
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
path_file = f"/content/{file_name}"
print("File yang dipakai:", path_file)


# ---- Baca dataset ----
df_raw = read_kbli_file(path_file)

print("\nPreview data mentah:")
display(df_raw.head())

print("\nInfo dataset:")
df_raw.info()

print("\nJumlah missing value per kolom:")
display(df_raw.isnull().sum())


# =========================
# PREPROCESSING
# =========================

df = df_raw.copy()

# Split r213 -> nama bisnis/pemilik (opsional)
if 'r213' in df.columns:
    sp = split_business_owner(df['r213'])
    df = pd.concat([df, sp], axis=1)

# Target kbli2_true dari r216
if 'r216_value' in df.columns:
    df['kbli2_true'] = df['r216_value'].astype(str).str.extract(r'(\d{2})')
elif 'r216_label' in df.columns:
    df['kbli2_true'] = df['r216_label'].astype(str).str.extract(r'\[(\d{2})\]')
else:
    df['kbli2_true'] = np.nan

# Fitur teks
feat_cols = [c for c in ['r215a1_label', 'r215b', 'r215d'] if c in df.columns]
if not feat_cols:
    raise ValueError("Tidak ditemukan kolom r215a1_label / r215b / r215d.")

X_all = df[feat_cols].fillna('')
y = df['kbli2_true']

print("\nFitur yang digunakan:", feat_cols)
print("Jumlah baris:", len(df))
print("Jumlah label tidak null:", y.notna().sum())
print("Jumlah kelas unik kbli2_true:", y.nunique())


# =========================
# MODELING: SMOTE TOMEK + RandomForest (FILTER KELAS JARANG)
# =========================

ct = ColumnTransformer(
    [('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), feat_cols)]
)
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight='balanced_subsample',
    n_jobs=-1
)
pipe = Pipeline([('prep', ct), ('clf', rf)])

has_y = y.notna().sum() >= 50 and y.nunique() >= 2

if has_y:
    X_t = X_all[y.notna()]
    y_t = y[y.notna()]

    # Buang kelas yang terlalu jarang, misal < 5 sampel
    vc = y_t.value_counts()
    kept_classes = vc[vc >= 5].index
    mask_kept = y_t.isin(kept_classes)
    X_t = X_t[mask_kept]
    y_t = y_t[mask_kept]

    print("\nDistribusi kelas setelah filter (<5 dibuang):")
    print(Counter(y_t))

    # SMOTE Tomek butuh fitur numerik → one-hot sementara
    ohe_tmp = OneHotEncoder(handle_unknown='ignore')
    X_num = ohe_tmp.fit_transform(X_t)

    # SMOTE Tomek dengan k_neighbors lebih kecil
    smt = SMOTETomek(random_state=42, smote_kwargs={"k_neighbors": 3})
    X_res, y_res = smt.fit_resample(X_num, y_t)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_res, y_res,
        test_size=0.2,
        random_state=42,
        stratify=y_res
    )

    # Latih pipeline pakai sampel asli dengan index selaras y_res
    pipe.fit(X_t.iloc[y_res.index], y_res)
    print("\nModel dilatih dengan SMOTE Tomek (k_neighbors=3, kelas <5 dibuang).")
else:
    pipe.fit(
        X_all,
        np.random.choice([f"{i:02d}" for i in range(10, 34)], size=len(X_all))
    )
    print("\nTidak cukup label r216, model hanya difit dummy agar bisa prediksi.")


# =========================
# PREDIKSI & PEMBAGIAN HASIL (FORMAT KOLOM FIX)
# =========================

# Prediksi awal
pred = pipe.predict(X_all)
proba = pipe.predict_proba(X_all).max(axis=1)

out = df.copy()
out['kbli2_pred'] = pred
out['kbli2_pred_label'] = out['kbli2_pred'].map(label_map)
out['kbli2_pred_proba'] = proba

# Aturan iteratif (keyword-based)
out_iter = apply_iterative_rules_simple(out, feat_cols, max_iters=3, conf_thr=0.70)

# Flag kategori C & status kesesuaian
catC = [f"{i:02d}" for i in range(10, 34)]
out_iter['is_catC_pred'] = out_iter['kbli2_pred'].isin(catC)
out_iter['is_catC_true'] = out_iter['kbli2_true'].isin(catC)
mismatch = out_iter['kbli2_true'].notna() & (out_iter['kbli2_true'] != out_iter['kbli2_pred'])

out_iter['status_kesesuaian'] = np.where(
    out_iter['is_catC_pred'] & out_iter['is_catC_true'] & (~mismatch), 'Sesuai C',
    np.where(
        ~out_iter['is_catC_pred'] & out_iter['is_catC_true'], 'True C vs Pred non-C',
        np.where(
            out_iter['is_catC_pred'] & ~out_iter['is_catC_true'],
            'True non-C vs Pred C',
            'True non-C & Pred non-C'
        )
    )
)

# Bagi menjadi tiga DataFrame
klasifikasi = out_iter.copy()
bersih = out_iter.loc[
    out_iter['is_catC_pred'] & out_iter['is_catC_true'] & (~mismatch)
].copy()
anomali = out_iter.loc[
    (~out_iter['is_catC_pred']) | (~out_iter['is_catC_true']) | mismatch
].copy()

# Kolom utama & urutan (sesuai file bersih_textC kamu)
ordered_cols = [
    "r101","r102","r103","r104","r105","r106","r107",
    "r213",
    "r215a1_label","r215b","r215d",
    "r216_label",
    "kbli2_true","kbli2_pred","kbli2_pred_label",
    "kbli2_pred_proba","status_kesesuaian",
]

for dfx in [klasifikasi, bersih, anomali]:
    for col in ordered_cols:
        if col not in dfx.columns and col in df.columns:
            dfx[col] = df[col]

bersih_cols = [c for c in ordered_cols if c in bersih.columns]
klasifikasi_cols = [c for c in ordered_cols if c in klasifikasi.columns]
anomali_cols = [c for c in ordered_cols if c in anomali.columns]

print("\nKlasifikasi (preview):")
display(klasifikasi[klasifikasi_cols].head())

print("\nBersih (preview):")
display(bersih[bersih_cols].head())

print("\nAnomali (preview):")
display(anomali[anomali_cols].head())

# Simpan hasil & model
save_prefix = "hasil_textC"
klasifikasi[klasifikasi_cols].to_csv(f"{save_prefix}_klasifikasi_r216_vs_textC.csv", index=False)
bersih[bersih_cols].to_csv(f"{save_prefix}_bersih_textC.csv", index=False)
anomali[anomali_cols].to_csv(f"{save_prefix}_anomali_kbli.csv", index=False)

joblib.dump(pipe, f"{save_prefix}_kbli2_rf.joblib")
print("\nFile CSV & model tersimpan dengan prefix:", save_prefix)
