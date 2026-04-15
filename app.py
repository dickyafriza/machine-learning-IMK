import streamlit as st
import pandas as pd
import numpy as np
import re
import chardet
from io import StringIO

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import joblib

st.set_page_config(page_title="Klasifikasi KBLI 2 Digit", layout="wide")
st.title("Klasifikasi KBLI 2 Digit dari Teks")

st.write(
    "Upload file mentah (CSV/Excel) dari FASIH atau VIMK25. "
    "Sistem akan otomatis mendeteksi format kolom yang digunakan."
)

uploaded_file = st.file_uploader(
    "Upload file CSV atau Excel",
    type=["csv", "xlsx", "xls"]
)

# ========= Fungsi util =========

def split_business_owner(series):
    angle_pat = re.compile(r'<([^<>]*)>')  # termasuk kosong
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
        (r'\bKURSI\b|\bMEJA\b|\bLEMARI\b|\bDIPAN\b|\bSOFA\b', '31'),
        (r'\bKERTAS\b|\bAGENDA MAP\b', '17'),
        (r'\bCETAK\b|\bPERCETAKAN\b|\bUNDANGAN\b|\bSTIKER\b|\bSABLON\b', '18'),
        (r'\bLEM\b|\bCAT\b|\bRESIN\b', '20'),
        (r'\bKARET\b|\bPLASTIK\b', '22'),
        (r'\bTEPUNG\b|\bSINGKONG\b|\bBERAS\b|\bKUE\b|\bTEMPE\b|\bGETHUK\b|\bTAHU\b', '10'),
        (r'\bAIR MINUM\b|\bSIRUP\b|\bMINUMAN\b|\bAIR ISI ULANG\b', '11'),
        (r'\bBATA\b|\bBATU BATA\b|\bGENTENG\b|\bTEGEL\b|\bPAVING\b', '23'),
        (r'\bKERAMIK\b|\bGRANIT\b', '23'),
        (r'\bKAOS\b|\bT-SHIRT\b|\bKOSTUM\b', '14'),
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

# cek apakah URL mengarah ke file gambar (foto produk)
def is_image_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    s = url.strip()
    if s == '' or s.lower() == 'nan':
        return False
    s_low = s.lower()
    base_no_query = s_low.split('?', 1)[0]
    if 'bucket1.cloud.bps.go.id' in s_low and ('r215c' in s_low or 'r316c' in s_low):
        return True
    if 'drive.google.com' in s_low and '/file/' in s_low:
        return True
    img_ext = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    if base_no_query.endswith(img_ext):
        return True
    return False

# Deteksi format data dan mapping kolom
def detect_and_map_columns(df):
    """Deteksi apakah data dari FASIH atau VIMK25, lalu kembalikan mapping kolom."""
    # VIMK25 punya r316a_label/r316a_value, r316b, r316d, v317_lab/r317_label
    is_vimk = any(c in df.columns for c in [
        'r316a_label', 'r316a_value', 'r316b', 'v317_lab', 'r317_label'
    ])
    # FASIH punya r215a1_label, r215b, r215d, r216_label
    is_fasih = 'r215a1_label' in df.columns or 'r215b' in df.columns

    if is_vimk:
        # Kolom fitur teks — coba semua kemungkinan nama kolom VIMK25
        vimk_feat_candidates = [
            'r316a_label', 'r316a_value', 'r316a_lain',
            'r316b', 'r316d'
        ]
        feat_cols = [c for c in vimk_feat_candidates if c in df.columns]

        # Kolom KBLI — bisa v317_lab atau r317_label
        kbli_label = None
        for candidate in ['v317_lab', 'r317_label']:
            if candidate in df.columns:
                kbli_label = candidate
                break

        return {
            'format': 'VIMK25',
            'nama_usaha': 'r314',
            'feat_cols': feat_cols,
            'foto_url': 'r316c_url',
            'kbli_label': kbli_label,
            'kbli_value': None,
            'wilayah_cols': ['prov', 'kab', 'kec', 'des', 'bs', 'sbs'],
        }
    else:
        return {
            'format': 'FASIH',
            'nama_usaha': 'r213',
            'feat_cols': [c for c in ['r215a1_label', 'r215b', 'r215d'] if c in df.columns],
            'foto_url': 'r215c_url',
            'kbli_label': 'r216_label',
            'kbli_value': 'r216_value',
            'wilayah_cols': ['r101', 'r102', 'r103', 'r104', 'r105', 'r106', 'r107'],
        }

# Pemeriksaan logis sesuai pedoman VIMK25-L2 (PPT)
def check_vimk_rules(df, col_map):
    """Pemeriksaan logis berdasarkan pedoman pemeriksaan VIMK25-L2."""
    issues = []
    for i, row in df.iterrows():
        r = []

        # --- Rule 1: R314 (Nama Usaha) wajib terisi ---
        nama_col = col_map['nama_usaha']
        if nama_col in df.columns:
            val = str(row.get(nama_col, '')).strip()
            if val in ('', 'nan', '-', '.'):
                r.append("R314 Nama Usaha kosong")

        # --- Rule 2: R304 – Penggunaan bangunan ---
        if 'r304_value' in df.columns:
            r304 = str(row.get('r304_value', '')).strip()
            # r304_value = 1 atau 2 (khusus/campuran) -> harus ada usaha
            if r304 not in ('', 'nan') and r304 in ('1', '2'):
                # Jika r304 = 1 atau 2 tapi r308 bukan 1, berarti inkonsisten
                if 'r308_value' in df.columns:
                    r308 = str(row.get('r308_value', '')).strip()
                    if r308 != '1':
                        r.append("R304 khusus/campuran tapi R308 bukan ada usaha IMK")

        # --- Rule 3: R316a/R316b (Kegiatan Utama) wajib terisi ---
        for fc in col_map['feat_cols']:
            pass  # dicek di level fitur teks
        feat_text = ' '.join(str(row.get(fc, '')) for fc in col_map['feat_cols']).strip()
        if feat_text in ('', 'nan') or len(feat_text) < 3:
            r.append("Deskripsi kegiatan/produk kosong atau terlalu pendek")

        # --- Rule 4: KBLI (R317/R216) wajib terisi ---
        kbli_col = col_map['kbli_label']
        if kbli_col and kbli_col in df.columns:
            kbli_val = str(row.get(kbli_col, '')).strip()
            if kbli_val in ('', 'nan', '-'):
                r.append("Kode KBLI kosong")

        # --- Rule 5: R310 – Jumlah ART ---
        if 'r310_value' in df.columns:
            r310 = str(row.get('r310_value', '')).strip()
            if r310 not in ('', 'nan'):
                try:
                    r310_int = int(float(r310))
                    if r310_int <= 0:
                        r.append("R310 jumlah ART tidak wajar (<=0)")
                except (ValueError, OverflowError):
                    pass

        # --- Rule 6: R311 – Jumlah kegiatan usaha ---
        if 'r311_value' in df.columns:
            r311 = str(row.get('r311_value', '')).strip()
            if r311 not in ('', 'nan'):
                try:
                    r311_int = int(float(r311))
                    if r311_int <= 0:
                        r.append("R311 jumlah kegiatan usaha tidak wajar (<=0)")
                except (ValueError, OverflowError):
                    pass

        issues.append("; ".join(r) if r else "")
    return issues

# ========= Proses utama =========

if uploaded_file is not None:
    raw_name = uploaded_file.name
    raw_bytes = uploaded_file.getvalue()

    # Baca Excel vs CSV
    if raw_name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
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

    # Normalisasi kolom & strip spasi
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # Deteksi format data
    col_map = detect_and_map_columns(df)
    st.info(f"Format data terdeteksi: **{col_map['format']}**")

    # Filter hanya data yang "SUBMITTED BY Pencacah"
    if 'assignment_status_alias' in df.columns:
        total_sebelum = len(df)
        df = df[
            df['assignment_status_alias'].str.upper() == 'SUBMITTED BY PENCACAH'
        ].reset_index(drop=True)
        total_sesudah = len(df)
        st.info(
            f"Filter: hanya data **SUBMITTED BY Pencacah**. "
            f"{total_sesudah} dari {total_sebelum} baris diproses."
        )
        if total_sesudah == 0:
            st.warning("Tidak ada data dengan status 'SUBMITTED BY Pencacah'.")
            st.stop()
    else:
        st.warning(
            "Kolom `assignment_status_alias` tidak ditemukan. "
            "Semua data akan diproses tanpa filter status."
        )

    st.subheader("Preview data mentah (setelah filter)")
    st.dataframe(df.head())

    # Split nama usaha -> nama_bisnis / pemilik
    nama_col = col_map['nama_usaha']
    if nama_col in df.columns:
        sp = split_business_owner(df[nama_col])
        df = pd.concat([df, sp], axis=1)

    # Target kbli2_true
    feat_cols = col_map['feat_cols']
    kbli_value_col = col_map.get('kbli_value')
    kbli_label_col = col_map.get('kbli_label')

    if kbli_value_col and kbli_value_col in df.columns:
        df['kbli2_true'] = df[kbli_value_col].astype(str).str.extract(r'(\d{2})')
    elif kbli_label_col and kbli_label_col in df.columns:
        # Coba extract dari format "[XX] ..." atau langsung 2 digit awal
        kbli_str = df[kbli_label_col].astype(str)
        extracted = kbli_str.str.extract(r'\[(\d{2})\]')
        # Jika format bracket tidak match, coba extract 2 digit pertama langsung
        mask_na = extracted[0].isna()
        if mask_na.any():
            extracted.loc[mask_na, 0] = kbli_str[mask_na].str.extract(r'(\d{2})')[0]
        df['kbli2_true'] = extracted[0]
    else:
        df['kbli2_true'] = np.nan

    # Fitur teks
    if not feat_cols:
        st.error("Tidak ditemukan kolom fitur teks (r215a1_label/r215b/r215d atau r316a_label/r316b/r316d).")
        st.stop()

    df['text_all'] = df[feat_cols].fillna('').agg(' '.join, axis=1)
    X_all = df[['text_all']].copy()

    # Dynamically set min_df based on dataset size
    effective_min_df = max(1, min(3, len(X_all) // 10))

    # --------- Pipeline dasar ---------
    ct = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=effective_min_df
            ), 'text_all')
        ],
        remainder='drop'
    )

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([('prep', ct), ('clf', rf)])
    # ----------------------------------

    # --------- GridSearchCV ----------
    param_grid = {
        'clf__n_estimators': [300, 600, 900],
        'clf__max_depth': [None, 20, 40],
        'clf__min_samples_split': [2, 5],
        'clf__min_samples_leaf': [1, 2],
        'clf__class_weight': ['balanced']
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    # ---------------------------------

    has_y = df['kbli2_true'].notna().sum() >= 50 and df['kbli2_true'].nunique() >= 2

    if has_y:
        X_t = df.loc[df['kbli2_true'].notna(), ['text_all']]
        y_t = df.loc[df['kbli2_true'].notna(), 'kbli2_true']

        vc = y_t.value_counts()
        ok = y_t.isin(vc[vc >= 2].index)

        if ok.sum() >= 2 and vc[vc >= 2].shape[0] >= 2:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_t[ok], y_t[ok],
                test_size=0.2,
                random_state=42,
                stratify=y_t[ok]
            )

            grid.fit(X_tr, y_tr)

            st.success("Model dilatih dengan GridSearchCV (TF-IDF + RandomForest).")
            st.write("Best params:", grid.best_params_)
            st.write("Best CV score:", f"{grid.best_score_:.3f}")

            best_model = grid.best_estimator_
        else:
            pipe.set_params(clf__n_estimators=800, clf__class_weight='balanced')
            pipe.fit(X_t, y_t)
            best_model = pipe
            st.warning("Model dilatih tanpa split/grid (kelas jarang).")
    else:
        pipe.set_params(clf__n_estimators=100, clf__class_weight='balanced',
                        prep__text__min_df=1)
        pipe.fit(
            X_all,
            np.random.choice([f"{i:02d}" for i in range(10, 34)], size=len(X_all))
        )
        best_model = pipe
        st.info("Tidak cukup label KBLI, model hanya difit dummy agar bisa prediksi.")

    # Prediksi + proba dengan best_model
    pred = best_model.predict(X_all)
    if hasattr(best_model.named_steps['clf'], "predict_proba"):
        proba = best_model.predict_proba(X_all).max(axis=1)
    else:
        proba = np.ones(len(X_all))

    out = df.copy()
    out['kbli2_pred'] = pred
    out['kbli2_pred_label'] = out['kbli2_pred'].map(label_map)
    out['kbli2_pred_proba'] = proba

    # Aturan iteratif pakai feat_cols
    out_iter = apply_iterative_rules_simple(out, feat_cols, max_iters=3, conf_thr=0.70)

    # Kategori C dan status kesesuaian
    catC = [f"{i:02d}" for i in range(10, 34)]
    out_iter['is_catC_pred'] = out_iter['kbli2_pred'].isin(catC)
    out_iter['is_catC_true'] = out_iter['kbli2_true'].isin(catC)

    # mismatch hanya jika ada ground truth DAN berbeda dengan prediksi
    has_true = out_iter['kbli2_true'].notna()
    mismatch = has_true & (out_iter['kbli2_true'] != out_iter['kbli2_pred'])

    out_iter['status_kesesuaian'] = np.where(
        out_iter['is_catC_pred'] & has_true & out_iter['is_catC_true'] & (~mismatch),
        'Sesuai C',
        np.where(
            out_iter['is_catC_pred'] & (~has_true),
            'Pred C (belum ada label)',
            np.where(
                ~out_iter['is_catC_pred'] & out_iter['is_catC_true'],
                'True C vs Pred non-C',
                np.where(
                    out_iter['is_catC_pred'] & has_true & ~out_iter['is_catC_true'],
                    'True non-C vs Pred C',
                    np.where(
                        ~out_iter['is_catC_pred'] & (~has_true),
                        'Pred non-C (belum ada label)',
                        'Lainnya'
                    )
                )
            )
        )
    )

    # Pemeriksaan logis sesuai pedoman VIMK25-L2
    out_iter['pemeriksaan_logis'] = check_vimk_rules(out_iter, col_map)

    # =====  Bagi output =====
    klasifikasi = out_iter.copy()

    # Data BERSIH:
    #   - Prediksi masuk kategori C (10-33)
    #   - Jika ada ground truth (kbli2_true), harus sesuai (tidak mismatch)
    #   - Lolos semua pemeriksaan logis VIMK25-L2
    #   - Tanpa gambar tetap dianggap bersih
    bersih = out_iter.loc[
        out_iter['is_catC_pred']
        & (~mismatch)
        & (out_iter['pemeriksaan_logis'] == '')  # lolos semua pemeriksaan logis
    ].copy()

    # Data ANOMALI:
    #   - Prediksi bukan kategori C, ATAU
    #   - Ada mismatch prediksi vs ground truth, ATAU
    #   - Gagal pemeriksaan logis VIMK25-L2
    anomali = out_iter.loc[
        (~out_iter['is_catC_pred'])
        | mismatch
        | (out_iter['pemeriksaan_logis'] != '')  # ada masalah pemeriksaan logis
    ].copy()

    # Tambah alasan anomali
    reasons = []
    for i, row in anomali.iterrows():
        r = []
        if row.get('kbli2_true') in catC and row.get('kbli2_pred') not in catC:
            r.append("True C vs Pred non-C")
        elif row.get('kbli2_true') not in catC and row.get('kbli2_pred') in catC:
            r.append("True non-C vs Pred C")
        if pd.isna(row.get('kbli2_true')):
            r.append("KBLI kosong")
        # Tambah hasil pemeriksaan logis
        logis = str(row.get('pemeriksaan_logis', ''))
        if logis and logis != 'nan':
            r.append(logis)
        reasons.append("; ".join(r) if r else "Periksa manual")
    anomali['alasan_anomali'] = reasons

    # =====  Kolom & urutan =====
    # Kolom yang akan ditampilkan (adaptif sesuai format)
    foto_col = col_map['foto_url']
    kbli_col_display = col_map['kbli_label']
    wilayah_cols = [c for c in col_map['wilayah_cols'] if c in df.columns]

    base_cols_ba = (
        wilayah_cols
        + [nama_col]
        + feat_cols
        + ([kbli_col_display] if kbli_col_display and kbli_col_display in df.columns else [])
        + ['kbli2_true', 'kbli2_pred', 'kbli2_pred_label',
           'kbli2_pred_proba', 'status_kesesuaian', 'pemeriksaan_logis']
        + ([foto_col] if foto_col and foto_col in df.columns else [])
    )

    for dfx in [bersih, anomali]:
        for col in base_cols_ba:
            if col not in dfx.columns and col in df.columns:
                dfx[col] = df[col]

    ordered_cols = base_cols_ba.copy()

    klasifikasi_cols = [c for c in ordered_cols if c in klasifikasi.columns]
    bersih_cols      = [c for c in ordered_cols if c in bersih.columns]
    anomali_cols     = [c for c in ordered_cols if c in anomali.columns] + ['alasan_anomali']

    def view_cols(dfv, cols):
        if foto_col and foto_col in cols:
            if dfv[foto_col].astype(str).str.strip().eq('').all():
                return [c for c in cols if c != foto_col]
        return cols

    klasifikasi_view = view_cols(klasifikasi, klasifikasi_cols)
    bersih_view      = view_cols(bersih, bersih_cols)
    anomali_view     = view_cols(anomali, anomali_cols)

    # =====  Ringkasan =====
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", len(klasifikasi))
    with col2:
        st.metric("Data Bersih", len(bersih))
    with col3:
        st.metric("Data Anomali", len(anomali))

    if 'status_kesesuaian' in klasifikasi.columns:
        total_labeled = (klasifikasi['kbli2_true'].notna()).sum()
        sesuai_c = (klasifikasi['status_kesesuaian'] == 'Sesuai C').sum()
        if total_labeled > 0:
            akurasi = sesuai_c / total_labeled
            st.metric("Proporsi 'Sesuai C' (KBLI 2 digit)", f"{akurasi:.1%}")

    # =====  Tampilkan di halaman =====
    st.subheader("Data klasifikasi (lengkap)")
    st.dataframe(klasifikasi[klasifikasi_view].head(20))

    st.subheader("Data bersih (Prediksi Kategori C + lolos pemeriksaan logis)")
    if len(bersih) > 0:
        st.dataframe(bersih[bersih_view].head(20))
    else:
        st.warning("Tidak ada data yang memenuhi kriteria bersih.")

    st.subheader("Data anomali (non‑C / mismatch / gagal pemeriksaan logis)")
    if len(anomali) > 0:
        st.dataframe(anomali[anomali_view].head(20))
    else:
        st.success("Tidak ada data anomali ditemukan.")

    # =====  Download CSV =====
    klasifikasi_csv = klasifikasi[klasifikasi_cols].to_csv(index=False).encode("utf-8")
    bersih_csv      = bersih[bersih_cols].to_csv(index=False).encode("utf-8")
    anomali_csv     = anomali[anomali_cols].to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download klasifikasi.csv",
        data=klasifikasi_csv,
        file_name="klasifikasi_kbli.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download data_bersih.csv",
        data=bersih_csv,
        file_name="data_bersih.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download data_anomali.csv",
        data=anomali_csv,
        file_name="data_anomali.csv",
        mime="text/csv"
    )

    # Opsional: simpan model
    if st.checkbox("Simpan model ke file .joblib di server"):
        joblib.dump(best_model, "model_kbli2_rf_tfidf_grid.joblib")
        st.success("Model disimpan sebagai model_kbli2_rf_tfidf_grid.joblib")