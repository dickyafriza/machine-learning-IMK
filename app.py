import re
import os
from io import StringIO
from datetime import datetime

import chardet
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =====================#
#  CSS: Modern Minimal #
# =====================#
CUSTOM_CSS = """
<style>
/* Global */
body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #f5f7fb 0, #f9fafb 40%, #ffffff 100%);
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.0);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Cards */
.app-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    border: 1px solid rgba(15, 23, 42, 0.06);
    box-shadow: 0 14px 35px rgba(15, 23, 42, 0.04);
    margin-bottom: 1.5rem;
}

/* Titles */
.app-title {
    font-weight: 650;
    letter-spacing: -0.03em;
    font-size: 1.9rem;
    margin-bottom: 0.15rem;
}

.app-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 500;
    background: rgba(37, 99, 235, 0.06);
    color: #1d4ed8;
    border: 1px solid rgba(37, 99, 235, 0.16);
    margin-right: 0.35rem;
}

/* Metrics chip */
.chip {
    display: inline-flex;
    align-items: center;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    background: #f3f4ff;
    color: #4338ca;
    border: 1px solid rgba(79, 70, 229, 0.2);
}

/* Buttons */
.stButton>button {
    border-radius: 999px;
    padding: 0.35rem 1.2rem;
    font-weight: 500;
    font-size: 0.9rem;
    border: 1px solid rgba(37, 99, 235, 0.2);
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    color: #ffffff;
}

.stButton>button:hover {
    border-color: rgba(37, 99, 235, 0.45);
    background: linear-gradient(135deg, #1d4ed8, #4338ca);
}

/* File uploader */
[data-testid="stFileUploader"] {
    border-radius: 16px;
    padding: 0.75rem;
    background: rgba(15, 23, 42, 0.02);
    border: 1px dashed rgba(148, 163, 184, 0.8);
}

/* Tables */
.dataframe tbody tr:hover {
    background-color: #f9fafb;
}
</style>
"""

st.set_page_config(
    page_title="Klasifikasi KBLI C (2-Digit)",
    page_icon="📊",
    layout="centered"
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================#
#  Helpers             #
# =====================#
def split_business_owner(series: pd.Series) -> pd.DataFrame:
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
            "nama_bisnis": biz,
            "nama_pemilik": owner_main,
            "nama_pemilik_lain": owner_others,
        }
    )


def apply_iterative_rules_simple(df, cols, label_map, max_iters=3, conf_thr=0.70):
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
        cand = out2['kbli2_pred_proba'] < conf_thr

        for pattern, target in rules:
            m = cand & txt.str.contains(pattern, regex=True, na=False) & (
                out2['kbli2_pred'] != target
            )
            if m.any():
                out2.loc[m, 'kbli2_pred'] = target
                out2.loc[m, 'kbli2_pred_label'] = out2.loc[m, 'kbli2_pred'].map(label_map)
                changed = True

    return out2


@st.cache_data(show_spinner=False)
def convert_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =====================#
#  Layout: Header      #
# =====================#
with st.container():
    st.markdown(
        """
        <div class="app-card">
            <div class="badge">Beta • Internal tool</div>
            <div class="app-title">Klasifikasi KBLI 2-Digit (Kategori C)</div>
            <div class="app-subtitle">
                Upload data survei, latih model ringan di belakang layar, lalu unduh hasil klasifikasi, data bersih, dan anomali.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.container():
    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        uploaded_file = st.file_uploader(
            "Upload CSV (boleh kotor / hasil ekspor apa adanya)",
            type=["csv"],
        )
    with col_r:
        st.markdown(
            """
            <div class="app-card" style="padding:0.9rem 1.1rem;">
                <div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.3rem;">Panduan singkat</div>
                <ul style="padding-left:1.1rem;margin:0;font-size:0.8rem;color:#4b5563;">
                    <li>File .csv boleh mengandung catatan di atas header.</li>
                    <li>Kolom utama yang ideal: <code>r213</code>, <code>r215a1_label</code>, <code>r215b</code>, <code>r215d</code>, dan <code>r216_value/r216_label</code>.</li>
                    <li>Output akan berupa 3 file: klasifikasi lengkap, bersih (C & sesuai), dan anomali.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =====================#
#  Main Logic          #
# =====================#
if uploaded_file is not None:
    with st.spinner("Membaca & membersihkan CSV..."):
        raw_bytes = uploaded_file.read()
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

    st.markdown(
        f"""
        <div class="app-card" style="margin-top:0.5rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-size:0.95rem;font-weight:600;">Preview data</div>
                    <div style="font-size:0.8rem;color:#6b7280;">{df.shape[0]} baris • {df.shape[1]} kolom</div>
                </div>
                <div class="chip">Encoding terdeteksi: {enc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(10), use_container_width=True)

    # ---- Transform kolom r213 -> nama bisnis/pemilik ----
    if 'r213' in df.columns:
        sp = split_business_owner(df['r213'])
        df = pd.concat([df.drop(columns=['r213']), sp], axis=1)

    # ---- Target kbli2_true ----
    if 'r216_value' in df.columns:
        df['kbli2_true'] = df['r216_value'].astype(str).str.extract(r'(\d{2})')
    elif 'r216_label' in df.columns:
        df['kbli2_true'] = df['r216_label'].astype(str).str.extract(r'\[(\d{2})\]')
    else:
        df['kbli2_true'] = np.nan

    feat_cols = [c for c in ['r215a1_label', 'r215b', 'r215d'] if c in df.columns]
    if not feat_cols:
        st.error("Kolom teks r215a1_label / r215b / r215d tidak ditemukan. Tambahkan minimal satu kolom tersebut.")
        st.stop()

    X_all = df[feat_cols].fillna('')

    ct = ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), feat_cols)]
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1,
    )
    pipe = Pipeline([('prep', ct), ('clf', rf)])

    label_map = {
        '10': 'Industri Makanan',
        '11': 'Industri Minuman',
        '12': 'Industri Pengolahan Tembakau',
        '13': 'Industri Tekstil',
        '14': 'Industri Pakaian Jadi',
        '15': 'Industri Kulit dan Alas Kaki',
        '16': 'Industri Kayu',
        '17': 'Industri Kertas',
        '18': 'Industri Pencetakan dan Reproduksi Media Rekaman',
        '19': 'Industri Produk dari Batu Bara dan Pengilangan Minyak Bumi',
        '20': 'Industri Bahan Kimia dan Barang dari Bahan Kimia',
        '21': 'Industri Farmasi, Produk Obat Kimia dan Obat Tradisional',
        '22': 'Industri Karet, Barang dari Karet dan Plastik',
        '23': 'Industri Barang Galian Bukan Logam',
        '24': 'Industri Logam Dasar',
        '25': 'Industri Barang dari Logam, Bukan Mesin dan Peralatannya',
        '26': 'Industri Komputer, Barang Elektronik dan Optik',
        '27': 'Industri Peralatan Listrik',
        '28': 'Industri Mesin dan Perlengkapan',
        '29': 'Industri Kendaraan Bermotor, Trailer dan Semi Trailer',
        '30': 'Industri Alat Angkutan Lainnya',
        '31': 'Industri Furnitur',
        '32': 'Industri Pengolahan Lainnya',
        '33': 'Jasa Reparasi dan Pemasangan Mesin dan Peralatan',
    }

    has_y = df['kbli2_true'].notna().sum() >= 50 and df['kbli2_true'].nunique() >= 2

    with st.spinner("Melatih / memuat model & melakukan prediksi..."):
        metrics_text = ""
        if has_y:
            X_t = df.loc[df['kbli2_true'].notna(), feat_cols].fillna('')
            y_t = df.loc[df['kbli2_true'].notna(), 'kbli2_true']

            vc = y_t.value_counts()
            ok = y_t.isin(vc[vc >= 2].index)

            if ok.sum() >= 2 and vc[vc >= 2].shape[0] >= 2:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_t[ok],
                    y_t[ok],
                    test_size=0.2,
                    random_state=42,
                    stratify=y_t[ok],
                )
                pipe.fit(X_tr, y_tr)
                yp = pipe.predict(X_te)

                # ringkasan singkat
                report = classification_report(
                    y_te, yp, output_dict=True, zero_division=0
                )
                acc = report.get("accuracy", 0.0)
                macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
                metrics_text = f"Akurasi: {acc:.2f} • Macro F1: {macro_f1:.2f}"
            else:
                pipe.fit(X_t, y_t)
                metrics_text = "Model dilatih tanpa split (kelas jarang)."
        else:
            pipe.fit(
                X_all,
                np.random.choice(
                    [f"{i:02d}" for i in range(10, 34)], size=len(X_all)
                ),
            )
            metrics_text = "Tidak cukup label r216 untuk evaluasi, model hanya dipakai untuk prediksi awal."

        pred = pipe.predict(X_all)
        proba = pipe.predict_proba(X_all).max(axis=1)

    with st.container():
        st.markdown(
            f"""
            <div class="app-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:0.95rem;font-weight:600;">Status model</div>
                        <div style="font-size:0.8rem;color:#6b7280;">{metrics_text}</div>
                    </div>
                    <div class="chip">n = {len(df)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    out = df.copy()
    out['kbli2_pred'] = pred
    out['kbli2_pred_label'] = out['kbli2_pred'].map(label_map)
    out['kbli2_pred_proba'] = proba

    out_iter = apply_iterative_rules_simple(out, feat_cols, label_map, max_iters=3, conf_thr=0.70)

    catC = [f"{i:02d}" for i in range(10, 34)]
    out_iter['is_catC_pred'] = out_iter['kbli2_pred'].isin(catC)
    out_iter['is_catC_true'] = out_iter['kbli2_true'].isin(catC)

    mismatch = (
        out_iter['kbli2_true'].notna()
        & (out_iter['kbli2_true'] != out_iter['kbli2_pred'])
    )

    out_iter['status_kesesuaian'] = np.where(
        out_iter['is_catC_pred'] & out_iter['is_catC_true'] & (~mismatch),
        'Sesuai C',
        np.where(
            ~out_iter['is_catC_pred'] & out_iter['is_catC_true'],
            'True C vs Pred non-C',
            np.where(
                out_iter['is_catC_pred'] & ~out_iter['is_catC_true'],
                'True non-C vs Pred C',
                'True non-C & Pred non-C',
            ),
        ),
    )

    klasifikasi = out_iter.copy()
    bersih = out_iter.loc[
        out_iter['is_catC_pred'] & out_iter['is_catC_true'] & (~mismatch)
    ].copy()
    anomali = out_iter.loc[
        (~out_iter['is_catC_pred']) | (~out_iter['is_catC_true']) | mismatch
    ].copy()

    for dfx in [klasifikasi, bersih, anomali]:
        for col in ['r215a1_label', 'r215b', 'r215d', 'r216_label']:
            if col not in dfx.columns and col in df.columns:
                dfx[col] = df[col]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    id_cols = [
        c
        for c in [
            'r101',
            'r102',
            'r103',
            'r104',
            'r105',
            'r106',
            'r107',
            'r206',
            'r208',
        ]
        if c in klasifikasi.columns
    ]
    show_cols = id_cols + [
        c
        for c in [
            'nama_bisnis',
            'nama_pemilik',
            'r215a1_label',
            'r215b',
            'r215d',
            'r216_label',
            'kbli2_true',
            'kbli2_pred',
            'kbli2_pred_label',
            'kbli2_pred_proba',
            'status_kesesuaian',
        ]
        if c in klasifikasi.columns
    ]
    klasifikasi_view = klasifikasi[show_cols]

    bersih_cols = id_cols + [
        c
        for c in [
            'nama_bisnis',
            'nama_pemilik',
            'r215a1_label',
            'r215b',
            'r215d',
            'r216_label',
            'kbli2_pred',
            'kbli2_pred_label',
        ]
        if c in bersih.columns
    ]
    bersih_view = bersih[bersih_cols]

    anomali_view = anomali[show_cols]

    # =====================#
    #  Preview & download  #
    # =====================#
    st.markdown(
        """
        <div class="app-card">
            <div style="font-size:0.95rem;font-weight:600;margin-bottom:0.35rem;">Hasil ringkas</div>
            <div style="font-size:0.8rem;color:#4b5563;">
                <ul style="padding-left:1.1rem;margin:0;">
                    <li>Total baris: <b>{total}</b></li>
                    <li>Baris kategori C & sesuai label r216: <b>{ok}</b></li>
                    <li>Baris anomali / tidak konsisten: <b>{anom}</b></li>
                </ul>
            </div>
        </div>
        """.format(
            total=len(klasifikasi_view),
            ok=len(bersih_view),
            anom=len(anomali_view),
        ),
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Klasifikasi Lengkap", "Bersih (Kategori C)", "Anomali"]
    )

    with tab1:
        st.dataframe(klasifikasi_view.head(200), use_container_width=True)
        csv1 = convert_df_to_csv_bytes(klasifikasi_view)
        st.download_button(
            "📥 Unduh CSV klasifikasi",
            data=csv1,
            file_name=f"klasifikasi_r216_vs_textC_{stamp}.csv",
            mime="text/csv",
        )

    with tab2:
        st.dataframe(bersih_view.head(200), use_container_width=True)
        csv2 = convert_df_to_csv_bytes(bersih_view)
        st.download_button(
            "📥 Unduh CSV bersih (C & sesuai)",
            data=csv2,
            file_name=f"bersih_textC_{stamp}.csv",
            mime="text/csv",
        )

    with tab3:
        st.dataframe(anomali_view.head(200), use_container_width=True)
        csv3 = convert_df_to_csv_bytes(anomali_view)
        st.download_button(
            "📥 Unduh CSV anomali",
            data=csv3,
            file_name=f"anomali_kbli_{stamp}.csv",
            mime="text/csv",
        )

    # Opsional: simpan model
    with st.expander("Opsi lanjutan: simpan model Random Forest"):
        if st.button("Simpan model ke file .joblib"):
            joblib.dump(pipe, "model_kbli2_rf.joblib")
            with open("model_kbli2_rf.joblib", "rb") as f:
                st.download_button(
                    "📦 Download model_kbli2_rf.joblib",
                    data=f,
                    file_name="model_kbli2_rf.joblib",
                    mime="application/octet-stream",
                )
else:
    st.info("Mulai dengan meng-upload satu file CSV di atas.")
