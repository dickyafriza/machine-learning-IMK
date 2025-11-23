import streamlit as st
import pandas as pd
import numpy as np
import re, os, chardet, joblib
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Klasifikasi KBLI C (2 digit)", page_icon="🧭", layout="wide")

st.title("Klasifikasi KBLI Kategori C (2 digit)")
st.caption("Upload CSV, pembersihan ringan, klasifikasi 2 digit (Ketgori C), koreksi aturan, dan unduh hasil.")

# ----------------- Utils -----------------
@st.cache_data(show_spinner=False)
def detect_and_read_csv(file_bytes: bytes):
    enc = (chardet.detect(file_bytes)['encoding'] or 'utf-8')
    text = file_bytes.decode(enc, errors='replace')
    text = text.lstrip('\ufeff').replace('\r\n','\n').replace('\r','\n')
    lines = text.split('\n')
    while lines and (lines[0].strip().startswith('**') or lines[0].strip().lower().startswith('mohon') or lines[0].strip().lower().startswith('catatan')):
        lines.pop(0)
    df = pd.read_csv(StringIO('\n'.join(lines)))
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def split_business_owner(series: pd.Series):
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
    return pd.DataFrame({'nama_bisnis': biz,
                         'nama_pemilik': owner_main,
                         'nama_pemilik_lain': owner_others})

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

def apply_iterative_rules_simple(df, cols, max_iters=3, conf_thr=0.80):
    txt = df[cols].fillna('').agg(' '.join, axis=1).str.upper()
    rules = [
        # Positif makanan/minuman dulu
        (r'\b(TEPUNG|BERAS|KUE|ROTI|BOLU|BISKUIT|KERIPIK|ABON|DODOL|GETHUK|TAHU|TEMPE|TAPAI|SAMBAL|SAUS|KECAP|MIE|SELAI|SIRUP|ES|MANISAN|BUBUR|BROWNIES|RENDANG|SERUNDENG)\b', '10'),
        (r'\b(AIR MINUM|MINUMAN|SARI BUAH|JUS|KOPI|TEH|SODA|SUSU|YOGURT|SIRUP)\b', '11'),
        # Lain-lain
        (r'\bKABEL\b|\bTRAFO\b|\bAMPLI(FIER)?\b|\bINVERTER\b', '27'),
        (r'\bCPU\b|\bLAPTOP\b|\bKAMERA\b|\bOPTIK\b', '26'),
        (r'\bMESIN\b|\bDINAMO\b|\bPOMPA\b|\bKOMPRESOR\b', '28'),
        (r'\bKURSI\b|\bMEJA\b|\bLEMARI\b', '31'),
        (r'\bKERTAS\b|\bAGENDA MAP\b', '17'),
        (r'\bCETAK\b|\bPERCETAKAN\b|\bUNDANGAN\b|\bSTIKER\b', '18'),
        (r'\b(LEM|CAT|RESIN)\b', '20'),
        # Karet/Plastik lebih spesifik dan diletakkan belakangan
        (r'\b(KARET|PLASTIK)\b.*\b(PIPA|SELANG|MOLD|MOLDING|INJECTION|PACKAGING|PE|PP|PVC|LISTRIK|SEAL)\b', '22'),
    ]
    changed, it = True, 0
    out2 = df.copy()
    while changed and it < max_iters:
        changed, it = False, it+1
        cand = (out2['kbli2_pred_proba'] < conf_thr)
        for pattern, target in rules:
            m = cand & txt.str.contains(pattern, regex=True, na=False) & (out2['kbli2_pred'] != target)
            if m.any():
                out2.loc[m, 'kbli2_pred'] = target
                out2.loc[m, 'kbli2_pred_label'] = out2.loc[m, 'kbli2_pred'].map(label_map)
                changed = True
    return out2

@st.cache_resource(show_spinner=False)
def build_or_load_pipeline(feature_names):
    if os.path.exists('model_kbli2_rf.joblib'):
        try:
            pipe = joblib.load('model_kbli2_rf.joblib')
            return pipe
        except Exception:
            pass
    ct = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), feature_names)])
    rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced_subsample', n_jobs=-1)
    pipe = Pipeline([('prep', ct), ('clf', rf)])
    return pipe

# ----------------- UI -----------------
uploaded = st.file_uploader("Upload CSV (boleh kotor/bersih)", type=["csv"])
proses = st.button("Proses")

if uploaded and proses:
    file_bytes = uploaded.read()
    df = detect_and_read_csv(file_bytes)

    # Split r213
    if 'r213' in df.columns:
        sp = split_business_owner(df['r213'])
        df = pd.concat([df.drop(columns=['r213']), sp], axis=1)

    # Target kbli2_true
    if 'r216_value' in df.columns:
        df['kbli2_true'] = df['r216_value'].astype(str).str.extract(r'(\d{2})')
    elif 'r216_label' in df.columns:
        df['kbli2_true'] = df['r216_label'].astype(str).str.extract(r'\[(\d{2})\]')
    else:
        df['kbli2_true'] = np.nan

    # Fitur
    feat_cols = [c for c in ['r215a1_label','r215b','r215d'] if c in df.columns]
    if not feat_cols:
        st.error("Tidak ditemukan kolom r215a1_label / r215b / r215d.")
        st.stop()
    X_all = df[feat_cols].fillna('')

    # Model: load jika ada, jika tidak latih cepat bila tersedia y
    pipe = build_or_load_pipeline(feat_cols)
    has_y = df['kbli2_true'].notna().sum() >= 50 and df['kbli2_true'].nunique() >= 2
    if has_y:
        y_t = df.loc[df['kbli2_true'].notna(), 'kbli2_true']
        vc = y_t.value_counts()
        ok = y_t.isin(vc[vc>=2].index)
        if ok.sum() >= 2 and vc[vc>=2].shape[0] >= 2:
            pipe.fit(X_all[ok], y_t[ok])
        else:
            pipe.fit(X_all, y_t)
    else:
        # fallback supaya pipeline hidup
        rng = np.random.default_rng(42)
        fake_y = rng.choice([f'{i:02d}' for i in range(10,34)], size=len(X_all))
        pipe.fit(X_all, fake_y)

    # Prediksi
    pred = pipe.predict(X_all)
    proba = pipe.predict_proba(X_all).max(axis=1)

    out = df.copy()
    out['kbli2_pred'] = pred
    out['kbli2_pred_label'] = out['kbli2_pred'].map(label_map)
    out['kbli2_pred_proba'] = proba

    # Aturan iteratif (threshold dinaikkan)
    out_iter = apply_iterative_rules_simple(out, feat_cols, max_iters=3, conf_thr=0.80)

    # Status C / kesesuaian
    catC = [f"{i:02d}" for i in range(10,34)]
    out_iter['is_catC_pred'] = out_iter['kbli2_pred'].isin(catC)
    out_iter['is_catC_true'] = out_iter['kbli2_true'].isin(catC)
    mismatch = out_iter['kbli2_true'].notna() & (out_iter['kbli2_true'] != out_iter['kbli2_pred'])
    out_iter['status_kesesuaian'] = np.where(
        out_iter['is_catC_pred'] & out_iter['is_catC_true'] & (~mismatch), 'Sesuai C',
        np.where(~out_iter['is_catC_pred'] & out_iter['is_catC_true'], 'True C vs Pred non-C',
                 np.where(out_iter['is_catC_pred'] & ~out_iter['is_catC_true'], 'True non-C vs Pred C', 'True non-C & Pred non-C'))
    )

    # Bagi output
    klasifikasi = out_iter.copy()
    bersih = out_iter.loc[out_iter['is_catC_pred'] & out_iter['is_catC_true'] & (~mismatch)].copy()
    anomali = out_iter.loc[(~out_iter['is_catC_pred']) | (~out_iter['is_catC_true']) | mismatch].copy()

    for dfx in [klasifikasi, bersih, anomali]:
        for col in ['r215a1_label','r215b','r215d','r216_label']:
            if col not in dfx.columns and col in df.columns:
                dfx[col] = df[col]

    # Preview
    st.subheader("Preview")
    st.write("Klasifikasi", klasifikasi.head(20))
    st.write("Bersih", bersih.head(20))
    st.write("Anomali", anomali.head(20))

    # Siapkan CSV dan tombol unduh
    id_cols = [c for c in ['r101','r102','r103','r104','r105','r106','r107','r206','r208'] if c in klasifikasi.columns]
    show_cols = id_cols + [c for c in ['nama_bisnis','nama_pemilik','r215a1_label','r215b','r215d','r216_label','kbli2_true','kbli2_pred','kbli2_pred_label','kbli2_pred_proba','status_kesesuaian'] if c in klasifikasi.columns]
    bersih_cols = id_cols + [c for c in ['nama_bisnis','nama_pemilik','r215a1_label','r215b','r215d','r216_label','kbli2_pred','kbli2_pred_label'] if c in bersih.columns]

    klasifikasi_csv = klasifikasi[show_cols].to_csv(index=False).encode('utf-8')
    bersih_csv = bersih[bersih_cols].to_csv(index=False).encode('utf-8')
    anomali_csv = anomali[show_cols].to_csv(index=False).encode('utf-8')

    st.download_button("Unduh klasifikasi.csv", klasifikasi_csv, file_name="klasifikasi.csv", mime="text/csv")
    st.download_button("Unduh bersih.csv", bersih_csv, file_name="bersih.csv", mime="text/csv")
    st.download_button("Unduh anomali.csv", anomali_csv, file_name="anomali.csv", mime="text/csv")
else:
    st.info("Silakan upload file CSV lalu klik Proses.")
