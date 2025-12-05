import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from io import StringIO, BytesIO
import chardet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pickle

# ====== Config Streamlit ======
st.set_page_config(page_title="Klasifikasi KBLI Kategori C", layout="wide")
st.title("🏭 Klasifikasi KBLI Kategori C - Machine Learning Pipeline")

# ====== Sidebar: Konfigurasi ======
st.sidebar.header("⚙️ Pengaturan")
conf_threshold = st.sidebar.slider("Threshold Confidence Aturan Iteratif", 0.5, 0.95, 0.70, 0.05)
n_estimators = st.sidebar.slider("Jumlah Pohon (Random Forest)", 100, 1000, 500, 50)

# ====== Helper Functions ======
def split_business_owner(series):
    """Pisahkan Nama Bisnis<Nama Pemilik> dari r213"""
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

def apply_iterative_rules(df, cols, max_iters=3, conf_thr=0.70):
    """Aturan iteratif koreksi salah klasifikasi jelas"""
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
        changed, it = False, it+1
        cand = (out2['kbli2_pred_proba'] < conf_thr)
        for pattern, target in rules:
            m = cand & txt.str.contains(pattern, regex=True, na=False) & (out2['kbli2_pred'] != target)
            if m.any():
                out2.loc[m, 'kbli2_pred'] = target
                out2.loc[m, 'kbli2_pred_label'] = out2.loc[m, 'kbli2_pred'].map(label_map)
                changed = True
    return out2

# ====== Main UI ======
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Bersihkan", "🔬 Klasifikasi", "📊 Hasil", "⬇️ Download"])

with tab1:
    st.subheader("1️⃣ Upload CSV")
    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
    
    if uploaded_file:
        # Deteksi encoding
        raw_bytes = uploaded_file.read()
        enc = chardet.detect(raw_bytes)['encoding'] or 'utf-8'
        text = raw_bytes.decode(enc, errors='replace')
        text = text.lstrip('\ufeff').replace('\r\n','\n').replace('\r','\n')
        
        # Buang header komentar
        lines = text.split('\n')
        while lines and (lines[0].strip().startswith('**') or lines[0].strip().lower().startswith('mohon') or lines[0].strip().lower().startswith('catatan')):
            lines.pop(0)
        
        # Baca CSV
        df = pd.read_csv(StringIO('\n'.join(lines)))
        df.columns = [str(c).strip() for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()
        
        st.success(f"✅ File dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
        st.write("**Preview data:**", df.head())
        
        # Simpan ke session
        st.session_state['df'] = df.copy()
        
        # ====== Langkah 2: Pisahkan r213 ======
        st.subheader("2️⃣ Pisahkan Nama Bisnis & Pemilik (r213)")
        if 'r213' in df.columns:
            sp = split_business_owner(df['r213'])
            df = pd.concat([df.drop(columns=['r213']), sp], axis=1)
            st.success("✅ r213 berhasil dipisahkan → nama_bisnis, nama_pemilik, nama_pemilik_lain")
            st.write("**Preview split:**", df[['nama_bisnis','nama_pemilik','nama_pemilik_lain']].head())
            st.session_state['df'] = df.copy()
        else:
            st.warning("⚠️ Kolom r213 tidak ditemukan")
        
        # ====== Langkah 3: Ekstrak kbli2_true ======
        st.subheader("3️⃣ Ekstrak Target KBLI (r216)")
        if 'r216_value' in df.columns:
            df['kbli2_true'] = df['r216_value'].astype(str).str.extract(r'(\d{2})')
        elif 'r216_label' in df.columns:
            df['kbli2_true'] = df['r216_label'].astype(str).str.extract(r'\[(\d{2})\]')
        else:
            df['kbli2_true'] = np.nan
        st.success(f"✅ Ditemukan {df['kbli2_true'].notna().sum()} baris dengan target KBLI")
        st.session_state['df'] = df.copy()

with tab2:
    if 'df' not in st.session_state:
        st.info("📥 Silahkan upload file terlebih dahulu di tab 'Upload & Bersihkan'")
    else:
        df = st.session_state['df'].copy()
        st.subheader("🤖 Pelatihan & Prediksi Model")
        
        # Siapkan fitur
        feat_cols = [c for c in ['r215a1_label','r215b','r215d'] if c in df.columns]
        if not feat_cols:
            st.error("❌ Kolom r215a1_label/r215b/r215d tidak ditemukan")
        else:
            X_all = df[feat_cols].fillna('')
            
            with st.spinner("⏳ Melatih model Random Forest..."):
                # Build & train pipeline
                ct = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), feat_cols)])
                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced_subsample', n_jobs=-1)
                pipe = Pipeline([('prep', ct), ('clf', rf)])
                
                has_y = df['kbli2_true'].notna().sum() >= 50 and df['kbli2_true'].nunique() >= 2
                if has_y:
                    X_t = df.loc[df['kbli2_true'].notna(), feat_cols].fillna('')
                    y_t = df.loc[df['kbli2_true'].notna(), 'kbli2_true']
                    vc = y_t.value_counts()
                    ok = y_t.isin(vc[vc>=2].index)
                    if ok.sum() >= 2 and vc[vc>=2].shape[0] >= 2:
                        X_tr, X_te, y_tr, y_te = train_test_split(X_t[ok], y_t[ok], test_size=0.2, random_state=42, stratify=y_t[ok])
                        pipe.fit(X_tr, y_tr)
                        yp = pipe.predict(X_te)
                        acc = (yp == y_te).sum() / len(y_te)
                        st.success(f"✅ Model dilatih! Akurasi uji: {acc:.2%}")
                        st.write("**Classification Report:**")
                        st.text(classification_report(y_te, yp, zero_division=0))
                    else:
                        pipe.fit(X_t, y_t)
                        st.info("ℹ️ Model dilatih tanpa split (kelas jarang)")
                else:
                    pipe.fit(X_all, np.random.choice([f'{i:02d}' for i in range(10,34)], size=len(X_all)))
                    st.warning("⚠️ Target kurang; model fit dummy untuk inferensi saja")
            
            # Prediksi
            with st.spinner("⏳ Memprediksi seluruh data..."):
                pred = pipe.predict(X_all)
                proba = pipe.predict_proba(X_all).max(axis=1)
                
                df['kbli2_pred'] = pred
                df['kbli2_pred_label'] = df['kbli2_pred'].map(label_map)
                df['kbli2_pred_proba'] = proba
                
                # Aturan iteratif
                df = apply_iterative_rules(df, feat_cols, max_iters=3, conf_thr=conf_threshold)
                
                # Kategori C & status
                catC = [f"{i:02d}" for i in range(10,34)]
                df['is_catC_pred'] = df['kbli2_pred'].isin(catC)
                df['is_catC_true'] = df['kbli2_true'].isin(catC)
                mismatch = df['kbli2_true'].notna() & (df['kbli2_true'] != df['kbli2_pred'])
                df['status_kesesuaian'] = np.where(
                    df['is_catC_pred'] & df['is_catC_true'] & (~mismatch), 'Sesuai C',
                    np.where(~df['is_catC_pred'] & df['is_catC_true'], 'True C vs Pred non-C',
                             np.where(df['is_catC_pred'] & ~df['is_catC_true'], 'True non-C vs Pred C', 'True non-C & Pred non-C'))
                )
                
                st.session_state['df_result'] = df.copy()
                st.session_state['pipe'] = pipe
                st.success("✅ Prediksi selesai!")
                st.metric("Total Baris", len(df))
                st.metric("Kategori C", df['is_catC_pred'].sum())
                st.metric("Anomali", (~df['is_catC_pred'] | ~df['is_catC_true'] | mismatch).sum())

with tab3:
    if 'df_result' not in st.session_state:
        st.info("📊 Jalankan klasifikasi terlebih dahulu di tab 'Klasifikasi'")
    else:
        df_result = st.session_state['df_result'].copy()
        
        st.subheader("📈 Statistik & Distribusi")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediksi Kategori C", df_result['is_catC_pred'].sum())
        with col2:
            st.metric("True Kategori C", df_result['is_catC_true'].sum())
        with col3:
            st.metric("Sesuai C", (df_result['is_catC_pred'] & df_result['is_catC_true'] & 
                                    (df_result['kbli2_true'] == df_result['kbli2_pred'])).sum())
        
        st.subheader("📋 Data Bersih (Kategori C)")
        bersih = df_result.loc[df_result['is_catC_pred'] & df_result['is_catC_true'] & 
                               (df_result['kbli2_true'] == df_result['kbli2_pred'])].copy()
        st.write(f"Total baris bersih: {len(bersih)}")
        st.dataframe(bersih[['nama_bisnis','nama_pemilik','r215a1_label','kbli2_pred','kbli2_pred_label']].head(20))
        
        st.subheader("⚠️ Data Anomali")
        anomali = df_result.loc[(~df_result['is_catC_pred']) | (~df_result['is_catC_true']) | 
                                (df_result['kbli2_true'] != df_result['kbli2_pred'])].copy()
        st.write(f"Total baris anomali: {len(anomali)}")
        st.dataframe(anomali[['nama_bisnis','r215a1_label','kbli2_true','kbli2_pred','status_kesesuaian']].head(20))

with tab4:
    if 'df_result' not in st.session_state:
        st.info("⬇️ Jalankan klasifikasi terlebih dahulu di tab 'Klasifikasi'")
    else:
        st.subheader("💾 Download Hasil")
        df_result = st.session_state['df_result'].copy()
        
        # Persiapan data
        id_cols = [c for c in ['r101','r102','r103','r104','r105','r106','r107','r206','r208'] if c in df_result.columns]
        
        # Klasifikasi lengkap
        show_cols = id_cols + [c for c in ['nama_bisnis','nama_pemilik','r215a1_label','r215b','r215d','r216_label',
                                            'kbli2_true','kbli2_pred','kbli2_pred_label','kbli2_pred_proba','status_kesesuaian'] 
                               if c in df_result.columns]
        klasifikasi_df = df_result[show_cols]
        
        # Data bersih
        bersih_df = df_result.loc[df_result['is_catC_pred'] & df_result['is_catC_true'] & 
                                   (df_result['kbli2_true'] == df_result['kbli2_pred'])].copy()
        bersih_cols = id_cols + [c for c in ['nama_bisnis','nama_pemilik','r215a1_label','r215b','r215d','r216_label','kbli2_pred','kbli2_pred_label'] 
                                 if c in bersih_df.columns]
        bersih_df = bersih_df[bersih_cols]
        
        # Anomali
        anomali_df = df_result.loc[(~df_result['is_catC_pred']) | (~df_result['is_catC_true']) | 
                                    (df_result['kbli2_true'] != df_result['kbli2_pred'])].copy()
        anomali_df = anomali_df[show_cols]
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            csv_klasifikasi = klasifikasi_df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Klasifikasi Lengkap",
                data=csv_klasifikasi,
                file_name="klasifikasi.csv",
                mime="text/csv"
            )
        with col2:
            csv_bersih = bersih_df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Data Bersih (Kategori C)",
                data=csv_bersih,
                file_name="data_bersih_kbli_C.csv",
                mime="text/csv"
            )
        with col3:
            csv_anomali = anomali_df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Anomali",
                data=csv_anomali,
                file_name="anomali.csv",
                mime="text/csv"
            )
        
        st.success("✅ Klik tombol di atas untuk mengunduh file")