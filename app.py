# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import chardet
# from io import StringIO
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# st.set_page_config(page_title="Klasifikasi KBLI 2 Digit", layout="wide")
# st.title("Klasifikasi KBLI 2 Digit dari Teks")

# st.write(
#     "Upload file mentah (CSV/Excel) berisi minimal kolom r101–r107, r213, "
#     "r215a1_label / r215b / r215d, r216_value / r216_label, dan r215c_url untuk gambar."
# )

# uploaded_file = st.file_uploader(
#     "Upload file CSV atau Excel",
#     type=["csv", "xlsx", "xls"]
# )

# # ========= Fungsi util =========

# def split_business_owner(series):
#     angle_pat = re.compile(r'<([^<>]*)>')  # termasuk kosong
#     invalid_tokens = {'', '-', '—', '.', '..', '...'}
#     biz, owner_main, owner_others = [], [], []
#     for val in series.fillna(''):
#         s = str(val).strip()
#         s = re.sub(r'\s*<\s*', '<', s)
#         s = re.sub(r'\s*>\s*', '>', s)
#         raw_owners = angle_pat.findall(s)
#         owners = []
#         for o in raw_owners:
#             oc = re.sub(r'\s+', ' ', o).strip(' <>-_./|')
#             if oc.upper() not in invalid_tokens and oc != '':
#                 owners.append(oc)
#         name_raw = angle_pat.sub('', s).strip()
#         name_clean = re.sub(r'\s{2,}', ' ', name_raw).strip(' -_/|')
#         if not name_clean and '<' in s:
#             name_clean = s.split('<', 1)[0].strip()
#         biz.append(name_clean)
#         owner_main.append(owners[0] if owners else '')
#         owner_others.append(', '.join(owners[1:]) if len(owners) > 1 else '')
#     return pd.DataFrame(
#         {
#             'nama_bisnis': biz,
#             'nama_pemilik': owner_main,
#             'nama_pemilik_lain': owner_others
#         }
#     )

# label_map = {
#  '10':'Industri Makanan','11':'Industri Minuman','12':'Industri Pengolahan Tembakau','13':'Industri Tekstil',
#  '14':'Industri Pakaian Jadi','15':'Industri Kulit dan Alas Kaki','16':'Industri Kayu','17':'Industri Kertas',
#  '18':'Industri Pencetakan dan Reproduksi Media Rekaman','19':'Industri Produk dari Batu Bara dan Pengilangan Minyak Bumi',
#  '20':'Industri Bahan Kimia dan Barang dari Bahan Kimia','21':'Industri Farmasi, Produk Obat Kimia dan Obat Tradisional',
#  '22':'Industri Karet, Barang dari Karet dan Plastik','23':'Industri Barang Galian Bukan Logam','24':'Industri Logam Dasar',
#  '25':'Industri Barang dari Logam, Bukan Mesin dan Peralatannya','26':'Industri Komputer, Barang Elektronik dan Optik',
#  '27':'Industri Peralatan Listrik','28':'Industri Mesin dan Perlengkapan','29':'Industri Kendaraan Bermotor, Trailer dan Semi Trailer',
#  '30':'Industri Alat Angkutan Lainnya','31':'Industri Furnitur','32':'Industri Pengolahan Lainnya',
#  '33':'Jasa Reparasi dan Pemasangan Mesin dan Peralatan'
# }

# def apply_iterative_rules_simple(df, cols, max_iters=3, conf_thr=0.70):
#     txt = df[cols].fillna('').agg(' '.join, axis=1).str.upper()
#     rules = [
#         (r'\bKABEL\b|\bTRAFO\b|\bAMPLI(FIER)?\b|\bINVERTER\b', '27'),
#         (r'\bCPU\b|\bLAPTOP\b|\bKAMERA\b|\bOPTIK\b', '26'),
#         (r'\bMESIN\b|\bDINAMO\b|\bPOMPA\b|\bKOMPRESOR\b', '28'),
#         (r'\bKURSI\b|\bMEJA\b|\bLEMARI\b', '31'),
#         (r'\bKERTAS\b|\bAGENDA MAP\b', '17'),
#         (r'\bCETAK\b|\bPERCETAKAN\b|\bUNDANGAN\b|\bSTIKER\b', '18'),
#         (r'\bLEM\b|\bCAT\b|\bRESIN\b', '20'),
#         (r'\bKARET\b|\bPLASTIK\b', '22'),
#         (r'\bTEPUNG\b|\bSINGKONG\b|\bBERAS\b|\bKUE\b|\bTEMPE\b|\bGETHUK\b|\bTAHU\b', '10'),
#         (r'\bAIR MINUM\b|\bSIRUP\b|\bMINUMAN\b', '11'),
#     ]
#     changed, it = True, 0
#     out2 = df.copy()
#     while changed and it < max_iters:
#         changed, it = False, it + 1
#         cand = (out2['kbli2_pred_proba'] < conf_thr)
#         for pattern, target in rules:
#             m = cand & txt.str.contains(pattern, regex=True, na=False) & (out2['kbli2_pred'] != target)
#             if m.any():
#                 out2.loc[m, 'kbli2_pred'] = target
#                 out2.loc[m, 'kbli2_pred_label'] = out2.loc[m, 'kbli2_pred'].map(label_map)
#                 changed = True
#     return out2

# # ========= Proses utama =========

# if uploaded_file is not None:
#     raw_name = uploaded_file.name
#     raw_bytes = uploaded_file.getvalue()

#     # Baca Excel vs CSV (dengan pembersihan header untuk CSV)
#     if raw_name.lower().endswith((".xlsx", ".xls")):
#         df = pd.read_excel(uploaded_file)  # [web:43]
#     else:
#         enc = (chardet.detect(raw_bytes)['encoding'] or 'utf-8')
#         text = raw_bytes.decode(enc, errors='replace')
#         text = text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
#         lines = text.split('\n')
#         while lines and (
#             lines[0].strip().startswith('**')
#             or lines[0].strip().lower().startswith('mohon')
#             or lines[0].strip().lower().startswith('catatan')
#         ):
#             lines.pop(0)
#         df = pd.read_csv(StringIO('\n'.join(lines)))

#     # Normalisasi kolom & strip spasi
#     df.columns = [str(c).strip() for c in df.columns]
#     for c in df.columns:
#         if df[c].dtype == object:
#             df[c] = df[c].astype(str).str.strip()

#     st.subheader("Preview data mentah")
#     st.dataframe(df.head())

#     # Split r213 -> nama_bisnis / pemilik
#     if 'r213' in df.columns:
#         sp = split_business_owner(df['r213'])
#         df = pd.concat([df, sp], axis=1)

#     # Target kbli2_true dari r216
#     if 'r216_value' in df.columns:
#         df['kbli2_true'] = df['r216_value'].astype(str).str.extract(r'(\d{2})')
#     elif 'r216_label' in df.columns:
#         df['kbli2_true'] = df['r216_label'].astype(str).str.extract(r'\[(\d{2})\]')
#     else:
#         df['kbli2_true'] = np.nan

#     # Fitur teks
#     feat_cols = [c for c in ['r215a1_label', 'r215b', 'r215d'] if c in df.columns]
#     if not feat_cols:
#         st.error("Tidak ditemukan kolom r215a1_label / r215b / r215d.")
#         st.stop()
#     X_all = df[feat_cols].fillna('')

#     # Model
#     ct = ColumnTransformer(
#         [('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), feat_cols)]
#     )
#     rf = RandomForestClassifier(
#         n_estimators=500,
#         random_state=42,
#         class_weight='balanced_subsample',
#         n_jobs=-1
#     )
#     pipe = Pipeline([('prep', ct), ('clf', rf)])

#     has_y = df['kbli2_true'].notna().sum() >= 50 and df['kbli2_true'].nunique() >= 2

#     if has_y:
#         X_t = df.loc[df['kbli2_true'].notna(), feat_cols].fillna('')
#         y_t = df.loc[df['kbli2_true'].notna(), 'kbli2_true']
#         vc = y_t.value_counts()
#         ok = y_t.isin(vc[vc >= 2].index)
#         if ok.sum() >= 2 and vc[vc >= 2].shape[0] >= 2:
#             X_tr, X_te, y_tr, y_te = train_test_split(
#                 X_t[ok], y_t[ok],
#                 test_size=0.2,
#                 random_state=42,
#                 stratify=y_t[ok]
#             )
#             pipe.fit(X_tr, y_tr)
#             st.success("Model dilatih dengan train/test split.")
#         else:
#             pipe.fit(X_t, y_t)
#             st.warning("Model dilatih tanpa split (kelas jarang).")
#     else:
#         pipe.fit(
#             X_all,
#             np.random.choice([f"{i:02d}" for i in range(10, 34)], size=len(X_all))
#         )
#         st.info("Tidak cukup label r216, model hanya difit dummy agar bisa prediksi.")

#     # Prediksi + label kategori
#     pred = pipe.predict(X_all)
#     proba = pipe.predict_proba(X_all).max(axis=1)

#     out = df.copy()
#     out['kbli2_pred'] = pred
#     out['kbli2_pred_label'] = out['kbli2_pred'].map(label_map)
#     out['kbli2_pred_proba'] = proba

#     # Aturan iteratif
#     out_iter = apply_iterative_rules_simple(out, feat_cols, max_iters=3, conf_thr=0.70)

#     # Kategori C dan status
#     catC = [f"{i:02d}" for i in range(10, 34)]
#     out_iter['is_catC_pred'] = out_iter['kbli2_pred'].isin(catC)
#     out_iter['is_catC_true'] = out_iter['kbli2_true'].isin(catC)
#     mismatch = out_iter['kbli2_true'].notna() & (out_iter['kbli2_true'] != out_iter['kbli2_pred'])
#     out_iter['status_kesesuaian'] = np.where(
#         out_iter['is_catC_pred'] & out_iter['is_catC_true'] & (~mismatch), 'Sesuai C',
#         np.where(
#             ~out_iter['is_catC_pred'] & out_iter['is_catC_true'], 'True C vs Pred non-C',
#             np.where(
#                 out_iter['is_catC_pred'] & ~out_iter['is_catC_true'],
#                 'True non-C vs Pred C',
#                 'True non-C & Pred non-C'
#             )
#         )
#     )

#     # Flag baris yang tidak punya gambar
#     if 'r215c_url' in out_iter.columns:
#         no_image = out_iter['r215c_url'].isna() | (out_iter['r215c_url'].astype(str).str.strip() == '')
#     else:
#         no_image = pd.Series(False, index=out_iter.index)

#     # =====  Bagi output =====
#     klasifikasi = out_iter.copy()

#     # Bersih: kategori C sesuai, tidak mismatch, dan gambar ada
#     bersih = out_iter.loc[
#         out_iter['is_catC_pred']
#         & out_iter['is_catC_true']
#         & (~mismatch)
#         & (~no_image)
#     ].copy()

#     # Anomali: sisanya, termasuk yang gambar kosong
#     anomali = out_iter.loc[
#         (~out_iter['is_catC_pred'])
#         | (~out_iter['is_catC_true'])
#         | mismatch
#         | no_image
#     ].copy()

#     # Pastikan kolom utama + gambar ada di semua dataframe
#     base_cols = [
#         'r101','r102','r103','r104','r105','r106','r107',
#         'r213',
#         'r215a1_label','r215b','r215d',
#         'r216_label',
#         'kbli2_true','kbli2_pred','kbli2_pred_label',
#         'kbli2_pred_proba','status_kesesuaian',
#         'r215c_url'
#     ]
#     for dfx in [klasifikasi, bersih, anomali]:
#         for col in base_cols:
#             if col not in dfx.columns and col in df.columns:
#                 dfx[col] = df[col]

#     # Urutan kolom untuk semua output (tambah kolom gambar di akhir)
#     ordered_cols = [
#         'r101','r102','r103','r104','r105','r106','r107',
#         'r213',
#         'r215a1_label','r215b','r215d',
#         'r216_label',
#         'kbli2_true','kbli2_pred','kbli2_pred_label',
#         'kbli2_pred_proba','status_kesesuaian',
#         'r215c_url'
#     ]
#     bersih_cols = [c for c in ordered_cols if c in bersih.columns]      # [file:63]
#     klasifikasi_cols = [c for c in ordered_cols if c in klasifikasi.columns]
#     anomali_cols = [c for c in ordered_cols if c in anomali.columns]

#     # =====  Tampilkan di halaman =====
#     st.subheader("Data klasifikasi (lengkap)")
#     st.dataframe(klasifikasi[klasifikasi_cols].head())

#     st.subheader("Data bersih (C sesuai & punya gambar)")
#     st.dataframe(bersih[bersih_cols].head())

#     st.subheader("Data anomali (non‑C / mismatch / tanpa gambar)")
#     st.dataframe(anomali[anomali_cols].head())

#     # =====  Download tiga file =====
#     klasifikasi_csv = klasifikasi[klasifikasi_cols].to_csv(index=False).encode("utf-8")
#     bersih_csv = bersih[bersih_cols].to_csv(index=False).encode("utf-8")
#     anomali_csv = anomali[anomali_cols].to_csv(index=False).encode("utf-8")

#     st.download_button(
#         "Download klasifikasi_r216_vs_textC.csv",
#         data=klasifikasi_csv,
#         file_name="klasifikasi_r216_vs_textC.csv",
#         mime="text/csv"
#     )
#     st.download_button(
#         "Download bersih_textC.csv",
#         data=bersih_csv,
#         file_name="bersih_textC.csv",
#         mime="text/csv"
#     )
#     st.download_button(
#         "Download anomali_kbli.csv",
#         data=anomali_csv,
#         file_name="anomali_kbli.csv",
#         mime="text/csv"
#     )

#     # Opsional: simpan model di server
#     if st.checkbox("Simpan model ke file .joblib di server"):
#         joblib.dump(pipe, "model_kbli2_rf.joblib")
#         st.success("Model disimpan sebagai model_kbli2_rf.joblib")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import chardet
# from io import StringIO
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# import joblib


# st.set_page_config(page_title="Klasifikasi KBLI 2 Digit", layout="wide")
# st.title("Klasifikasi KBLI 2 Digit dari Teks")

# st.write(
#     "Upload file mentah (CSV/Excel) berisi minimal kolom r101–r107, r213, "
#     "r215a1_label / r215b / r215d, r216_value / r216_label, dan r215c_url untuk gambar."
# )

# uploaded_file = st.file_uploader(
#     "Upload file CSV atau Excel",
#     type=["csv", "xlsx", "xls"]
# )

# # ========= Fungsi util =========

# def split_business_owner(series):
#     angle_pat = re.compile(r'<([^<>]*)>')  # termasuk kosong
#     invalid_tokens = {'', '-', '—', '.', '..', '...'}
#     biz, owner_main, owner_others = [], [], []
#     for val in series.fillna(''):
#         s = str(val).strip()
#         s = re.sub(r'\s*<\s*', '<', s)
#         s = re.sub(r'\s*>\s*', '>', s)
#         raw_owners = angle_pat.findall(s)
#         owners = []
#         for o in raw_owners:
#             oc = re.sub(r'\s+', ' ', o).strip(' <>-_./|')
#             if oc.upper() not in invalid_tokens and oc != '':
#                 owners.append(oc)
#         name_raw = angle_pat.sub('', s).strip()
#         name_clean = re.sub(r'\s{2,}', ' ', name_raw).strip(' -_/|')
#         if not name_clean and '<' in s:
#             name_clean = s.split('<', 1)[0].strip()
#         biz.append(name_clean)
#         owner_main.append(owners[0] if owners else '')
#         owner_others.append(', '.join(owners[1:]) if len(owners) > 1 else '')
#     return pd.DataFrame(
#         {
#             'nama_bisnis': biz,
#             'nama_pemilik': owner_main,
#             'nama_pemilik_lain': owner_others
#         }
#     )

# label_map = {
#  '10':'Industri Makanan','11':'Industri Minuman','12':'Industri Pengolahan Tembakau','13':'Industri Tekstil',
#  '14':'Industri Pakaian Jadi','15':'Industri Kulit dan Alas Kaki','16':'Industri Kayu','17':'Industri Kertas',
#  '18':'Industri Pencetakan dan Reproduksi Media Rekaman','19':'Industri Produk dari Batu Bara dan Pengilangan Minyak Bumi',
#  '20':'Industri Bahan Kimia dan Barang dari Bahan Kimia','21':'Industri Farmasi, Produk Obat Kimia dan Obat Tradisional',
#  '22':'Industri Karet, Barang dari Karet dan Plastik','23':'Industri Barang Galian Bukan Logam','24':'Industri Logam Dasar',
#  '25':'Industri Barang dari Logam, Bukan Mesin dan Peralatannya','26':'Industri Komputer, Barang Elektronik dan Optik',
#  '27':'Industri Peralatan Listrik','28':'Industri Mesin dan Perlengkapan','29':'Industri Kendaraan Bermotor, Trailer dan Semi Trailer',
#  '30':'Industri Alat Angkutan Lainnya','31':'Industri Furnitur','32':'Industri Pengolahan Lainnya',
#  '33':'Jasa Reparasi dan Pemasangan Mesin dan Peralatan'
# }

# def apply_iterative_rules_simple(df, cols, max_iters=3, conf_thr=0.70):
#     txt = df[cols].fillna('').agg(' '.join, axis=1).str.upper()
#     rules = [
#         (r'\bKABEL\b|\bTRAFO\b|\bAMPLI(FIER)?\b|\bINVERTER\b', '27'),
#         (r'\bCPU\b|\bLAPTOP\b|\bKAMERA\b|\bOPTIK\b', '26'),
#         (r'\bMESIN\b|\bDINAMO\b|\bPOMPA\b|\bKOMPRESOR\b', '28'),
#         (r'\bKURSI\b|\bMEJA\b|\bLEMARI\b', '31'),
#         (r'\bKERTAS\b|\bAGENDA MAP\b', '17'),
#         (r'\bCETAK\b|\bPERCETAKAN\b|\bUNDANGAN\b|\bSTIKER\b', '18'),
#         (r'\bLEM\b|\bCAT\b|\bRESIN\b', '20'),
#         (r'\bKARET\b|\bPLASTIK\b', '22'),
#         (r'\bTEPUNG\b|\bSINGKONG\b|\bBERAS\b|\bKUE\b|\bTEMPE\b|\bGETHUK\b|\bTAHU\b', '10'),
#         (r'\bAIR MINUM\b|\bSIRUP\b|\bMINUMAN\b', '11'),
#     ]
#     changed, it = True, 0
#     out2 = df.copy()
#     while changed and it < max_iters:
#         changed, it = False, it + 1
#         cand = (out2['kbli2_pred_proba'] < conf_thr)
#         for pattern, target in rules:
#             m = cand & txt.str.contains(pattern, regex=True, na=False) & (out2['kbli2_pred'] != target)
#             if m.any():
#                 out2.loc[m, 'kbli2_pred'] = target
#                 out2.loc[m, 'kbli2_pred_label'] = out2.loc[m, 'kbli2_pred'].map(label_map)
#                 changed = True
#     return out2

# # ========= Proses utama =========

# if uploaded_file is not None:
#     raw_name = uploaded_file.name
#     raw_bytes = uploaded_file.getvalue()

#     # Baca Excel vs CSV (dengan pembersihan header untuk CSV)
#     if raw_name.lower().endswith((".xlsx", ".xls")):
#         df = pd.read_excel(uploaded_file)
#     else:
#         enc = (chardet.detect(raw_bytes)['encoding'] or 'utf-8')
#         text = raw_bytes.decode(enc, errors='replace')
#         text = text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
#         lines = text.split('\n')
#         while lines and (
#             lines[0].strip().startswith('**')
#             or lines[0].strip().lower().startswith('mohon')
#             or lines[0].strip().lower().startswith('catatan')
#         ):
#             lines.pop(0)
#         df = pd.read_csv(StringIO('\n'.join(lines)))

#     # Normalisasi kolom & strip spasi
#     df.columns = [str(c).strip() for c in df.columns]
#     for c in df.columns:
#         if df[c].dtype == object:
#             df[c] = df[c].astype(str).str.strip()

#     st.subheader("Preview data mentah")
#     st.dataframe(df.head())

#     # Split r213 -> nama_bisnis / pemilik
#     if 'r213' in df.columns:
#         sp = split_business_owner(df['r213'])
#         df = pd.concat([df, sp], axis=1)

#     # Target kbli2_true dari r216
#     if 'r216_value' in df.columns:
#         df['kbli2_true'] = df['r216_value'].astype(str).str.extract(r'(\d{2})')
#     elif 'r216_label' in df.columns:
#         df['kbli2_true'] = df['r216_label'].astype(str).str.extract(r'\[(\d{2})\]')
#     else:
#         df['kbli2_true'] = np.nan

#     # Fitur teks
#     feat_cols = [c for c in ['r215a1_label', 'r215b', 'r215d'] if c in df.columns]
#     if not feat_cols:
#         st.error("Tidak ditemukan kolom r215a1_label / r215b / r215d.")
#         st.stop()
#     X_all = df[feat_cols].fillna('')

#     # Model
#     ct = ColumnTransformer(
#         [('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), feat_cols)]
#     )
#     rf = RandomForestClassifier(
#         n_estimators=500,
#         random_state=42,
#         class_weight='balanced_subsample',
#         n_jobs=-1
#     )
#     pipe = Pipeline([('prep', ct), ('clf', rf)])

#     has_y = df['kbli2_true'].notna().sum() >= 50 and df['kbli2_true'].nunique() >= 2

#     if has_y:
#         X_t = df.loc[df['kbli2_true'].notna(), feat_cols].fillna('')
#         y_t = df.loc[df['kbli2_true'].notna(), 'kbli2_true']
#         vc = y_t.value_counts()
#         ok = y_t.isin(vc[vc >= 2].index)
#         if ok.sum() >= 2 and vc[vc >= 2].shape[0] >= 2:
#             X_tr, X_te, y_tr, y_te = train_test_split(
#                 X_t[ok], y_t[ok],
#                 test_size=0.2,
#                 random_state=42,
#                 stratify=y_t[ok]
#             )
#             pipe.fit(X_tr, y_tr)
#             st.success("Model dilatih dengan train/test split.")
#         else:
#             pipe.fit(X_t, y_t)
#             st.warning("Model dilatih tanpa split (kelas jarang).")
#     else:
#         pipe.fit(
#             X_all,
#             np.random.choice([f"{i:02d}" for i in range(10, 34)], size=len(X_all))
#         )
#         st.info("Tidak cukup label r216, model hanya difit dummy agar bisa prediksi.")

#     # Prediksi + label kategori
#     pred = pipe.predict(X_all)
#     proba = pipe.predict_proba(X_all).max(axis=1)

#     out = df.copy()
#     out['kbli2_pred'] = pred
#     out['kbli2_pred_label'] = out['kbli2_pred'].map(label_map)
#     out['kbli2_pred_proba'] = proba

#     # Aturan iteratif
#     out_iter = apply_iterative_rules_simple(out, feat_cols, max_iters=3, conf_thr=0.70)

#     # Kategori C dan status
#     catC = [f"{i:02d}" for i in range(10, 34)]
#     out_iter['is_catC_pred'] = out_iter['kbli2_pred'].isin(catC)
#     out_iter['is_catC_true'] = out_iter['kbli2_true'].isin(catC)
#     mismatch = out_iter['kbli2_true'].notna() & (out_iter['kbli2_true'] != out_iter['kbli2_pred'])
#     out_iter['status_kesesuaian'] = np.where(
#         out_iter['is_catC_pred'] & out_iter['is_catC_true'] & (~mismatch), 'Sesuai C',
#         np.where(
#             ~out_iter['is_catC_pred'] & out_iter['is_catC_true'], 'True C vs Pred non-C',
#             np.where(
#                 out_iter['is_catC_pred'] & ~out_iter['is_catC_true'],
#                 'True non-C vs Pred C',
#                 'True non-C & Pred non-C'
#             )
#         )
#     )

#     # Flag baris yang tidak punya gambar
#     if 'r215c_url' in out_iter.columns:
#         no_image = out_iter['r215c_url'].isna() | (out_iter['r215c_url'].astype(str).str.strip() == '')
#     else:
#         no_image = pd.Series(False, index=out_iter.index)

#     # =====  Bagi output =====
#     klasifikasi = out_iter.copy()  # tidak diubah isinya

#     bersih = out_iter.loc[
#         out_iter['is_catC_pred']
#         & out_iter['is_catC_true']
#         & (~mismatch)
#         & (~no_image)
#     ].copy()

#     anomali = out_iter.loc[
#         (~out_iter['is_catC_pred'])
#         | (~out_iter['is_catC_true'])
#         | mismatch
#         | no_image
#     ].copy()

#     # Tambah alasan anomali untuk membantu pemeriksaan (mirip remark PML)
#     reasons = []
#     for i, row in anomali.iterrows():
#         r = []
#         if row.get('kbli2_true') in catC and row.get('kbli2_pred') not in catC:
#             r.append("True C vs Pred non-C")
#         elif row.get('kbli2_true') not in catC and row.get('kbli2_pred') in catC:
#             r.append("True non-C vs Pred C")
#         if pd.isna(row.get('kbli2_true')):
#             r.append("KBLI r216 kosong")
#         if i in anomali.index and no_image.loc[i]:
#             r.append("Tanpa gambar")
#         reasons.append("; ".join(r) if r else "Periksa manual")
#     anomali['alasan_anomali'] = reasons

#     # =====  Kolom & urutan =====
#     # bersih/anomali boleh ditambah kolom dari df, klasifikasi dibiarkan apa adanya
#     base_cols_ba = [
#         'r101','r102','r103','r104','r105','r106','r107',
#         'r213',
#         'r215a1_label','r215b','r215d',
#         'r216_label',
#         'kbli2_true','kbli2_pred','kbli2_pred_label',
#         'kbli2_pred_proba','status_kesesuaian',
#         'r215c_url'
#     ]
#     for dfx in [bersih, anomali]:
#         for col in base_cols_ba:
#             if col not in dfx.columns and col in df.columns:
#                 dfx[col] = df[col]

#     ordered_cols = [
#         'r101','r102','r103','r104','r105','r106','r107',
#         'r213',
#         'r215a1_label','r215b','r215d',
#         'r216_label',
#         'kbli2_true','kbli2_pred','kbli2_pred_label',
#         'kbli2_pred_proba','status_kesesuaian',
#         'r215c_url'
#     ]

#     klasifikasi_cols = [c for c in ordered_cols if c in klasifikasi.columns]
#     bersih_cols      = [c for c in ordered_cols if c in bersih.columns]
#     anomali_cols     = [c for c in ordered_cols if c in anomali.columns] + ['alasan_anomali']

#     # Sembunyikan kolom gambar di view bila semua kosong
#     def view_cols(df, cols):
#         if 'r215c_url' in cols and df['r215c_url'].astype(str).str.strip().eq('').all():
#             return [c for c in cols if c != 'r215c_url']
#         return cols

#     klasifikasi_view = view_cols(klasifikasi, klasifikasi_cols)
#     bersih_view      = view_cols(bersih, bersih_cols)
#     anomali_view     = view_cols(anomali, anomali_cols)

#     # =====  Ringkasan akurasi (proporsi Sesuai C) =====
#     if 'status_kesesuaian' in klasifikasi.columns:
#         total_labeled = (klasifikasi['kbli2_true'].notna()).sum()
#         sesuai_c = (klasifikasi['status_kesesuaian'] == 'Sesuai C').sum()
#         if total_labeled > 0:
#             akurasi = sesuai_c / total_labeled
#             st.metric("Proporsi 'Sesuai C' (KBLI 2 digit)", f"{akurasi:.1%}")

#     # =====  Tampilkan di halaman =====
#     st.subheader("Data klasifikasi (lengkap, hanya urutan kolom diatur)")
#     st.dataframe(klasifikasi[klasifikasi_view].head())

#     st.subheader("Data bersih (C sesuai & punya gambar)")
#     st.dataframe(bersih[bersih_view].head())

#     st.subheader("Data anomali (non‑C / mismatch / tanpa gambar)")
#     st.dataframe(anomali[anomali_view].head())

#     # =====  Download CSV =====
#     klasifikasi_csv = klasifikasi[klasifikasi_cols].to_csv(index=False).encode("utf-8")
#     bersih_csv      = bersih[bersih_cols].to_csv(index=False).encode("utf-8")
#     anomali_csv     = anomali[anomali_cols].to_csv(index=False).encode("utf-8")

#     st.download_button(
#         "Download klasifikasi_r216_vs_textC.csv",
#         data=klasifikasi_csv,
#         file_name="klasifikasi_r216_vs_textC.csv",
#         mime="text/csv"
#     )
#     st.download_button(
#         "Download bersih_textC.csv",
#         data=bersih_csv,
#         file_name="bersih_textC.csv",
#         mime="text/csv"
#     )
#     st.download_button(
#         "Download anomali_kbli.csv",
#         data=anomali_csv,
#         file_name="anomali_kbli.csv",
#         mime="text/csv"
#     )

#     # Opsional: simpan model di server
#     if st.checkbox("Simpan model ke file .joblib di server"):
#         joblib.dump(pipe, "model_kbli2_rf.joblib")
#         st.success("Model disimpan sebagai model_kbli2_rf.joblib")

import streamlit as st
import pandas as pd
import numpy as np
import re
import chardet
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import joblib

st.set_page_config(page_title="Klasifikasi KBLI 2 Digit (TF-IDF)", layout="wide")
st.title("Klasifikasi KBLI 2 Digit dari Teks — versi perbaikan (TF-IDF)")

st.write(
    "Upload CSV/Excel yang berisi minimal kolom r215a1_label / r215b / r215d, "
    "dan (opsional) r216_value / r216_label untuk label."
)

uploaded_file = st.file_uploader("Upload CSV atau Excel", type=["csv", "xlsx", "xls"])

# ---- util --------------------------------------------------------------
def normalize_text(s):
    s = str(s).upper()
    s = re.sub(r'[^\w\s]', ' ', s)      # hapus punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    return s

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
    return pd.DataFrame({'nama_bisnis': biz, 'nama_pemilik': owner_main, 'nama_pemilik_lain': owner_others})

# label map tetap sama (10-33)
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

# ---- aturan iteratif yang diperhalus ----------------------------------
def apply_iterative_rules_improved(df, text_col='text', max_iters=2, conf_thr=0.35):
    txt = df[text_col].fillna('').str.upper()
    # Specific exception rules FIRST
    #  - Sablon kaos -> pakaian jadi (14)
    #  - Rebana / alat musik -> pengolahan lainnya (32)
    #  - Kemeja / kaos -> pakaian (14) if context fits
    exception_rules = [
        (r'\b(SABLON).*(KAOS|KEMEJA|BAJU)\b', '14'),
        (r'\b(KAOS|KEMEJA).*(SABLON)\b', '14'),
        (r'\b(REBANA|GENDANG|ALAT MUSIK)\b', '32'),
    ]
    # Generic rules (kept but with negative lookahead to avoid false hits)
    generic_rules = [
        (r'(?<!KAOS.*)\bCETAK\b|\bPERCETAKAN\b|\bUNDANGAN\b|\bSTIKER\b', '18'),
        (r'\bKABEL\b|\bTRAFO\b|\bAMPLI(FIER)?\b|\bINVERTER\b', '27'),
        (r'\bCPU\b|\bLAPTOP\b|\bKAMERA\b|\bOPTIK\b', '26'),
        (r'\bMESIN\b|\bDINAMO\b|\bPOMPA\b|\bKOMPRESOR\b', '28'),
        (r'\bKURSI\b|\bMEJA\b|\bLEMARI\b', '31'),
        (r'\bKERTAS\b(?!.*KAOS)', '17'),
        (r'\bLEM\b|\bCAT\b|\bRESIN\b', '20'),
        (r'\bKARET\b|\bPLASTIK\b', '22'),
        (r'\bTEPUNG\b|\bSINGKONG\b|\bBERAS\b|\bKUE\b|\bTEMPE\b|\bGETHUK\b|\bTAHU\b', '10'),
        (r'\bAIR MINUM\b|\bSIRUP\b|\bMINUMAN\b', '11'),
    ]

    out = df.copy()
    it = 0
    changed = True
    while changed and it < max_iters:
        changed = False
        it += 1
        cand = (out['kbli2_pred_proba'] < conf_thr)
        # apply exceptions first
        for pat, tgt in exception_rules:
            m = cand & out[text_col].str.contains(pat, regex=True, na=False) & (out['kbli2_pred'] != tgt)
            if m.any():
                out.loc[m, 'kbli2_pred'] = tgt
                out.loc[m, 'kbli2_pred_label'] = out.loc[m, 'kbli2_pred'].map(label_map)
                changed = True
        # then generic rules but skip when exceptions keywords exist (to reduce false positives)
        for pat, tgt in generic_rules:
            m = cand & out[text_col].str.contains(pat, regex=True, na=False) & (out['kbli2_pred'] != tgt)
            if m.any():
                out.loc[m, 'kbli2_pred'] = tgt
                out.loc[m, 'kbli2_pred_label'] = out.loc[m, 'kbli2_pred'].map(label_map)
                changed = True
    return out

# ---- main --------------------------------------------------------------
if uploaded_file is not None:
    raw_name = uploaded_file.name
    raw_bytes = uploaded_file.getvalue()
    # Read file
    if raw_name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        enc = (chardet.detect(raw_bytes)['encoding'] or 'utf-8')
        text = raw_bytes.decode(enc, errors='replace')
        text = text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
        lines = text.split('\n')
        while lines and (lines[0].strip().startswith('**') or lines[0].strip().lower().startswith('mohon') or lines[0].strip().lower().startswith('catatan')):
            lines.pop(0)
        df = pd.read_csv(StringIO('\n'.join(lines)))

    # normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    st.subheader("Preview data mentah")
    st.dataframe(df.head())

    # split r213
    if 'r213' in df.columns:
        sp = split_business_owner(df['r213'])
        df = pd.concat([df, sp], axis=1)

    # target
    if 'r216_value' in df.columns:
        df['kbli2_true'] = df['r216_value'].astype(str).str.extract(r'(\d{2})')
    elif 'r216_label' in df.columns:
        df['kbli2_true'] = df['r216_label'].astype(str).str.extract(r'\[(\d{2})\]')
    else:
        df['kbli2_true'] = np.nan

    # build text feature
    feat_cols = [c for c in ['r215a1_label', 'r215b', 'r215d'] if c in df.columns]
    if not feat_cols:
        st.error("Tidak ditemukan kolom r215a1_label / r215b / r215d.")
        st.stop()
    df['text'] = df[feat_cols].fillna('').agg(' '.join, axis=1).apply(normalize_text)

    # pipeline TF-IDF + classifier
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga')

    pipe = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf)
    ])

    # decide to train only if enough labelled data
    has_y = df['kbli2_true'].notna().sum() >= 50 and df['kbli2_true'].nunique() >= 2
    if has_y:
        X_t = df.loc[df['kbli2_true'].notna(), 'text']
        y_t = df.loc[df['kbli2_true'].notna(), 'kbli2_true']
        vc = y_t.value_counts()
        ok = y_t.isin(vc[vc >= 2].index)
        if ok.sum() >= 10 and vc[vc >= 2].shape[0] >= 2:
            X_tr, X_te, y_tr, y_te = train_test_split(X_t[ok], y_t[ok], test_size=0.2, random_state=42, stratify=y_t[ok])
            pipe.fit(X_tr, y_tr)
            st.success("Model dilatih dengan train/test split.")
            # report simple metrics
            try:
                from sklearn.metrics import classification_report, accuracy_score
                ypred = pipe.predict(X_te)
                st.text("Eval (simple) on held-out set:")
                st.text(classification_report(y_te, ypred))
                st.text(f"Accuracy: {accuracy_score(y_te, ypred):.3f}")
            except Exception:
                pass
        else:
            pipe.fit(X_t, y_t)
            st.warning("Model dilatih tanpa split (kelas jarang).")
    else:
        # if not enough labels, fit on text to get vectorizer but use dummy predictions later
        pipe.fit(df['text'], np.random.choice([f"{i:02d}" for i in range(10,34)], size=len(df)))
        st.info("Tidak cukup label r216 — model fit dummy agar bisa prediksi, tetapi hasil mungkin tidak akurat.")

    # predict
    df['kbli2_pred'] = pipe.predict(df['text'])
    # store proba (max)
    try:
        df['kbli2_pred_proba'] = pipe.predict_proba(df['text']).max(axis=1)
    except Exception:
        df['kbli2_pred_proba'] = 0.0

    # map label
    df['kbli2_pred_label'] = df['kbli2_pred'].map(label_map)

    # apply improved iterative rules
    out_iter = apply_iterative_rules_improved(df.copy(), text_col='text', max_iters=2, conf_thr=0.35)

    # category C detection (10-33)
    catC = [f"{i:02d}" for i in range(10,34)]
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

    # treat no_image as info, not automatically anomali (unless user wants)
    if 'r215c_url' in out_iter.columns:
        no_image = out_iter['r215c_url'].isna() | (out_iter['r215c_url'].astype(str).str.strip() == '')
    else:
        no_image = pd.Series(False, index=out_iter.index)

    out_iter['tanpa_gambar'] = no_image

    # split outputs
    klasifikasi = out_iter.copy()
    bersih = out_iter.loc[
        out_iter['is_catC_pred'] &
        out_iter['is_catC_true'] &
        (~mismatch) &
        (~no_image)
    ].copy()
    anomali = out_iter.loc[
        (~out_iter['is_catC_pred']) |
        (~out_iter['is_catC_true']) |
        mismatch
    ].copy()

    # add alasan_anomali
    reasons = []
    for i, row in anomali.iterrows():
        r = []
        if pd.isna(row.get('kbli2_true')):
            r.append("KBLI r216 kosong")
        if row.get('kbli2_true') in catC and row.get('kbli2_pred') not in catC:
            r.append("True C vs Pred non-C")
        if row.get('kbli2_pred') in catC and row.get('kbli2_true') not in catC and not pd.isna(row.get('kbli2_true')):
            r.append("True non-C vs Pred C")
        if no_image.loc[i]:
            r.append("Tanpa gambar")
        reasons.append("; ".join(r) if r else "Periksa manual")
    anomali['alasan_anomali'] = reasons

    # show metrics
    total_labeled = (klasifikasi['kbli2_true'].notna()).sum()
    sesuai_c = (klasifikasi['status_kesesuaian'] == 'Sesuai C').sum()
    if total_labeled > 0:
        akurasi = sesuai_c / total_labeled
        st.metric("Proporsi 'Sesuai C' (KBLI 2 digit)", f"{akurasi:.1%}")

    # display
    st.subheader("Preview klasifikasi (perbaikan)")
    st.dataframe(klasifikasi.head())

    st.subheader("Contoh bersih (C sesuai & ada gambar)")
    st.dataframe(bersih.head())

    st.subheader("Contoh anomali (non-C / mismatch)")
    st.dataframe(anomali.head())

    # download
    st.download_button("Download semua klasifikasi (csv)",
                       data=klasifikasi.to_csv(index=False).encode('utf-8'),
                       file_name="klasifikasi_kbli_fixed.csv",
                       mime="text/csv")
    st.download_button("Download bersih (csv)",
                       data=bersih.to_csv(index=False).encode('utf-8'),
                       file_name="bersih_kbli_fixed.csv",
                       mime="text/csv")
    st.download_button("Download anomali (csv)",
                       data=anomali.to_csv(index=False).encode('utf-8'),
                       file_name="anomali_kbli_fixed.csv",
                       mime="text/csv")

    if st.checkbox("Simpan model ke file .joblib di server"):
        joblib.dump(pipe, "model_kbli2_tfidf_lr.joblib")
        st.success("Model disimpan sebagai model_kbli2_tfidf_lr.joblib")
