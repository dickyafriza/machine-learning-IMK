"""
Perbandingan akurasi: Random Forest vs KNN
untuk klasifikasi KBLI 2 digit.

Cara pakai:
  streamlit run compare_algorithms.py
  atau
  python compare_algorithms.py   (jika pakai argparse file path)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import chardet
import time
from io import StringIO

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

st.set_page_config(page_title="Perbandingan RF vs KNN", layout="wide")
st.title("⚔️ Perbandingan Akurasi: Random Forest vs KNN")
st.write("Upload file data (CSV/Excel) untuk membandingkan kedua algoritma.")

uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx", "xls"])


# ========= Fungsi util (sama dengan app.py) =========

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
    return pd.DataFrame({
        'nama_bisnis': biz,
        'nama_pemilik': owner_main,
        'nama_pemilik_lain': owner_others
    })


def detect_and_map_columns(df):
    is_vimk = any(c in df.columns for c in [
        'r316a_label', 'r316a_value', 'r316b', 'v317_lab', 'r317_label'
    ])

    if is_vimk:
        vimk_feat_candidates = [
            'r316a_label', 'r316a_value', 'r316a_lain', 'r316b', 'r316d'
        ]
        feat_cols = [c for c in vimk_feat_candidates if c in df.columns]
        kbli_label = None
        for candidate in ['v317_lab', 'r317_label']:
            if candidate in df.columns:
                kbli_label = candidate
                break
        return {
            'format': 'VIMK25',
            'nama_usaha': 'r314',
            'feat_cols': feat_cols,
            'kbli_label': kbli_label,
            'kbli_value': None,
        }
    else:
        return {
            'format': 'FASIH',
            'nama_usaha': 'r213',
            'feat_cols': [c for c in ['r215a1_label', 'r215b', 'r215d'] if c in df.columns],
            'kbli_label': 'r216_label',
            'kbli_value': 'r216_value',
        }


# ========= Proses utama =========

if uploaded_file is not None:
    raw_name = uploaded_file.name
    raw_bytes = uploaded_file.getvalue()

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

    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    col_map = detect_and_map_columns(df)
    st.info(f"Format data terdeteksi: **{col_map['format']}**")

    # Filter submitted
    if 'assignment_status_alias' in df.columns:
        df = df[df['assignment_status_alias'].str.upper() == 'SUBMITTED BY PENCACAH'].reset_index(drop=True)

    # Ekstrak kbli2_true
    feat_cols = col_map['feat_cols']
    kbli_value_col = col_map.get('kbli_value')
    kbli_label_col = col_map.get('kbli_label')

    if kbli_value_col and kbli_value_col in df.columns:
        df['kbli2_true'] = df[kbli_value_col].astype(str).str.extract(r'(\d{2})')
    elif kbli_label_col and kbli_label_col in df.columns:
        kbli_str = df[kbli_label_col].astype(str)
        extracted = kbli_str.str.extract(r'\[(\d{2})\]')
        mask_na = extracted[0].isna()
        if mask_na.any():
            extracted.loc[mask_na, 0] = kbli_str[mask_na].str.extract(r'(\d{2})')[0]
        df['kbli2_true'] = extracted[0]
    else:
        df['kbli2_true'] = np.nan

    if not feat_cols:
        st.error("Tidak ditemukan kolom fitur teks.")
        st.stop()

    # Bangun fitur teks
    df['text_all'] = (
        df[feat_cols]
        .fillna('')
        .astype(str)
        .apply(lambda col: col.str.replace(r'(?i)^nan$', '', regex=True))
        .apply(lambda col: col.str.replace(r'(?i)^none$', '', regex=True))
        .agg(' '.join, axis=1)
        .str.strip()
    )

    # Filter data yang punya label
    has_label = df['kbli2_true'].notna()
    df_labeled = df[has_label].copy()

    # Filter kelas yang punya >= 2 sampel
    vc = df_labeled['kbli2_true'].value_counts()
    ok_classes = vc[vc >= 2].index
    df_labeled = df_labeled[df_labeled['kbli2_true'].isin(ok_classes)].reset_index(drop=True)

    st.write(f"**Jumlah data berlabel**: {len(df_labeled)} baris, **{df_labeled['kbli2_true'].nunique()}** kelas KBLI")

    if len(df_labeled) < 10 or df_labeled['kbli2_true'].nunique() < 2:
        st.error("Data berlabel terlalu sedikit untuk perbandingan. Minimal 10 baris dengan ≥2 kelas.")
        st.stop()

    # ========= Split data =========
    X = df_labeled[['text_all']]
    y = df_labeled['kbli2_true']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.write(f"**Train**: {len(X_train)} | **Test**: {len(X_test)}")

    effective_min_df = max(1, min(3, len(X_train) // 10))

    # ========= TF-IDF (shared) =========
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=effective_min_df
    )

    ct = ColumnTransformer(
        transformers=[('text', tfidf, 'text_all')],
        remainder='drop'
    )

    # =============================================
    # MODEL 1: Random Forest + GridSearchCV
    # =============================================
    st.subheader("🌲 Random Forest")
    with st.spinner("Training Random Forest dengan GridSearchCV..."):
        rf_pipe = Pipeline([
            ('prep', ColumnTransformer(
                transformers=[('text', TfidfVectorizer(
                    lowercase=True, ngram_range=(1, 2), min_df=effective_min_df
                ), 'text_all')],
                remainder='drop'
            )),
            ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])

        rf_param_grid = {
            'clf__n_estimators': [300, 600, 900],
            'clf__max_depth': [None, 20, 40],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2],
            'clf__class_weight': ['balanced']
        }

        rf_grid = GridSearchCV(
            estimator=rf_pipe,
            param_grid=rf_param_grid,
            cv=3, n_jobs=-1, verbose=0, scoring='accuracy'
        )

        t0 = time.time()
        rf_grid.fit(X_train, y_train)
        rf_train_time = time.time() - t0

        rf_best = rf_grid.best_estimator_
        rf_pred = rf_best.predict(X_test)

        rf_acc = accuracy_score(y_test, rf_pred)
        rf_prec = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
        rf_rec = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
        rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)

    st.write(f"**Best params**: {rf_grid.best_params_}")
    st.write(f"**Best CV score**: {rf_grid.best_score_:.4f}")

    # =============================================
    # MODEL 2: KNN + GridSearchCV
    # =============================================
    st.subheader("📐 K-Nearest Neighbors (KNN)")
    with st.spinner("Training KNN dengan GridSearchCV..."):
        knn_pipe = Pipeline([
            ('prep', ColumnTransformer(
                transformers=[('text', TfidfVectorizer(
                    lowercase=True, ngram_range=(1, 2), min_df=effective_min_df
                ), 'text_all')],
                remainder='drop'
            )),
            ('clf', KNeighborsClassifier())
        ])

        # Tentukan range k berdasarkan jumlah data
        max_k = min(25, len(X_train) // 2)
        k_values = list(range(1, max_k + 1, 2))  # 1, 3, 5, 7, ...

        knn_param_grid = {
            'clf__n_neighbors': k_values,
            'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['cosine', 'euclidean'],
        }

        knn_grid = GridSearchCV(
            estimator=knn_pipe,
            param_grid=knn_param_grid,
            cv=3, n_jobs=-1, verbose=0, scoring='accuracy'
        )

        t0 = time.time()
        knn_grid.fit(X_train, y_train)
        knn_train_time = time.time() - t0

        knn_best = knn_grid.best_estimator_
        knn_pred = knn_best.predict(X_test)

        knn_acc = accuracy_score(y_test, knn_pred)
        knn_prec = precision_score(y_test, knn_pred, average='weighted', zero_division=0)
        knn_rec = recall_score(y_test, knn_pred, average='weighted', zero_division=0)
        knn_f1 = f1_score(y_test, knn_pred, average='weighted', zero_division=0)

    st.write(f"**Best params**: {knn_grid.best_params_}")
    st.write(f"**Best CV score**: {knn_grid.best_score_:.4f}")

    # =============================================
    # TABEL PERBANDINGAN
    # =============================================
    st.subheader("📊 Tabel Perbandingan")

    comparison = pd.DataFrame({
        'Metrik': ['Accuracy', 'Precision (weighted)', 'Recall (weighted)',
                   'F1-Score (weighted)', 'Best CV Score', 'Training Time (s)'],
        'Random Forest': [
            f"{rf_acc:.4f}", f"{rf_prec:.4f}", f"{rf_rec:.4f}",
            f"{rf_f1:.4f}", f"{rf_grid.best_score_:.4f}", f"{rf_train_time:.1f}"
        ],
        'KNN': [
            f"{knn_acc:.4f}", f"{knn_prec:.4f}", f"{knn_rec:.4f}",
            f"{knn_f1:.4f}", f"{knn_grid.best_score_:.4f}", f"{knn_train_time:.1f}"
        ],
    })

    # Tambah kolom pemenang
    winners = []
    rf_vals = [rf_acc, rf_prec, rf_rec, rf_f1, rf_grid.best_score_, -rf_train_time]
    knn_vals = [knn_acc, knn_prec, knn_rec, knn_f1, knn_grid.best_score_, -knn_train_time]
    for rv, kv in zip(rf_vals, knn_vals):
        if rv > kv:
            winners.append("🌲 Random Forest")
        elif kv > rv:
            winners.append("📐 KNN")
        else:
            winners.append("🤝 Seri")
    comparison['Pemenang'] = winners

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # Highlight pemenang
    diff = rf_acc - knn_acc
    if diff > 0:
        st.success(
            f"🌲 **Random Forest MENANG** dengan selisih akurasi **{abs(diff)*100:.2f}%** "
            f"(RF: {rf_acc:.4f} vs KNN: {knn_acc:.4f})"
        )
    elif diff < 0:
        st.success(
            f"📐 **KNN MENANG** dengan selisih akurasi **{abs(diff)*100:.2f}%** "
            f"(KNN: {knn_acc:.4f} vs RF: {rf_acc:.4f})"
        )
    else:
        st.info(f"🤝 **Seri!** Akurasi sama: {rf_acc:.4f}")

    # =============================================
    # CLASSIFICATION REPORT detail
    # =============================================
    with st.expander("📋 Classification Report — Random Forest"):
        st.text(classification_report(y_test, rf_pred, zero_division=0))

    with st.expander("📋 Classification Report — KNN"):
        st.text(classification_report(y_test, knn_pred, zero_division=0))

    # =============================================
    # CROSS-VALIDATION pada seluruh data berlabel
    # =============================================
    st.subheader("🔄 Cross-Validation (5-Fold) pada Seluruh Data Berlabel")

    with st.spinner("Running 5-fold CV..."):
        rf_cv = cross_val_score(rf_best, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        knn_cv = cross_val_score(knn_best, X, y, cv=5, scoring='accuracy', n_jobs=-1)

    cv_df = pd.DataFrame({
        'Fold': [f"Fold {i+1}" for i in range(5)] + ['Mean', 'Std'],
        'Random Forest': [f"{s:.4f}" for s in rf_cv] + [f"{rf_cv.mean():.4f}", f"{rf_cv.std():.4f}"],
        'KNN': [f"{s:.4f}" for s in knn_cv] + [f"{knn_cv.mean():.4f}", f"{knn_cv.std():.4f}"],
    })
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

    st.info(
        f"**CV Mean — RF**: {rf_cv.mean():.4f} ± {rf_cv.std():.4f} | "
        f"**CV Mean — KNN**: {knn_cv.mean():.4f} ± {knn_cv.std():.4f}"
    )
