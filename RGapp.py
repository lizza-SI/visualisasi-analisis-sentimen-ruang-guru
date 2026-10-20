import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# =====================================================
# 1. KONFIGURASI
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR, "model_nb_sentimen.joblib"
)

DATA_FILE = os.path.join(
    BASE_DIR, "data_bersih_berlabel.csv"
)

FEATURE_FILE = os.path.join(
    BASE_DIR, "influential_features.csv"
)

ASPECT_FILE = os.path.join(
    BASE_DIR, "hasil_analisis_aspek.csv"
)

COLORS = {
    "Negatif": "#EF4444",
    "Netral": "#F59E0B",
    "Positif": "#10B981"
}

st.set_page_config(
    page_title="Sentiment Mining Ruangguru",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Sentiment Mining Dashboard")
st.subheader("Analisis Sentimen Ulasan Ruangguru")
st.caption(
    "Multinomial Naive Bayes | TF-IDF | Analisis Teks"
)

# =====================================================
# 2. MEMUAT MODEL DAN DATA
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)

    data["content"] = data["content"].fillna("").astype(str)
    data["clean_content"] = (
        data["clean_content"].fillna("").astype(str)
    )
    data["at"] = pd.to_datetime(
        data["at"], errors="coerce"
    )

    return data


if not os.path.exists(MODEL_FILE):
    st.error("File model belum ditemukan: model_nb_sentimen.joblib")
    st.stop()

if not os.path.exists(DATA_FILE):
    st.error("File data belum ditemukan: data_bersih_berlabel.csv")
    st.stop()

bundle = load_model()
df = load_data()

vectorizer = bundle["vectorizer"]
model = bundle["model"]
classes = list(model.classes_)
metrics = bundle.get("metrics", {})

# =====================================================
# 3. RINGKASAN
# =====================================================

st.header("1. Ringkasan Dataset")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Ulasan", f"{len(df):,}")
c2.metric("Rata-rata Rating", f"{df['score'].mean():.2f}")
c3.metric(
    "Sentimen Positif",
    f"{(df['label'] == 'Positif').mean() * 100:.1f}%"
)
c4.metric(
    "Sentimen Negatif",
    f"{(df['label'] == 'Negatif').mean() * 100:.1f}%"
)

st.subheader("Distribusi Sentimen")

distribusi = (
    df["label"]
    .value_counts()
    .reindex(classes, fill_value=0)
    .rename_axis("Sentimen")
    .reset_index(name="Jumlah")
)

fig = px.pie(
    distribusi,
    names="Sentimen",
    values="Jumlah",
    hole=0.45,
    color="Sentimen",
    color_discrete_map=COLORS,
    title="Proporsi Sentimen Pengguna"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 4. EKSPLORASI DATA
# =====================================================

st.header("2. Eksplorasi Data")

col1, col2 = st.columns(2)

with col1:
    rating = (
        df["score"]
        .value_counts()
        .sort_index()
        .rename_axis("Rating")
        .reset_index(name="Jumlah")
    )

    fig_rating = px.bar(
        rating,
        x="Rating",
        y="Jumlah",
        title="Distribusi Rating",
        text="Jumlah"
    )

    st.plotly_chart(
        fig_rating,
        use_container_width=True
    )

with col2:
    df["panjang_ulasan"] = (
        df["content"].str.split().str.len()
    )

    fig_length = px.histogram(
        df,
        x="panjang_ulasan",
        nbins=30,
        title="Panjang Ulasan"
    )

    st.plotly_chart(
        fig_length,
        use_container_width=True
    )

st.subheader("Data Ulasan")

pilihan_sentimen = st.multiselect(
    "Filter sentimen",
    options=classes,
    default=classes
)

df_filter = df[df["label"].isin(pilihan_sentimen)]

st.dataframe(
    df_filter[
        ["content", "score", "label", "clean_content"]
    ],
    use_container_width=True
)

st.download_button(
    "Unduh Data Hasil Filter",
    data=df_filter.to_csv(index=False).encode("utf-8-sig"),
    file_name="data_ulasan_filter.csv",
    mime="text/csv"
)

# =====================================================
# 5. EVALUASI MODEL
# =====================================================

st.header("3. Evaluasi Model")

metric_cols = st.columns(4)

metric_names = [
    "Accuracy",
    "Precision (macro)",
    "Recall (macro)",
    "F1-Score (macro)"
]

for col, name in zip(metric_cols, metric_names):
    value = metrics.get(name)

    if value is not None:
        col.metric(name, f"{value * 100:.2f}%")
    else:
        col.metric(name, "Tidak tersedia")

st.subheader("Confusion Matrix")

if "clean_content" in df.columns:
    # Split harus konsisten dengan notebook.
    from sklearn.model_selection import train_test_split

    X = df["clean_content"].astype(str)
    y = df["label"].astype(str)

    try:
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=bundle.get("seed", 42)
        )

        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=classes
        )

        fig_cm = px.imshow(
            cm,
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="Purples",
            labels={
                "x": "Prediksi",
                "y": "Aktual",
                "color": "Jumlah"
            }
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

        st.text(
            classification_report(
                y_test,
                y_pred,
                labels=classes,
                zero_division=0
            )
        )

    except ValueError as exc:
        st.warning(f"Evaluasi ulang tidak dapat dilakukan: {exc}")

# =====================================================
# 6. INFLUENTIAL FEATURES
# =====================================================

st.header("4. Influential Features")

if os.path.exists(FEATURE_FILE):
    features = pd.read_csv(FEATURE_FILE)

    kelas_feature = st.selectbox(
        "Pilih kelas sentimen",
        options=classes,
        key="kelas_feature"
    )

    data_feature = features[
        features["Kelas"] == kelas_feature
    ].sort_values("Skor Pengaruh", ascending=True).tail(15)

    fig_feature = px.bar(
        data_feature,
        x="Skor Pengaruh",
        y="Fitur",
        orientation="h",
        title=f"Kata Paling Khas: {kelas_feature}",
        color_discrete_sequence=[
            COLORS.get(kelas_feature, "#6366F1")
        ]
    )

    st.plotly_chart(
        fig_feature,
        use_container_width=True
    )

    st.dataframe(
        data_feature.sort_values(
            "Skor Pengaruh", ascending=False
        ),
        use_container_width=True
    )
else:
    st.info(
        "File influential_features.csv belum ditemukan. "
        "Jalankan tahap Influential Features pada notebook."
    )

# =====================================================
# 7. ANALISIS ASPEK
# =====================================================

st.header("5. Analisis Aspek")

if os.path.exists(ASPECT_FILE):
    aspek = pd.read_csv(ASPECT_FILE)

    fig_aspek = px.bar(
        aspek,
        x="Jumlah Ulasan",
        y="Aspek",
        orientation="h",
        title="Jumlah Ulasan per Aspek",
        text="Jumlah Ulasan"
    )

    st.plotly_chart(
        fig_aspek,
        use_container_width=True
    )

    kolom_sentimen = [
        "% Negatif", "% Netral", "% Positif"
    ]

    tersedia = [
        k for k in kolom_sentimen if k in aspek.columns
    ]

    if tersedia:
        data_long = aspek.melt(
            id_vars="Aspek",
            value_vars=tersedia,
            var_name="Sentimen",
            value_name="Persentase"
        )

        fig_komposisi = px.bar(
            data_long,
            x="Persentase",
            y="Aspek",
            color="Sentimen",
            orientation="h",
            barmode="stack",
            title="Komposisi Sentimen per Aspek"
        )

        st.plotly_chart(
            fig_komposisi,
            use_container_width=True
        )

    st.dataframe(aspek, use_container_width=True)
else:
    st.info(
        "File hasil_analisis_aspek.csv belum ditemukan. "
        "Jalankan tahap Analisis Aspek pada notebook."
    )

# =====================================================
# 8. PREDIKSI ULASAN BARU
# =====================================================

st.header("6. Prediksi Sentimen Ulasan Baru")

STOPWORDS = {
    "di", "dari", "ke", "pada", "dengan", "untuk", "oleh",
    "tentang", "kepada", "yang", "dan", "atau", "tetapi",
    "tapi", "karena", "meskipun", "jika", "agar", "saya",
    "aku", "kamu", "dia", "kita", "mereka", "ini", "itu",
    "lah", "tuh", "nya", "dong", "aja", "gitu", "deh",
    "emang", "ya", "sih", "kok", "lho", "nih", "si", "pun",
    "juga", "sudah", "akan", "lagi", "pernah", "hanya",
    "adalah", "dalam", "ada", "para", "sebagai", "saat",
    "ketika", "kalau", "jadi", "sama", "mau"
}

NEGATIONS = {
    "tidak", "bukan", "belum", "jangan", "kurang",
    "gak", "ga", "nggak", "ngga", "tak", "gk", "enggak"
}

SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak",
    "ngga": "tidak", "gk": "tidak", "enggak": "tidak",
    "tdk": "tidak", "tak": "tidak", "bgt": "banget",
    "bgus": "bagus", "bagu": "bagus", "mantul": "mantap",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "blm": "belum", "yg": "yang", "dgn": "dengan",
    "krn": "karena", "karna": "karena", "utk": "untuk",
    "jd": "jadi", "tp": "tapi", "tpi": "tapi",
    "klo": "kalau", "kalo": "kalau", "sy": "saya",
    "sm": "sama", "sma": "sama", "bs": "bisa",
    "bsa": "bisa", "gabisa": "tidak bisa",
    "lemot": "lambat", "lag": "lambat",
    "ngelag": "lambat", "eror": "error", "erorr": "error",
    "apk": "aplikasi", "app": "aplikasi",
    "makasih": "terima kasih", "terimakasih": "terima kasih",
    "trims": "terima kasih", "oke": "ok", "okey": "ok",
    "rekomen": "recommended", "kereen": "keren",
    "jelek": "buruk", "jlek": "buruk", "pake": "pakai",
    "pke": "pakai", "jg": "juga", "dr": "dari",
    "dl": "dulu", "bgmn": "bagaimana", "gmn": "bagaimana",
    "knp": "kenapa", "kyk": "seperti", "bnyk": "banyak"
}

DOMAIN_STOPWORDS = {"ruangguru", "rg", "ruang", "kak"}


@st.cache_resource
def get_stemmer():
    return StemmerFactory().create_stemmer()


def preprocess_baru(teks):
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)
    teks = re.sub(r"\bruang\s+guru\b", "ruangguru", teks)
    teks = re.sub(r"(.)\1{2,}", r"\1", teks)
    teks = re.sub(r"[^a-z\s]", " ", teks)

    tokens = []
    for kata in teks.split():
        tokens.extend(SLANG.get(kata, kata).split())

    tokens = [
        kata for kata in tokens
        if (kata not in STOPWORDS or kata in NEGATIONS)
        and kata not in DOMAIN_STOPWORDS
        and len(kata) > 1
    ]

    stemmer = get_stemmer()
    return " ".join(stemmer.stem(kata) for kata in tokens)


teks_input = st.text_area(
    "Masukkan ulasan pengguna:",
    placeholder="Contoh: Aplikasi ini bagus, tetapi terkadang lambat."
)

if st.button("Analisis Sentimen", type="primary"):
    if not teks_input.strip():
        st.warning("Masukkan ulasan terlebih dahulu.")
    else:
        teks_bersih = preprocess_baru(teks_input)

        if not teks_bersih.strip():
            st.warning(
                "Teks tidak memiliki kata yang dapat dianalisis "
                "setelah preprocessing."
            )
        else:
            X_baru = vectorizer.transform([teks_bersih])

            prediksi = model.predict(X_baru)[0]
            probabilitas = model.predict_proba(X_baru)[0]

            st.subheader("Hasil Prediksi")

            st.metric("Sentimen", prediksi)

            st.write("**Teks setelah preprocessing:**")
            st.write(teks_bersih)

            df_prob = pd.DataFrame({
                "Sentimen": classes,
                "Probabilitas": probabilitas
            })

            df_prob["Persentase"] = (
                df_prob["Probabilitas"] * 100
            )

            fig_prob = px.bar(
                df_prob,
                x="Sentimen",
                y="Persentase",
                color="Sentimen",
                color_discrete_map=COLORS,
                text=df_prob["Persentase"].map(
                    lambda x: f"{x:.2f}%"
                ),
                title="Probabilitas Setiap Sentimen"
            )

            st.plotly_chart(
                fig_prob,
                use_container_width=True
            )

            st.dataframe(
                df_prob,
                use_container_width=True
            )

# =====================================================
# 9. INFORMASI METODOLOGI
# =====================================================

st.header("7. Metodologi")

st.markdown("""
- **Sumber data:** Ulasan Google Play Store aplikasi Ruangguru.
- **Pelabelan:** Rating 1–2 negatif, 3 netral, dan 4–5 positif.
- **Preprocessing:** Case folding, cleaning, normalisasi slang,
  stopword removal, dan stemming Sastrawi.
- **Ekstraksi fitur:** TF-IDF.
- **Penyeimbangan kelas:** RandomOverSampler pada data latih.
- **Algoritma:** Multinomial Naive Bayes.
- **Tuning:** GridSearchCV dengan F1-macro.
- **Evaluasi:** Accuracy, Precision, Recall, F1-Score,
  Confusion Matrix, ROC-AUC, dan Cross-Validation.
""")

st.caption(
    "Hasil dashboard bergantung pada dataset dan model yang "
    "dihasilkan dari notebook praktikum."
)