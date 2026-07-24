# ==========================================
# K-MEANS CLUSTERING TOWER SUTT MANOKWARI
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_and_preprocess_data(file_path):
    """
    Load dan preprocessing dataset
    """

    # Load dataset
    df = pd.read_csv(file_path, decimal=",")

    # Membersihkan nama kolom
    df.columns = df.columns.str.strip()

    # -----------------------------
    # Handling Data Numerik
    # -----------------------------

    # Jarak antar tower
    df["Jarak  antar tower(meter)"] = pd.to_numeric(
        df["Jarak  antar tower(meter)"],
        errors="coerce"
    ).fillna(0)

    # Elevasi
    df["elevasi permukaan tanah"] = pd.to_numeric(
        df["elevasi permukaan tanah"],
        errors="coerce"
    )

    df["elevasi permukaan tanah"] = (
        df["elevasi permukaan tanah"]
        .fillna(df["elevasi permukaan tanah"].mean())
    )

    # -----------------------------
    # Label Encoding
    # -----------------------------
    le_tutupan = LabelEncoder()
    le_tower = LabelEncoder()

    df["Tutupan_Lahan_Encoded"] = le_tutupan.fit_transform(
        df["Tutupan Lahan"]
    )

    df["Jenis_Tower_Encoded"] = le_tower.fit_transform(
        df["Jenis_tower"]
    )

    return df


def prepare_features(df):
    """
    Seleksi fitur dan standardisasi
    """

    fitur_pilihan = [
        "Tutupan_Lahan_Encoded",
        "Biaya_Pembebasan_Lahan",
        "Jarak  antar tower(meter)",
        "elevasi permukaan tanah",
        "jarak ke jalan",
        "Biaya_Struktur_Tower",
        "jarak kesungai",
        "Jenis_Tower_Encoded"
    ]

    X = df[fitur_pilihan].copy()

    for col in fitur_pilihan:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def find_best_k(X_scaled):
    """
    Menentukan K terbaik menggunakan Silhouette Score
    """

    silhouette_scores = {}

    print("\n--- Pencarian K Optimal ---")

    for k in range(2, 8):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels
        )

        silhouette_scores[k] = score

        print(
            f"K = {k} | Silhouette Score = {score:.4f}"
        )

    best_k = max(
        silhouette_scores,
        key=silhouette_scores.get
    )

    print(
        f"\n>> K Optimal = {best_k} "
        f"(Score = {silhouette_scores[best_k]:.4f})"
    )

    return best_k


def elbow_method(X_scaled):
    """
    Elbow Method
    """

    inertia_values = []

    for k in range(2, 8):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        inertia_values.append(
            model.inertia_
        )

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(2, 8),
        inertia_values,
        marker="o"
    )

    plt.title("Elbow Method")
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Inertia")
    plt.grid(True)

    plt.show()


def compare_k(X_scaled):
    """
    Perbandingan K=2 dan K=3
    """

    print("\n--- Perbandingan K ---")

    for k in [2, 3]:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels
        )

        print(
            f"K={k} | Silhouette Score={score:.4f}"
        )


def stability_test(X_scaled):
    """
    Uji stabilitas cluster
    """

    print(
        "\n--- Uji Stabilitas Random State ---"
    )

    for seed in range(10):

        model = KMeans(
            n_clusters=2,
            random_state=seed,
            n_init=10
        )

        labels = model.fit_predict(
            X_scaled
        )

        cluster_counts = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
        )

        print(
            f"\nRandom State = {seed}"
        )

        print(cluster_counts)


def final_clustering(
    df,
    X_scaled,
    best_k
):
    """
    Clustering final
    """

    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(
        X_scaled
    )

    return df


def main():

    file_path = "Tower SUTT Manokwari.csv"

    # Load data
    df = load_and_preprocess_data(
        file_path
    )

    # Feature selection
    X_scaled = prepare_features(df)

    # Cari K terbaik
    best_k = find_best_k(
        X_scaled
    )

    # Elbow Method
    elbow_method(
        X_scaled
    )

    # Perbandingan K
    compare_k(
        X_scaled
    )

    # Uji Stabilitas
    stability_test(
        X_scaled
    )

    # Clustering Final
    df = final_clustering(
        df,
        X_scaled,
        best_k
    )

    # Simpan hasil
    output_file = (
        "hasil_cluster_tower_sutt.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nHasil clustering disimpan ke: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()