import argparse
import io
import time
import warnings
from pathlib import Path

import librosa
import mlflow
import numpy as np
import pandas as pd
import soundfile as sf
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_lahaja_data(data_path):
    path = Path(data_path)

    if path.is_file() and path.suffix.lower() == ".parquet":
        parquet_files = [path]
    else:
        parquet_files = sorted(path.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found under: {path.resolve()}"
        )

    frames = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    required = {"audio_filepath", "sp_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "lang" in df.columns:
        df = df[df["lang"].astype(str).str.lower().eq("hi")].copy()

    return df.reset_index(drop=True)


def read_audio(audio_object):
    if isinstance(audio_object, dict):
        audio_bytes = audio_object.get("bytes")
        audio_path = audio_object.get("path")

        if audio_bytes:
            audio, sr = sf.read(io.BytesIO(audio_bytes))
        elif audio_path and Path(audio_path).is_file():
            audio, sr = sf.read(audio_path)
        else:
            raise ValueError("Audio object has neither usable bytes nor path.")
    elif isinstance(audio_object, (str, Path)) and Path(audio_object).is_file():
        audio, sr = sf.read(str(audio_object))
    else:
        raise ValueError("Unsupported audio_filepath format.")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio = librosa.resample(
            audio.astype(np.float32), orig_sr=sr, target_sr=16000
        )
        sr = 16000

    return audio.astype(np.float32), sr


def extract_mfcc_sequence(audio_object, n_mfcc=13, max_frames=120):
    audio, sr = read_audio(audio_object)

    if len(audio) < int(sr * 0.25):
        raise ValueError("Audio clip is too short.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=400,
        hop_length=160,
    ).T

    if len(mfcc) > max_frames:
        indices = np.linspace(0, len(mfcc) - 1, max_frames).astype(int)
        mfcc = mfcc[indices]

    return mfcc.astype(np.float64)


def extract_dataset_features(df, max_samples):
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)

    sequences = []
    kept_rows = []

    for i, row in df.iterrows():
        try:
            seq = extract_mfcc_sequence(row["audio_filepath"])
            sequences.append(seq)
            kept_rows.append(row)
        except Exception as exc:
            print(f"Skipping row {i}: {exc}")

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)} recordings")

    if len(sequences) < 10:
        raise ValueError(
            "Too few usable audio recordings. Check the dataset path and audio column."
        )

    return pd.DataFrame(kept_rows).reset_index(drop=True), sequences


def run_hierarchical_optimization(df, sequences):
    print("\nUNIT 3 — Hierarchical Clustering Optimization")

    vectors = np.vstack([
        np.concatenate([seq.mean(axis=0), seq.std(axis=0)])
        for seq in sequences
    ])
    X = StandardScaler().fit_transform(vectors)

    cluster_counts = [k for k in [2, 3, 4, 5, 6, 8, 10, 12, 15, 18] if k < len(X)]
    linkages = ["ward", "complete", "average"]

    mlflow.set_experiment("LAHAJA_Unit3_Hierarchical_Clustering")

    results = []
    with mlflow.start_run(run_name="Hierarchical_Clustering_Optimization"):
        mlflow.log_param("feature_type", "mean_and_std_MFCC")
        mlflow.log_param("recordings", len(X))
        mlflow.log_param("mfcc_coefficients", 13)

        for linkage in linkages:
            for n_clusters in cluster_counts:
                start = time.time()
                with mlflow.start_run(
                    run_name=f"{linkage}_k{n_clusters}", nested=True
                ):
                    model = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage=linkage,
                    )
                    labels = model.fit_predict(X)
                    elapsed = time.time() - start
                    unique_labels = np.unique(labels)

                    if len(unique_labels) > 1 and len(unique_labels) < len(X):
                        silhouette = silhouette_score(
                            X, labels, sample_size=min(2000, len(X)),
                            random_state=42
                        )
                        db_score = davies_bouldin_score(X, labels)
                        ch_score = calinski_harabasz_score(X, labels)
                    else:
                        silhouette, db_score, ch_score = np.nan, np.nan, np.nan

                    mlflow.log_param("linkage", linkage)
                    mlflow.log_param("n_clusters", n_clusters)
                    mlflow.log_metric("silhouette_score", float(silhouette))
                    mlflow.log_metric("davies_bouldin_score", float(db_score))
                    mlflow.log_metric("calinski_harabasz_score", float(ch_score))
                    mlflow.log_metric("fit_time_seconds", elapsed)

                    results.append({
                        "linkage": linkage,
                        "n_clusters": n_clusters,
                        "silhouette_score": silhouette,
                        "davies_bouldin_score": db_score,
                        "calinski_harabasz_score": ch_score,
                        "fit_time_seconds": elapsed,
                    })

                    print(
                        f"{linkage:8s} k={n_clusters:2d} | "
                        f"Silhouette={silhouette:.4f} | "
                        f"DB={db_score:.4f} | CH={ch_score:.2f}"
                    )

    results_df = pd.DataFrame(results).sort_values(
        "silhouette_score", ascending=False
    )
    results_df.to_csv("unit3_hierarchical_results.csv", index=False)
    print("\nBest Unit 3 configuration:")
    print(results_df.iloc[0].to_string())
    print("Results saved to unit3_hierarchical_results.csv")


def run_hmm_optimization(df, sequences):
    print("\nUNIT 4 — Gaussian Hidden Markov Model Optimization")

    speaker_groups = df["sp_id"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(
        splitter.split(np.arange(len(sequences)), groups=speaker_groups)
    )

    train_sequences = [sequences[i] for i in train_idx]
    test_sequences = [sequences[i] for i in test_idx]

    scaler = StandardScaler()
    train_frames = np.vstack(train_sequences)
    scaler.fit(train_frames)

    train_scaled = [scaler.transform(seq) for seq in train_sequences]
    test_scaled = [scaler.transform(seq) for seq in test_sequences]

    train_lengths = [len(seq) for seq in train_scaled]
    X_train = np.vstack(train_scaled)

    mlflow.set_experiment("LAHAJA_Unit4_Gaussian_HMM")

    results = []
    state_counts = [3, 5, 8, 10, 12]

    with mlflow.start_run(run_name="Gaussian_HMM_Optimization"):
        mlflow.log_param("feature_type", "frame_level_MFCC")
        mlflow.log_param("train_recordings", len(train_sequences))
        mlflow.log_param("test_recordings", len(test_sequences))
        mlflow.log_param("split_type", "speaker_group_split")

        for n_states in state_counts:
            start = time.time()
            with mlflow.start_run(
                run_name=f"GaussianHMM_states{n_states}", nested=True
            ):
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=100,
                    tol=0.01,
                    random_state=42,
                    min_covar=1e-3,
                    verbose=False,
                )

                try:
                    model.fit(X_train, lengths=train_lengths)

                    total_log_likelihood = 0.0
                    total_frames = 0
                    for seq in test_scaled:
                        total_log_likelihood += model.score(seq)
                        total_frames += len(seq)

                    log_likelihood_per_frame = (
                        total_log_likelihood / total_frames
                    )
                    elapsed = time.time() - start
                    converged = bool(model.monitor_.converged)

                    mlflow.log_param("n_states", n_states)
                    mlflow.log_param("covariance_type", "diag")
                    mlflow.log_param("max_iterations", 100)
                    mlflow.log_metric(
                        "test_log_likelihood_per_frame",
                        float(log_likelihood_per_frame),
                    )
                    mlflow.log_metric("fit_time_seconds", elapsed)
                    mlflow.log_metric("converged", int(converged))
                    mlflow.log_metric(
                        "iterations_used", int(model.monitor_.iter)
                    )

                    results.append({
                        "n_states": n_states,
                        "test_log_likelihood_per_frame": log_likelihood_per_frame,
                        "converged": converged,
                        "iterations_used": int(model.monitor_.iter),
                        "fit_time_seconds": elapsed,
                    })

                    print(
                        f"States={n_states:2d} | "
                        f"Test log-likelihood/frame={log_likelihood_per_frame:.4f} | "
                        f"Converged={converged} | "
                        f"Iterations={model.monitor_.iter} | "
                        f"Time={elapsed:.1f}s"
                    )

                except Exception as exc:
                    mlflow.log_param("n_states", n_states)
                    mlflow.set_tag("run_status", "failed")
                    mlflow.set_tag("error", str(exc)[:500])
                    print(f"States={n_states}: failed — {exc}")

    if results:
        results_df = pd.DataFrame(results).sort_values(
            "test_log_likelihood_per_frame", ascending=False
        )
        results_df.to_csv("unit4_hmm_results.csv", index=False)
        print("\nBest Unit 4 configuration by held-out likelihood:")
        print(results_df.iloc[0].to_string())
        print("Results saved to unit4_hmm_results.csv")
    else:
        print("No HMM configuration completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize LAHAJA Hindi speech clustering and Gaussian HMM models."
    )
    parser.add_argument(
        "--data",
        default=r"D:\ML project\Lahaja dataset",
        help="Path to your LAHAJA dataset folder.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=800,
        help="Maximum recordings used for both models; use 0 for all recordings.",
    )
    parser.add_argument(
        "--model",
        choices=["both", "unit3", "unit4"],
        default="both",
        help="Run both models or only one unit.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI. The default stores tracking locally in mlflow.db.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    df = load_lahaja_data(args.data)
    print(f"Loaded {len(df)} Hindi recordings.")
    print(f"Unique speakers: {df['sp_id'].nunique()}")

    df, sequences = extract_dataset_features(
        df,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    print(f"Usable recordings: {len(sequences)}")

    if args.model in ("both", "unit3"):
        run_hierarchical_optimization(df, sequences)

    if args.model in ("both", "unit4"):
        run_hmm_optimization(df, sequences)

    print("\nCompleted. MLflow database and CSV results are in the current directory.")


if __name__ == "__main__":
    main()
