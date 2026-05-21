#!/usr/bin/env python3
"""
Build a reusable 3D galaxy payload from any weighted_factors CSV.

Design:
  - Similarity edges come from the selected factor space (scaled raw features by default)
  - Clusters default to the scaled raw factor space, with optional latent clustering
  - Visible coordinates come from a separate 3D UMAP layout

Outputs:
  - galaxy JSON compatible with the existing node/edge shape used by the web app
  - embeddings CSV with latent coordinates, layout coordinates, and cluster labels
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PIPELINE_ROOT / "results"
GALAXY_DIR = RESULTS_DIR / "galaxy"
DEFAULT_INPUT = PIPELINE_ROOT / "master_results" / "weighted_factors_14_26_all_rb.csv"

RANDOM_STATE = 42
PCA_COMPONENTS = 15
UMAP_LATENT_DIM = 5
UMAP_N_NEIGHBORS = 25
UMAP_MIN_DIST = 0.1
UMAP_3D_N_NEIGHBORS = 30
UMAP_3D_MIN_DIST = 0.3
HDBSCAN_MIN_CLUSTER_SIZE = 25
HDBSCAN_MIN_SAMPLES = 5
DEFAULT_TOP_K = 15

logger = logging.getLogger(__name__)

CLUSTER_NAME_PROFILES: dict[str, dict[int, str]] = {
    "eight_factor_kmeans_10_v1": {
        0: "Offensive Engines",
        1: "Replacement Level",
        2: "Scrappy Disruptors",
        3: "Defensive Playmakers",
        4: "Floor Spacers",
        5: "Defense-First Bigs",
        6: "Glass Crashers",
        7: "General Rotation",
        8: "Defensive Bigs",
        9: "Skill Guards",
    }
}


FACTOR_LABELS: dict[str, tuple[str, str]] = {
    "oTS": ("Scoring", "Low-Scoring"),
    "oEFG": ("Shotmaking", "Low-Accuracy"),
    "oFT": ("Foul-Pressure", "Low-FT-Pressure"),
    "oTOV": ("Ball-Secure", "Turnover-Prone"),
    "oREB": ("Offensive-Boarding", "Low-OREB"),
    "dTS": ("Rim-Protecting", "Poor-Defense"),
    "dEFG": ("Shot-Suppressing", "Leaky-Defense"),
    "dFT": ("Low-Fouling", "Foul-Prone"),
    "dTOV": ("Disruptive", "Low-Disruption"),
    "dREB": ("Rebounding", "Small"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 3D galaxy from a weighted_factors CSV.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Input weighted_factors CSV file.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["six_factor", "eight_factor", "custom"],
        default="six_factor",
        help="Feature preset to use for similarity + layout.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated custom feature list. Required with --feature-set custom.",
    )
    parser.add_argument(
        "--min-poss",
        type=int,
        default=0,
        help="Minimum possessions filter before building the galaxy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Nearest neighbors per node before edge dedupe.",
    )
    parser.add_argument(
        "--neighbor-space",
        choices=["raw", "latent"],
        default="raw",
        help="Use scaled raw feature space or 5D latent space for neighbor edges.",
    )
    parser.add_argument(
        "--cluster-space",
        choices=["raw", "latent"],
        default="raw",
        help="Use scaled raw feature space or 5D latent space for clustering.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "agglomerative", "hdbscan"],
        default="kmeans",
        help="Clustering algorithm for archetype buckets.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Fixed cluster count for kmeans/agglomerative clustering.",
    )
    parser.add_argument(
        "--cluster-name-profile",
        choices=["auto", *CLUSTER_NAME_PROFILES.keys()],
        default="auto",
        help="Optional manual display-name override profile for cluster ids.",
    )
    parser.add_argument(
        "--net-weight",
        type=float,
        default=0.5,
        help="Extra weight for standardized net_rapm as an added distance/clustering dimension.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON path. Defaults under results/galaxy/.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output embeddings CSV path. Defaults under results/galaxy/.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return (
        value.lower()
        .replace("&", " and ")
        .replace("/", " ")
        .replace("-", " ")
        .replace(".", "")
        .replace("'", "")
        .strip()
        .replace(" ", "_")
    )


def get_feature_columns(args: argparse.Namespace) -> list[str]:
    if args.feature_set == "six_factor":
        return ["oTS", "oTOV", "oREB", "dTS", "dTOV", "dREB"]
    if args.feature_set == "eight_factor":
        return ["oEFG", "oFT", "oTOV", "oREB", "dEFG", "dFT", "dTOV", "dREB"]
    if not args.features:
        raise ValueError("--features is required when --feature-set custom.")
    return [feature.strip() for feature in args.features.split(",") if feature.strip()]


def derive_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    working = df.copy()

    if "oEFG" in feature_columns and "oEFG" not in working.columns:
        required = {"oTS", "oFT"}
        if not required.issubset(working.columns):
            raise ValueError("Requested oEFG but input file is missing oTS/oFT.")
        working["oEFG"] = working["oTS"] - working["oFT"]

    if "dEFG" in feature_columns and "dEFG" not in working.columns:
        required = {"dTS", "dFT"}
        if not required.issubset(working.columns):
            raise ValueError("Requested dEFG but input file is missing dTS/dFT.")
        working["dEFG"] = working["dTS"] - working["dFT"]

    missing = [column for column in feature_columns if column not in working.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return working


def load_and_prepare(input_path: Path, min_poss: int, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(input_path)
    logger.info("Loaded %s rows from %s", len(df), input_path)

    if min_poss > 0:
        if "possessions" not in df.columns:
            raise ValueError("Cannot apply --min-poss because the input file has no possessions column.")
        df = df[df["possessions"].fillna(0) >= min_poss].copy()
        logger.info("Filtered to %s rows with possessions >= %s", len(df), min_poss)

    df = derive_feature_columns(df, feature_columns)

    if "player_name" not in df.columns:
        raise ValueError("Input file must contain player_name.")

    if "player_id" not in df.columns:
        logger.warning("Input file has no player_id column; ids will be name-based.")

    latest_year_series = (
        pd.to_numeric(df["Latest_Year"], errors="coerce").fillna(0).astype(int)
        if "Latest_Year" in df.columns
        else pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    )
    ids = []
    for idx, row in df.iterrows():
        player_name = str(row["player_name"]).strip()
        player_id = row.get("player_id")
        if pd.notna(player_id):
            ids.append(str(int(player_id)))
        else:
            ids.append(f"{slugify(player_name)}_{latest_year_series.loc[idx]}")
    df["galaxy_id"] = ids

    feature_frame = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
    return df.reset_index(drop=True), feature_columns


def build_scaled_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    net_weight: float,
) -> tuple[np.ndarray, StandardScaler | None]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_columns].values.astype(float))

    if net_weight == 0:
        return scaled, None

    if "net_rapm" not in df.columns:
        raise ValueError("Requested nonzero --net-weight but the input file is missing net_rapm.")

    net_series = pd.to_numeric(df["net_rapm"], errors="coerce").fillna(pd.to_numeric(df["net_rapm"], errors="coerce").median())
    net_scaled = StandardScaler().fit_transform(net_series.to_numpy().reshape(-1, 1)).reshape(-1)
    weighted_net = (net_scaled * net_weight).reshape(-1, 1)
    combined = np.concatenate([scaled, weighted_net], axis=1)
    return combined, scaler


def compute_embeddings(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_components = min(PCA_COMPONENTS, X_scaled.shape[1])
    X_pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(X_scaled)

    latent = umap.UMAP(
        n_components=UMAP_LATENT_DIM,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
    ).fit_transform(X_pca)

    layout_3d = umap.UMAP(
        n_components=3,
        n_neighbors=UMAP_3D_N_NEIGHBORS,
        min_dist=UMAP_3D_MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
    ).fit_transform(X_pca)
    layout_3d = layout_3d - layout_3d.mean(axis=0)

    return latent, layout_3d


def cluster_space(space: np.ndarray, method: str, n_clusters: int) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_STATE).fit_predict(space)
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(space)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(space)


def build_knn_graph(space: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(space) <= 1:
        raise ValueError("Need at least two rows to build a graph.")
    neighbors = min(top_k + 1, len(space))
    nn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
    nn.fit(space)
    distances, indices = nn.kneighbors(space)
    return indices, distances


def build_cluster_summary(
    df: pd.DataFrame,
    feature_columns: list[str],
    labels: np.ndarray,
    cluster_name_profile: str,
) -> tuple[dict[int, str], list[dict[str, object]]]:
    if len(df) != len(labels):
        raise ValueError("Cluster labels length does not match dataframe length.")

    work = df.copy()
    work["cluster"] = labels
    cluster_rows: list[dict[str, object]] = []
    cluster_names: dict[int, str] = {}

    clusters_only = [cluster_id for cluster_id in sorted(work["cluster"].unique()) if cluster_id != -1]
    if not clusters_only:
        return cluster_names, cluster_rows

    centroids = work.loc[work["cluster"] != -1].groupby("cluster")[feature_columns].mean()
    centroid_z = centroids.copy()
    for column in feature_columns:
        std = centroids[column].std()
        centroid_z[column] = 0.0 if std == 0 else (centroids[column] - centroids[column].mean()) / std

    used_names: dict[str, int] = {}
    manual_names = CLUSTER_NAME_PROFILES.get(cluster_name_profile, {}) if cluster_name_profile != "auto" else {}
    for cluster_id in clusters_only:
        z = centroid_z.loc[cluster_id]
        top_features = z.abs().sort_values(ascending=False).index[:2].tolist()
        base_name_parts = []
        for feature in top_features:
            positive_label, negative_label = FACTOR_LABELS.get(
                feature,
                (feature.replace("_", " "), feature.replace("_", " ")),
            )
            base_name_parts.append(positive_label if z[feature] >= 0 else negative_label)
        base_name = " + ".join(base_name_parts)
        used_names[base_name] = used_names.get(base_name, 0) + 1
        auto_name = base_name if used_names[base_name] == 1 else f"{base_name} {used_names[base_name]}"
        name = manual_names.get(int(cluster_id), auto_name)
        cluster_names[int(cluster_id)] = name

        members = work[work["cluster"] == cluster_id]
        sample_players = members["player_name"].head(5).tolist()
        cluster_rows.append(
            {
                "id": int(cluster_id),
                "name": name,
                "n_players": int(len(members)),
                "dominant_features": top_features,
                "centroid": {column: round(float(centroids.loc[cluster_id, column]), 4) for column in feature_columns},
                "sample_players": sample_players,
            }
        )

    return cluster_names, cluster_rows


def build_json(
    df: pd.DataFrame,
    feature_columns: list[str],
    similarity_columns: list[str],
    similarity_dim: int,
    latent: np.ndarray,
    layout_3d: np.ndarray,
    labels: np.ndarray,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    cluster_names: dict[int, str],
    cluster_rows: list[dict[str, object]],
    args: argparse.Namespace,
    input_path: Path,
) -> dict[str, object]:
    nodes: list[dict[str, object]] = []
    for i, row in df.iterrows():
        factor_values = {column: round(float(row[column]), 4) for column in feature_columns}
        cluster_id = int(labels[i])
        nodes.append(
            {
                "id": row["galaxy_id"],
                "position": {
                    "x": float(layout_3d[i, 0]),
                    "y": float(layout_3d[i, 1]),
                    "z": float(layout_3d[i, 2]),
                },
                "cluster": cluster_id,
                "data": {
                    "label": row["player_name"],
                    "player_id": None if pd.isna(row.get("player_id")) else int(row.get("player_id")),
                    "latest_year": None if pd.isna(row.get("Latest_Year")) else int(row.get("Latest_Year")),
                    "possessions": None if pd.isna(row.get("possessions")) else int(row.get("possessions")),
                    "off": None if pd.isna(row.get("off")) else float(row.get("off")),
                    "def": None if pd.isna(row.get("def")) else float(row.get("def")),
                    "net_rapm": None if pd.isna(row.get("net_rapm")) else float(row.get("net_rapm")),
                    "archetype": cluster_names.get(cluster_id),
                    "feature_values": factor_values,
                },
            }
        )

    edges: list[dict[str, object]] = []
    edge_set: set[tuple[str, str]] = set()
    ids = df["galaxy_id"].tolist()
    for i, neighbors in enumerate(knn_indices):
        for neighbor_offset, j in enumerate(neighbors[1:], start=1):
            source = ids[i]
            target = ids[j]
            key = tuple(sorted((source, target)))
            if key in edge_set:
                continue
            edge_set.add(key)
            edges.append(
                {
                    "id": f"e{len(edges)}",
                    "source": source,
                    "target": target,
                    "distance": float(knn_distances[i, neighbor_offset]),
                }
            )

    return {
        "nodes": nodes,
        "edges": edges,
        "clusters": cluster_rows,
        "metadata": {
            "version": 1,
            "model": "weighted_factors_galaxy",
            "feature_set": args.feature_set,
            "features": feature_columns,
            "similarity_features": similarity_columns,
            "source_file": input_path.name,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_clusters": len(cluster_rows),
            "noise_points": int((labels == -1).sum()),
            "neighbor_space": args.neighbor_space,
            "cluster_space": args.cluster_space,
            "cluster_method": args.cluster_method,
            "requested_n_clusters": args.n_clusters if args.cluster_method != "hdbscan" else None,
            "cluster_name_profile": args.cluster_name_profile,
            "min_possessions_filter": args.min_poss,
            "top_k_neighbors": args.top_k,
            "net_weight": args.net_weight,
            "latent_method": {
                "pca_components": min(PCA_COMPONENTS, similarity_dim),
                "umap_components": UMAP_LATENT_DIM,
                "umap_neighbors": UMAP_N_NEIGHBORS,
                "umap_min_dist": UMAP_MIN_DIST,
            },
            "layout_method": {
                "umap_components": 3,
                "umap_neighbors": UMAP_3D_N_NEIGHBORS,
                "umap_min_dist": UMAP_3D_MIN_DIST,
            },
            "clustering_method": {
                "name": args.cluster_method,
                "n_clusters": args.n_clusters if args.cluster_method != "hdbscan" else None,
                "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE if args.cluster_method == "hdbscan" else None,
                "min_samples": HDBSCAN_MIN_SAMPLES if args.cluster_method == "hdbscan" else None,
            },
        },
    }


def build_embeddings_csv(
    df: pd.DataFrame,
    feature_columns: list[str],
    latent: np.ndarray,
    layout_3d: np.ndarray,
    labels: np.ndarray,
    cluster_names: dict[int, str],
) -> pd.DataFrame:
    out = df.copy()
    for idx in range(latent.shape[1]):
        out[f"latent_{idx + 1}"] = latent[:, idx]
    out["x"] = layout_3d[:, 0]
    out["y"] = layout_3d[:, 1]
    out["z"] = layout_3d[:, 2]
    out["cluster"] = labels
    out["cluster_name"] = [cluster_names.get(int(label)) for label in labels]
    ordered_columns = [
        "galaxy_id",
        "player_id",
        "player_name",
        "Latest_Year",
        "possessions",
        *feature_columns,
        "cluster",
        "cluster_name",
        *[f"latent_{idx + 1}" for idx in range(latent.shape[1])],
        "x",
        "y",
        "z",
    ]
    remaining = [column for column in out.columns if column not in ordered_columns]
    return out[[column for column in ordered_columns if column in out.columns] + remaining]


def default_output_paths(input_path: Path, feature_set: str) -> tuple[Path, Path]:
    stem = input_path.stem
    suffix = feature_set.replace("_factor", "f")
    return (
        GALAXY_DIR / f"{stem}_{suffix}_galaxy.json",
        GALAXY_DIR / f"{stem}_{suffix}_embeddings.csv",
    )


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    configure_logging()
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    feature_columns = get_feature_columns(args)
    df, feature_columns = load_and_prepare(input_path, args.min_poss, feature_columns)

    if len(df) < 2:
        raise ValueError("Need at least two players after filtering.")

    X_scaled, _ = build_scaled_matrix(df, feature_columns, args.net_weight)
    latent, layout_3d = compute_embeddings(X_scaled)
    cluster_input = X_scaled if args.cluster_space == "raw" else latent
    labels = cluster_space(cluster_input, args.cluster_method, args.n_clusters)

    neighbor_space = X_scaled if args.neighbor_space == "raw" else latent
    knn_indices, knn_distances = build_knn_graph(neighbor_space, args.top_k)

    cluster_names, cluster_rows = build_cluster_summary(df, feature_columns, labels, args.cluster_name_profile)
    similarity_columns = feature_columns.copy()
    if args.net_weight != 0:
        similarity_columns.append(f"net_rapm*{args.net_weight:g}")

    payload = build_json(
        df,
        feature_columns,
        similarity_columns,
        X_scaled.shape[1],
        latent,
        layout_3d,
        labels,
        knn_indices,
        knn_distances,
        cluster_names,
        cluster_rows,
        args,
        input_path,
    )
    embeddings_df = build_embeddings_csv(df, feature_columns, latent, layout_3d, labels, cluster_names)

    default_json, default_csv = default_output_paths(input_path, args.feature_set)
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else default_json
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else default_csv
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w") as handle:
        json.dump(payload, handle)
    embeddings_df.to_csv(output_csv, index=False)

    cluster_sizes = sorted(
        ((cluster["id"], cluster["n_players"]) for cluster in cluster_rows),
        key=lambda item: item[1],
        reverse=True,
    )
    logger.info("Wrote %s", output_json)
    logger.info("Wrote %s", output_csv)
    logger.info(
        "Galaxy summary: %s nodes, %s edges, %s clusters, %s noise points",
        payload["metadata"]["n_nodes"],
        payload["metadata"]["n_edges"],
        payload["metadata"]["n_clusters"],
        payload["metadata"]["noise_points"],
    )
    if cluster_sizes:
        logger.info("Largest clusters: %s", cluster_sizes[:8])


if __name__ == "__main__":
    main()
