#!/usr/bin/env python3
"""
Player Clustering Script

Hierarchical clustering of NBA players by 6 RAPM factors (oTS, oTOV, oREB, dTS, dTOV, dREB).
Exports a JSON tree structure for Three.js bracket visualization.

Usage:
    python 04_cluster_players.py                          # Auto cluster count, JSON output
    python 04_cluster_players.py -k 8                     # Force 8 clusters
    python 04_cluster_players.py -i ../master_results/weighted_factors_23_26_all.csv
    python 04_cluster_players.py -p "Jokic" --top-n 15   # Quick similarity lookup
    python 04_cluster_players.py --min-poss 3000          # Higher possession filter
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Get the pipeline root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"
RESULTS_DIR = PIPELINE_ROOT / "results"
CLUSTERS_DIR = RESULTS_DIR / "clusters"

FACTORS = ['oTS', 'oTOV', 'oREB', 'dTS', 'dTOV', 'dREB']
DEFAULT_INPUT = MASTER_RESULTS_DIR / "weighted_factors_14_26_all_rb.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


###############################################################################
# 1) LOAD & FILTER
###############################################################################

def load_and_filter(filepath, min_poss):
    """Load weighted factors CSV and filter by minimum possessions."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} players from {filepath}")

    df_filtered = df[df['possessions'] >= min_poss].copy()
    logger.info(f"Filtered to {len(df_filtered)} players with {min_poss}+ possessions")

    # Verify required columns
    missing = [c for c in FACTORS if c not in df_filtered.columns]
    if missing:
        raise ValueError(f"Missing required factor columns: {missing}")

    return df_filtered


###############################################################################
# 2) SCALE & CLUSTER
###############################################################################

def find_optimal_k(Z, X_scaled, k_min=3, k_max=15):
    """Find optimal cluster count via silhouette score."""
    best_k, best_score = k_min, -1

    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, t=k, criterion='maxclust')
        score = silhouette_score(X_scaled, labels)
        logger.info(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score

    logger.info(f"Optimal k={best_k} (silhouette={best_score:.4f})")
    return best_k


def cluster_players(df, k=None):
    """Scale factors, run Ward's linkage, assign cluster labels."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FACTORS])

    logger.info("Computing Ward's linkage...")
    Z = linkage(X_scaled, method='ward', metric='euclidean')

    if k is None:
        logger.info("Auto-selecting cluster count via silhouette scoring...")
        k = find_optimal_k(Z, X_scaled)
    else:
        logger.info(f"Using user-specified k={k}")

    labels = fcluster(Z, t=k, criterion='maxclust')
    df = df.copy()
    df['cluster'] = labels

    return df, Z, X_scaled, scaler, k


###############################################################################
# 3) CLUSTER NAMING
###############################################################################

# Factor labels for naming: (factor, positive_label, negative_label)
FACTOR_LABELS = {
    'oTS': ('Scoring', 'Low-Scoring'),
    'oTOV': ('Ball-Secure', 'Turnover-Prone'),
    'oREB': ('Offensive-Boarding', 'Offensive-Boarding'),
    'dTS': ('Rim-Protecting', 'Poor-Defense'),
    'dTOV': ('Disruptive', 'Disruptive'),
    'dREB': ('Rebounding', 'Small'),
}


def name_clusters(df, scaler):
    """Generate archetype names using relative ranking between cluster centroids.

    Uses the two most distinctive factors (largest |z-score| relative to other
    clusters) to build a descriptive name. Compound patterns are checked first
    for basketball-meaningful labels.
    """
    cluster_ids = sorted(df['cluster'].unique())
    centroids_raw = df.groupby('cluster')[FACTORS].mean()

    # Z-score centroids relative to the cluster centroid distribution
    # This ensures names reflect how clusters differ from each other,
    # not from the player population (which yields weak absolute z-scores)
    centroids_z = centroids_raw.copy()
    for f in FACTORS:
        col = centroids_raw[f]
        std = col.std()
        if std > 0:
            centroids_z[f] = (col - col.mean()) / std
        else:
            centroids_z[f] = 0.0

    archetypes = {}
    for cid in cluster_ids:
        z = centroids_z.loc[cid]
        archetypes[cid] = _name_from_profile(z)

    # Deduplicate names by appending cluster ID if needed
    name_counts = {}
    for cid, name in archetypes.items():
        name_counts[name] = name_counts.get(name, 0) + 1
    seen = {}
    for cid in sorted(archetypes.keys()):
        name = archetypes[cid]
        if name_counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            archetypes[cid] = f"{name} {seen[name]}"

    df = df.copy()
    df['archetype'] = df['cluster'].map(archetypes)

    logger.info("Cluster archetypes:")
    for cid in cluster_ids:
        n = (df['cluster'] == cid).sum()
        z = centroids_z.loc[cid]
        z_str = ", ".join(f"{f}={z[f]:+.1f}" for f in FACTORS)
        logger.info(f"  Cluster {cid}: {archetypes[cid]} ({n} players) [{z_str}]")

    return df, archetypes, centroids_raw


def _name_from_profile(z):
    """Name a cluster from its z-score profile (relative to other clusters)."""
    # Compound patterns: check basketball-meaningful combinations
    off_factors = {f: z[f] for f in ['oTS', 'oTOV', 'oREB']}
    def_factors = {f: z[f] for f in ['dTS', 'dTOV', 'dREB']}
    off_sum = sum(off_factors.values())
    def_sum = sum(def_factors.values())

    # Two-Way Bigs: strong rebounding + defense
    if z['oREB'] > 0.5 and z['dREB'] > 0.5 and z['dTS'] > 0.3:
        return 'Two-Way Bigs'

    # Two-Way Stars: good offense + good defense
    if z['oTS'] > 0.5 and def_sum > 0.5:
        return 'Two-Way Stars'

    # Defensive Anchors: strong defense, weak offense
    if def_sum > 1.0 and off_sum < 0:
        return 'Defensive Anchors'

    # Elite Scorers: dominant offensive TS
    if z['oTS'] > 1.0:
        return 'Elite Scorers'

    # Perimeter Playmakers: good scoring + ball security, low rebounding
    if z['oTS'] > 0.3 and z['oTOV'] > 0.3 and z['oREB'] < 0:
        return 'Perimeter Playmakers'

    # Disruptive Glue Guys: high dTOV, moderate other
    if z['dTOV'] > 0.8:
        return 'Disruptive Glue Guys'

    # Offensive Bigs: oREB high, dREB high
    if z['oREB'] > 0.5 and z['dREB'] > 0.5:
        return 'Physical Bigs'

    # Fall back to top-2 most distinctive factors
    abs_z = z.abs().sort_values(ascending=False)
    tags = []
    for f in abs_z.index[:2]:
        pos_label, neg_label = FACTOR_LABELS[f]
        tags.append(pos_label if z[f] > 0 else neg_label)

    return ' + '.join(tags)


###############################################################################
# 4) NEAREST NEIGHBORS
###############################################################################

def compute_neighbors(X_scaled, df, top_n=10):
    """Compute top-N nearest neighbors for each player."""
    nn = NearestNeighbors(n_neighbors=min(top_n + 1, len(X_scaled)), metric='euclidean')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    player_ids = df['player_id'].values
    player_names = df['player_name'].values

    neighbors_list = []
    links_set = set()

    for i in range(len(df)):
        player_neighbors = []
        for j in range(1, distances.shape[1]):  # skip self
            neighbor_idx = indices[i, j]
            dist = round(float(distances[i, j]), 4)
            player_neighbors.append({
                'id': int(player_ids[neighbor_idx]),
                'name': player_names[neighbor_idx],
                'distance': dist
            })
            # Deduplicate links: store as sorted tuple
            pair = tuple(sorted([int(player_ids[i]), int(player_ids[neighbor_idx])]))
            links_set.add((*pair, dist))

        neighbors_list.append(player_neighbors)

    # Build deduplicated links array
    links = [{'source': s, 'target': t, 'distance': d} for s, t, d in links_set]
    links.sort(key=lambda x: x['distance'])

    return neighbors_list, links


###############################################################################
# 5) LINKAGE MATRIX → TREE JSON
###############################################################################

def linkage_to_tree(Z, df, archetypes):
    """Convert scipy linkage matrix to nested JSON tree."""
    n = len(df)
    player_ids = df['player_id'].values
    player_names = df['player_name'].values
    cluster_labels = df['cluster'].values

    # Build leaf nodes
    nodes = {}
    for i in range(n):
        nodes[i] = {
            'id': int(player_ids[i]),
            'name': player_names[i],
            'cluster': int(cluster_labels[i]),
            'leaf': True
        }

    # Build internal nodes from linkage matrix
    for i, row in enumerate(Z):
        left_idx = int(row[0])
        right_idx = int(row[1])
        distance = round(float(row[2]), 4)
        node_id = n + i

        left = nodes[left_idx]
        right = nodes[right_idx]

        # Determine if entire subtree belongs to one cluster
        left_cluster = _subtree_cluster(left)
        right_cluster = _subtree_cluster(right)
        if left_cluster is not None and right_cluster is not None and left_cluster == right_cluster:
            cluster = left_cluster
        else:
            cluster = None

        nodes[node_id] = {
            'id': f"node_{node_id}",
            'distance': distance,
            'cluster': cluster,
            'left': left,
            'right': right
        }

    # Root is the last internal node
    root = nodes[2 * n - 2]
    return root


def _subtree_cluster(node):
    """Return cluster ID if entire subtree is one cluster, else None."""
    if node.get('leaf'):
        return node['cluster']
    return node.get('cluster')


###############################################################################
# 6) BUILD JSON OUTPUT
###############################################################################

def build_json(df, Z, X_scaled, scaler, archetypes, centroids_raw, neighbors_list, links, source_file, k):
    """Assemble the complete JSON structure."""

    # Metadata
    metadata = {
        'n_players': len(df),
        'n_clusters': k,
        'factors': FACTORS,
        'linkage_method': 'ward',
        'distance_metric': 'euclidean',
        'min_possessions': int(df['possessions'].min()),
        'source_file': Path(source_file).name
    }

    # Tree
    tree = linkage_to_tree(Z, df, archetypes)

    # Players array
    players = []
    for i, (_, row) in enumerate(df.iterrows()):
        player = {
            'id': int(row['player_id']),
            'name': row['player_name'],
            'cluster': int(row['cluster']),
            'archetype': row['archetype'],
            'factors': {f: round(float(row[f]), 2) for f in FACTORS},
            'off': round(float(row['off']), 2),
            'def': round(float(row['def']), 2),
            'net_rapm': round(float(row['net_rapm']), 2),
            'possessions': int(row['possessions']),
            'latest_year': int(row['Latest_Year']),
            'neighbors': neighbors_list[i]
        }
        players.append(player)

    # Clusters array
    clusters = []
    for cid in sorted(df['cluster'].unique()):
        centroid = centroids_raw.loc[cid]
        clusters.append({
            'id': int(cid),
            'archetype': archetypes[cid],
            'n_players': int((df['cluster'] == cid).sum()),
            'centroid': {f: round(float(centroid[f]), 2) for f in FACTORS}
        })

    return {
        'metadata': metadata,
        'tree': tree,
        'players': players,
        'clusters': clusters,
        'links': links
    }


###############################################################################
# 7) PLAYER LOOKUP (CONSOLE)
###############################################################################

def player_lookup(df, X_scaled, query, top_n=10):
    """Print similar players to console."""
    # Fuzzy match on player name
    matches = df[df['player_name'].str.contains(query, case=False, na=False)]
    if matches.empty:
        print(f"No player found matching '{query}'")
        return

    if len(matches) > 1:
        print(f"Multiple matches for '{query}':")
        for _, row in matches.iterrows():
            print(f"  {row['player_name']} (ID: {row['player_id']}, {int(row['possessions'])} poss)")
        print(f"\nUsing first match: {matches.iloc[0]['player_name']}")

    target = matches.iloc[0]
    target_idx = df.index.get_loc(target.name)

    nn = NearestNeighbors(n_neighbors=min(top_n + 1, len(X_scaled)), metric='euclidean')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled[target_idx:target_idx + 1])

    print(f"\n{'='*70}")
    print(f"Player: {target['player_name']} (Cluster: {target.get('archetype', target['cluster'])})")
    print(f"Factors: " + ", ".join(f"{f}={target[f]:+.2f}" for f in FACTORS))
    print(f"RAPM: off={target['off']:+.2f}, def={target['def']:+.2f}, net={target['net_rapm']:+.2f}")
    print(f"Possessions: {int(target['possessions'])}")
    print(f"{'='*70}")
    print(f"\nTop {top_n} most similar players:")
    print(f"{'Rank':>4}  {'Player':<25} {'Distance':>8}  {'Cluster':<20}  {'Net RAPM':>8}")
    print(f"{'-'*4}  {'-'*25} {'-'*8}  {'-'*20}  {'-'*8}")

    for j in range(1, distances.shape[1]):
        idx = indices[0, j]
        row = df.iloc[idx]
        archetype = row.get('archetype', str(row['cluster']))
        print(f"{j:>4}  {row['player_name']:<25} {distances[0, j]:>8.4f}  {archetype:<20}  {row['net_rapm']:>+8.2f}")


###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Hierarchical clustering of NBA players by RAPM factors'
    )
    parser.add_argument('-i', '--input', type=str, default=str(DEFAULT_INPUT),
                        help='Input weighted_factors CSV file')
    parser.add_argument('-k', '--clusters', type=int, default=None,
                        help='Number of clusters (auto-select if omitted)')
    parser.add_argument('-p', '--player', type=str, default=None,
                        help='Player name for similarity lookup (prints to console)')
    parser.add_argument('--min-poss', type=int, default=2000,
                        help='Minimum possessions filter (default: 2000)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of nearest neighbors per player (default: 10)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON path (default: results/clusters/player_clusters.json)')
    args = parser.parse_args()

    # Load and filter
    df = load_and_filter(args.input, args.min_poss)

    # Cluster
    df, Z, X_scaled, scaler, k = cluster_players(df, k=args.clusters)

    # Name clusters
    df, archetypes, centroids_raw = name_clusters(df, scaler)

    # Player lookup mode (console only, no JSON export)
    if args.player:
        player_lookup(df, X_scaled, args.player, top_n=args.top_n)
        return

    # Compute neighbors
    logger.info(f"Computing top-{args.top_n} nearest neighbors...")
    neighbors_list, links = compute_neighbors(X_scaled, df, top_n=args.top_n)
    logger.info(f"Generated {len(links)} deduplicated similarity links")

    # Build JSON
    logger.info("Building JSON tree structure...")
    output = build_json(df, Z, X_scaled, scaler, archetypes, centroids_raw,
                        neighbors_list, links, args.input, k)

    # Write output
    out_path = Path(args.output) if args.output else CLUSTERS_DIR / "player_clusters.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Wrote {out_path} ({file_size_mb:.1f} MB)")
    logger.info(f"  {output['metadata']['n_players']} players, {output['metadata']['n_clusters']} clusters")


if __name__ == '__main__':
    main()
