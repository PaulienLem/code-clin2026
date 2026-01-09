from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import re
import time
from collections import Counter, defaultdict
from numba import njit, cuda

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, v_measure_score

@njit
def jaccard_numba(a_arr, b_arr):
    intersection = 0
    a_set = set(a_arr)
    b_set = set(b_arr)

    for item in a_set:
        if item in b_set:
            intersection += 1

    union = len(a_set) + len(b_set) - intersection
    if union == 0:
        return 0.0
    return intersection / union

@njit
def count_shared_verses(a_arr, b_arr):
    shared = 0
    a_set = set(a_arr)
    b_set = set(b_arr)

    for item in a_set:
        if item in b_set:
            shared += 1

    return shared

class PoemUnionFind:
    __slots__ = ['parent', 'rank']

    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def get_clusters(self):
        clusters = defaultdict(set)
        for elem in self.parent.keys():
            clusters[self.find(elem)].add(elem)
        return dict(clusters)

def compute_similarity_batch(args):
    pairs_batch, poem_to_array_dict, min_shared = args

    results = []
    for p1, p2 in pairs_batch:
        arr1 = poem_to_array_dict[p1]
        arr2 = poem_to_array_dict[p2]

        shared = count_shared_verses(arr1, arr2)

        if shared >= min_shared:
            sim = jaccard_numba(arr1, arr2)
            results.append({
                'poem1': p1,
                'poem2': p2,
                'similarity': sim,
                'shared_verses': shared
            })
    return results

class PoemThresholdSelector:
    def __init__(self, timinglogger, resource_monitor, system_analyzer, sample_size: int = 15000, random_seed: int = 42, min_shared_verses: int = 1):
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.min_shared_verses = min_shared_verses
        self.timing_logger = timinglogger
        self.resource_monitor = resource_monitor
        self.system_analyzer = system_analyzer
        np.random.seed(random_seed)

    @staticmethod
    def reconstruct_poems_vectorized(df):
        valid_mask = df['cluster_id'] != -1
        df_valid = df[valid_mask].copy()

        df_valid['idoriginal_poem'] = df_valid['idoriginal_poem'].astype(str)

        grouped = df_valid.groupby('idoriginal_poem')['cluster_id'].apply(
            lambda x: np.array(sorted(set(x)), dtype=np.int32)
        )

        return grouped.to_dict()

    @staticmethod
    def build_inverted_index_fast(poem_to_clusters):
        cluster_to_poems = defaultdict(set)
        for poem_id, clusters in poem_to_clusters.items():
            for c in clusters:
                cluster_to_poems[c].add(poem_id)
        return dict(cluster_to_poems)

    def find_candidate_pairs_for_sample(self, sample_poems, poem_to_clusters, cluster_to_poems):
        sample_set = set(sample_poems)
        candidate_pairs = set()

        poem_potential_matches = defaultdict(set)

        print("  Building potential matches using inverted index...")

        for cluster_id, poems_in_cluster in cluster_to_poems.items():
            sample_poems_in_cluster = list(poems_in_cluster & sample_set)

            if len(sample_poems_in_cluster) < 2:
                continue

            for poem in sample_poems_in_cluster:
                poem_potential_matches[poem].update(sample_poems_in_cluster)

        n_workers = self.system_analyzer.get_optimal_workers('io_intensive')

        def process_poem_batch(poems_batch):
            local_pairs = set()
            for poem_id in poems_batch:
                poem_id_str = str(poem_id)
                for other_poem in poem_potential_matches.get(poem_id, set()):
                    other_poem_str = str(other_poem)
                    if other_poem_str > poem_id_str:
                        local_pairs.add((poem_id_str, other_poem_str))
            return local_pairs

        poem_chunks = np.array_split(sample_poems, n_workers * 4)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            chunk_results = list(tqdm(executor.map(process_poem_batch, poem_chunks),
                                      total=len(poem_chunks), desc="Building sample pairs"))

        for chunk_pairs in chunk_results:
            candidate_pairs.update(chunk_pairs)

        return candidate_pairs

    def compute_sample_similarities(self, candidate_pairs, poem_to_array):
        pairs_list = list(candidate_pairs)

        if len(pairs_list) == 0:
            return pd.DataFrame()

        n_cores = self.system_analyzer.get_optimal_workers('cpu_intensive')
        chunk_size = self.system_analyzer.get_optimal_chunk_size(len(pairs_list), n_cores)
        chunks = [pairs_list[i:i + chunk_size] for i in range(0, len(pairs_list), chunk_size)]

        print(f"  Computing {len(pairs_list):,} pairs using {n_cores} cores with chunk size {chunk_size}...")

        poem_to_array_dict = {k: v for k, v in poem_to_array.items()}

        all_results = []
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            args_list = [(chunk, poem_to_array_dict, self.min_shared_verses) for chunk in chunks]
            futures = [executor.submit(compute_similarity_batch, args) for args in args_list]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing similarities"):
                all_results.extend(future.result())

        return pd.DataFrame(all_results)

    def stratified_sample_poems(self, df, poem_to_clusters):
        has_source = 'source_dataset' in df.columns

        if has_source:
            poem_to_source = df.groupby('idoriginal_poem')['source_dataset'].first().to_dict()

        poem_metadata = []
        for poem_id, clusters in poem_to_clusters.items():
            metadata = {
                'poem_id': poem_id,
                'n_clusters': len(clusters)
            }

            if has_source:
                metadata['source'] = poem_to_source.get(poem_id, 'unknown')

            poem_metadata.append(metadata)

        poem_df = pd.DataFrame(poem_metadata)

        poem_df['size_bin'] = pd.cut(poem_df['n_clusters'],
                                     bins=[0, 5, 10, 20, 50, np.inf],
                                     labels=['tiny', 'small', 'medium', 'large', 'huge'])

        sample_indices = []

        if has_source:
            print("  Stratifying by source dataset and poem size...")
            for (source, size_bin), group in poem_df.groupby(['source', 'size_bin']):
                n_in_group = len(group)
                proportion = n_in_group / len(poem_df)
                n_sample = max(1, int(self.sample_size * proportion))
                n_sample = min(n_sample, n_in_group)

                sampled = group.sample(n=n_sample, random_state=self.random_seed)
                sample_indices.extend(sampled['poem_id'].tolist())
        else:
            print("  Stratifying by poem size only...")
            for size_bin, group in poem_df.groupby('size_bin'):
                n_in_group = len(group)
                proportion = n_in_group / len(poem_df)
                n_sample = max(1, int(self.sample_size * proportion))
                n_sample = min(n_sample, n_in_group)

                sampled = group.sample(n=n_sample, random_state=self.random_seed)
                sample_indices.extend(sampled['poem_id'].tolist())

        if len(sample_indices) < self.sample_size:
            remaining = self.sample_size - len(sample_indices)
            available = set(poem_df['poem_id']) - set(sample_indices)
            if available:
                additional = np.random.choice(list(available),
                                              size=min(remaining, len(available)),
                                              replace=False)
                sample_indices.extend(additional)

        return sample_indices[:self.sample_size]

    def cluster_at_threshold(self, similarities_df, threshold, sample_poems, poem_to_array):
        valid_pairs = similarities_df[similarities_df['similarity'] >= threshold]

        uf = PoemUnionFind(sample_poems)

        for _, row in valid_pairs.iterrows():
            uf.union(row['poem1'], row['poem2'])

        poem_clusters = uf.get_clusters()
        cluster_assignments = {}
        for cluster_id, poems in poem_clusters.items():
            for poem in poems:
                cluster_assignments[poem] = cluster_id

        return cluster_assignments

    def compute_cluster_cohesion(self, poem_to_array, cluster_assignments, max_comparisons=10):
        poem_ids = list(poem_to_array.keys())
        cohesions = []

        clusters = defaultdict(list)
        for poem_id in poem_ids:
            cluster_id = cluster_assignments.get(poem_id)
            if cluster_id is not None:
                clusters[cluster_id].append(poem_id)

        for cluster_id, cluster_poems in clusters.items():
            if len(cluster_poems) < 2:
                continue

            if len(cluster_poems) > 20:
                sampled = np.random.choice(cluster_poems, 20, replace=False)
            else:
                sampled = cluster_poems

            sims = []
            for i in range(len(sampled)):
                for j in range(i + 1, min(i + 1 + max_comparisons, len(sampled))):
                    sim = jaccard_numba(poem_to_array[sampled[i]], poem_to_array[sampled[j]])
                    sims.append(sim)

            if sims:
                cohesions.append(np.mean(sims))

        return np.mean(cohesions) if cohesions else 0.0

    def compute_cluster_separation(self, poem_to_array, cluster_assignments, n_samples=200):
        poem_ids = list(poem_to_array.keys())
        unique_clusters = set(cluster_assignments.values())

        if len(unique_clusters) < 2:
            return 1.0

        separations = []
        cluster_to_poems = defaultdict(list)
        for poem_id, cluster_id in cluster_assignments.items():
            cluster_to_poems[cluster_id].append(poem_id)

        unique_clusters = list(unique_clusters)
        for _ in range(n_samples):
            c1, c2 = np.random.choice(unique_clusters, 2, replace=False)

            p1 = np.random.choice(cluster_to_poems[c1])
            p2 = np.random.choice(cluster_to_poems[c2])

            sim = jaccard_numba(poem_to_array[p1], poem_to_array[p2])
            separations.append(1 - sim)

        return np.mean(separations) if separations else 0.0

    def compute_silhouette_approximation(self, poem_to_array, cluster_assignments, n_samples=300):
        poem_ids = list(poem_to_array.keys())
        unique_clusters = set(cluster_assignments.values())

        if len(unique_clusters) < 2:
            return 0.0

        if len(poem_ids) > n_samples:
            sampled_poems = np.random.choice(poem_ids, n_samples, replace=False)
        else:
            sampled_poems = poem_ids

        silhouettes = []
        cluster_to_poems = defaultdict(list)
        for poem_id, cluster_id in cluster_assignments.items():
            cluster_to_poems[cluster_id].append(poem_id)

        convergence_window = 30
        convergence_threshold = 0.01

        for i, poem_id in enumerate(sampled_poems):
            cluster_id = cluster_assignments[poem_id]
            same_cluster = [p for p in cluster_to_poems[cluster_id] if p != poem_id]

            if len(same_cluster) == 0:
                continue

            if len(same_cluster) > 10:
                same_cluster = np.random.choice(same_cluster, 10, replace=False)

            a = np.mean([1 - jaccard_numba(poem_to_array[poem_id], poem_to_array[p])
                         for p in same_cluster])

            other_clusters = [c for c in unique_clusters if c != cluster_id]
            if len(other_clusters) == 0:
                continue

            min_b = float('inf')
            for other_cluster in other_clusters:
                other_poems = cluster_to_poems[other_cluster]

                if len(other_poems) > 10:
                    other_poems = np.random.choice(other_poems, 10, replace=False)

                b = np.mean([1 - jaccard_numba(poem_to_array[poem_id], poem_to_array[p])
                             for p in other_poems])
                min_b = min(min_b, b)

            s = (min_b - a) / max(a, min_b) if max(a, min_b) > 0 else 0
            silhouettes.append(s)

            if i > convergence_window and i % 30 == 0:
                recent_mean = np.mean(silhouettes[-convergence_window:])
                prev_mean = np.mean(silhouettes[-2 * convergence_window:-convergence_window])

                if abs(recent_mean - prev_mean) < convergence_threshold:
                    break

        return np.mean(silhouettes) if silhouettes else 0.0

    def evaluate_threshold(self, threshold, similarities_df, sample_poems, poem_to_array):
        cluster_assignments = self.cluster_at_threshold(
            similarities_df, threshold, sample_poems, poem_to_array
        )

        clusters = defaultdict(list)
        for poem_id, cluster_id in cluster_assignments.items():
            clusters[cluster_id].append(poem_id)

        n_clusters = len(clusters)
        cluster_sizes = [len(poems) for poems in clusters.values()]
        n_singletons = sum(1 for size in cluster_sizes if size == 1)
        avg_size = np.mean(cluster_sizes) if cluster_sizes else 0
        max_size = max(cluster_sizes) if cluster_sizes else 0

        cohesion = self.compute_cluster_cohesion(poem_to_array, cluster_assignments)
        separation = self.compute_cluster_separation(poem_to_array, cluster_assignments)
        silhouette = self.compute_silhouette_approximation(poem_to_array, cluster_assignments)

        n_pairs_above = len(similarities_df[similarities_df['similarity'] >= threshold])
        pct_pairs_above = (n_pairs_above / len(similarities_df) * 100) if len(similarities_df) > 0 else 0

        return {
            'threshold': threshold,
            'n_clusters': n_clusters,
            'n_singletons': n_singletons,
            'avg_cluster_size': avg_size,
            'max_cluster_size': max_size,
            'cohesion': cohesion,
            'separation': separation,
            'silhouette': silhouette,
            'n_pairs_above': n_pairs_above,
            'pct_pairs_above': pct_pairs_above
        }

    def grid_search_thresholds(self, similarities_df, sample_poems, poem_to_array):
        print("\n" + "=" * 70)
        print("ADAPTIVE GRID SEARCH: TWO-STAGE APPROACH")
        print("=" * 70)

        coarse_thresholds = np.linspace(0.01, 0.1, 7)
        print(f"Stage 1: Testing {len(coarse_thresholds)} coarse thresholds...")

        coarse_results = []
        for threshold in tqdm(coarse_thresholds, desc="Coarse search"):
            result = self.evaluate_threshold(threshold, similarities_df, sample_poems, poem_to_array)
            coarse_results.append(result)

        coarse_df = pd.DataFrame(coarse_results)

        def normalize(series):
            min_val = series.min()
            max_val = series.max()
            if max_val - min_val < 1e-10:
                return pd.Series(0.5, index=series.index)
            return (series - min_val) / (max_val - min_val)

        silhouette_score = normalize(coarse_df['silhouette'])
        cohesion_score = normalize(coarse_df['cohesion'])
        separation_score = normalize(coarse_df['separation'])
        singleton_ratio = coarse_df['n_singletons'] / len(sample_poems)
        balance_score = np.clip(1 - singleton_ratio, 0, 1)

        coarse_df['quality_score'] = (
                silhouette_score * 0.40 +
                cohesion_score * 0.30 +
                separation_score * 0.20 +
                balance_score * 0.10
        )

        best_idx = coarse_df['quality_score'].idxmax()
        best_coarse = coarse_df.loc[best_idx]
        best_thresh = best_coarse['threshold']

        print(f"  Best coarse threshold: {best_thresh:.3f} (quality: {best_coarse['quality_score']:.3f})")

        fine_range = 0.15
        fine_thresholds = np.linspace(
            max(0.3, best_thresh - fine_range),
            min(0.9, best_thresh + fine_range),
            9
        )

        print(f"Stage 2: Refining around {best_thresh:.3f} ± {fine_range}...")

        fine_results = []
        for threshold in tqdm(fine_thresholds, desc="Fine search"):
            if any(abs(r['threshold'] - threshold) < 0.01 for r in coarse_results):
                continue
            result = self.evaluate_threshold(threshold, similarities_df, sample_poems, poem_to_array)
            fine_results.append(result)

        all_results = coarse_results + fine_results
        results_df = pd.DataFrame(all_results)

        silhouette_score = normalize(results_df['silhouette'])
        cohesion_score = normalize(results_df['cohesion'])
        separation_score = normalize(results_df['separation'])
        singleton_ratio = results_df['n_singletons'] / len(sample_poems)
        balance_score = np.clip(1 - singleton_ratio, 0, 1)

        results_df['quality_score'] = (
                silhouette_score * 0.40 +
                cohesion_score * 0.30 +
                separation_score * 0.20 +
                balance_score * 0.10
        )

        results_df = results_df.sort_values('quality_score', ascending=False)
        results_df.to_csv('full_orthographic_results/poem_threshold_grid_search.csv', index=False)
        print(f"\nGrid search results saved")

        return results_df

    def plot_poem_grid_search_line(self, results_df, selected_threshold, results_folder):
        print("\nCreating poem-level line graph...")

        sns.set_palette("colorblind")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        thresholds = results_df['threshold'].values

        ax.plot(thresholds, results_df['quality_score'], 'o-',
                linewidth=2, markersize=8, color='#0173B2', label='Quality Score')
        ax.axvline(selected_threshold, color='#CC0000', linestyle='--', linewidth=2,
                   label=f'Selected: {selected_threshold:.3f}')
        ax.set_xlabel('Jaccard Similarity Threshold', fontweight='bold', fontsize=12)
        ax.set_ylabel('Quality Score', fontweight='bold', fontsize=12)
        ax.set_title('Poem-Level Quality Score vs Threshold', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        best_idx = results_df['quality_score'].idxmax()
        ax.scatter(results_df.loc[best_idx, 'threshold'],
                   results_df.loc[best_idx, 'quality_score'],
                   color='red', s=200, marker='*', edgecolors='black', linewidth=2, zorder=10)

        plt.tight_layout()
        plot_path = os.path.join(results_folder, 'poem_grid_search_line.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Poem line graph saved: {plot_path}")
        plt.close()

    def run_threshold_analysis(self, df):
        self.timing_logger.start_stage("02_poem_threshold_analysis")

        print("Poem level threshold selection")
        print(f"Minimum shared verses: {self.min_shared_verses}")

        poem_to_clusters = self.reconstruct_poems_vectorized(df)
        print(f"  Found {len(poem_to_clusters):,} poems")

        cluster_to_poems = self.build_inverted_index_fast(poem_to_clusters)
        print(f"  Found {len(cluster_to_poems):,} verse clusters")

        sample_poems = self.stratified_sample_poems(df, poem_to_clusters)
        print(f"  Sampled {len(sample_poems):,} poems")

        start_time = time.time()
        candidate_pairs = self.find_candidate_pairs_for_sample(
            sample_poems, poem_to_clusters, cluster_to_poems
        )
        print(f"  Found {len(candidate_pairs):,} candidate pairs in {time.time() - start_time:.1f}s")

        poem_to_array = {
            p: np.array(sorted(poem_to_clusters[p]), dtype=np.int32)
            for p in sample_poems
        }

        start_time = time.time()
        similarities_df = self.compute_sample_similarities(candidate_pairs, poem_to_array)
        print(f"  Computed {len(similarities_df):,} similarities in {time.time() - start_time:.1f}s")

        similarities_df.to_csv('full_orthographic_results/poem_similarities_sample.csv', index=False)

        grid_results = self.grid_search_thresholds(similarities_df, sample_poems, poem_to_array)

        best_result = grid_results.iloc[0]
        threshold = float(best_result['threshold'])

        print(f"Selected Threshold:   {threshold:.4f}")
        print(f"Quality Score:        {best_result['quality_score']:.4f}")
        print(f"Silhouette:           {best_result['silhouette']:.4f}")
        print(f"Cohesion:             {best_result['cohesion']:.4f}")
        print(f"Separation:           {best_result['separation']:.4f}")
        print(f"Clusters:             {int(best_result['n_clusters']):,}")
        print(f"Singletons:           {int(best_result['n_singletons']):,}")
        print(f"Avg Cluster Size:     {best_result['avg_cluster_size']:.2f}")

        print("\nStep 8: Creating visualizations...")
        self.plot_poem_grid_search_line(grid_results, threshold, 'full_orthographic_results')

        summary = {
            'selected_threshold': threshold,
            'quality_score': best_result['quality_score'],
            'silhouette': best_result['silhouette'],
            'cohesion': best_result['cohesion'],
            'separation': best_result['separation'],
            'n_clusters': int(best_result['n_clusters']),
            'n_singletons': int(best_result['n_singletons']),
            'avg_cluster_size': best_result['avg_cluster_size'],
            'min_shared_verses': self.min_shared_verses,
            'sample_size': len(sample_poems),
            'total_poems': len(poem_to_clusters)
        }

        pd.DataFrame([summary]).to_csv('full_orthographic_results/poem_enhanced_threshold_summary.csv', index=False)
        print(f"Summary saved")

        self.timing_logger.end_stage()
        return threshold, grid_results, similarities_df, poem_to_clusters