# Computes outlet-level soft cosine similarity and equal-sized bootstrap comparisons by rurality.

import argparse
from pathlib import Path

import gensim.downloader as api
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.similarities import SoftCosineSimilarity
from gensim.similarities.termsim import (
    SparseTermSimilarityMatrix,
    WordEmbeddingSimilarityIndex,
)
from gensim.utils import simple_preprocess


def find_corpus_file(path_string, corpus_dir):
    path = Path(path_string)

    if path.exists():
        return path

    matches = list(Path(corpus_dir).rglob(path.name))

    if matches:
        return matches[0]

    return None


def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def preprocess_text(text):
    tokens = simple_preprocess(
        str(text),
        deacc=True,
        min_len=3
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def load_data(metadata_path, corpus_dir):
    df = pd.read_csv(metadata_path)
    df.columns = df.columns.str.strip()

    df = df.rename(
        columns={
            "Newspaper ID": "newspaper_id",
            "Newspaper Name": "newspaper_name",
            "Rural": "rural",
            "Corpus": "corpus_file",
            "Corpus ": "corpus_file",
        }
    )

    required_cols = [
        "newspaper_id",
        "newspaper_name",
        "rural",
        "corpus_file",
    ]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=required_cols
    ).copy()

    df["rural"] = pd.to_numeric(
        df["rural"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["rural"]
    ).copy()

    df["rural"] = df["rural"].astype(int)

    df["actual_corpus_path"] = df[
        "corpus_file"
    ].apply(
        lambda x: find_corpus_file(
            x,
            corpus_dir
        )
    )

    df = df[
        df["actual_corpus_path"].notna()
    ].copy()

    df["combined_corpus"] = df[
        "actual_corpus_path"
    ].apply(read_text)

    df = df[
        df["combined_corpus"].str.len() > 0
    ].copy()

    df["tokens"] = df[
        "combined_corpus"
    ].apply(preprocess_text)

    df["token_count"] = df[
        "tokens"
    ].apply(len)

    df = df[
        df["token_count"] > 20
    ].copy()

    return df.reset_index(drop=True)


def build_soft_cosine_matrix(df):
    texts = df["tokens"].tolist()

    dictionary = Dictionary(texts)

    dictionary.filter_extremes(
        no_below=5,
        no_above=0.6,
        keep_n=10000
    )

    dictionary.compactify()

    bow_corpus = [
        dictionary.doc2bow(text)
        for text in texts
    ]

    tfidf_model = TfidfModel(
        bow_corpus
    )

    tfidf_corpus = [
        tfidf_model[doc]
        for doc in bow_corpus
    ]

    word_vectors = api.load(
        "glove-wiki-gigaword-50"
    )

    termsim_index = (
        WordEmbeddingSimilarityIndex(
            word_vectors
        )
    )

    similarity_matrix = (
        SparseTermSimilarityMatrix(
            termsim_index,
            dictionary,
            tfidf_model,
            nonzero_limit=50,
            dtype=np.float32
        )
    )

    soft_index = SoftCosineSimilarity(
        tfidf_corpus,
        similarity_matrix,
        num_best=None,
        chunksize=256
    )

    matrix = [
        np.asarray(
            soft_index[doc],
            dtype=np.float32
        )
        for doc in tfidf_corpus
    ]

    return np.vstack(matrix)


def create_pairwise_results(df, matrix):
    pairs = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            rural_i = df.loc[i, "rural"]
            rural_j = df.loc[j, "rural"]

            if rural_i == 1 and rural_j == 1:
                pair_type = "rural-rural"
            elif rural_i == 0 and rural_j == 0:
                pair_type = "urban-urban"
            else:
                pair_type = "rural-urban"

            pairs.append(
                {
                    "newspaper_id_1":
                        df.loc[i, "newspaper_id"],
                    "newspaper_name_1":
                        df.loc[i, "newspaper_name"],
                    "rural_1":
                        rural_i,
                    "newspaper_id_2":
                        df.loc[j, "newspaper_id"],
                    "newspaper_name_2":
                        df.loc[j, "newspaper_name"],
                    "rural_2":
                        rural_j,
                    "pair_type":
                        pair_type,
                    "soft_cosine_similarity":
                        matrix[i, j],
                }
            )

    return pd.DataFrame(pairs)


def summarize_pairwise(pairwise_df):
    return (
        pairwise_df
        .groupby(
            "pair_type"
        )["soft_cosine_similarity"]
        .agg(
            n_pairs="count",
            mean_soft_cosine="mean",
            median_soft_cosine="median",
            std_soft_cosine="std",
            min_soft_cosine="min",
            max_soft_cosine="max",
        )
        .reset_index()
    )


def bootstrap_pairwise(
    pairwise_df,
    n_iterations=5000,
    random_seed=42
):
    rng = np.random.default_rng(
        random_seed
    )

    group_counts = (
        pairwise_df[
            "pair_type"
        ].value_counts()
    )

    min_group_size = (
        group_counts.min()
    )

    pair_types = sorted(
        pairwise_df[
            "pair_type"
        ].unique()
    )

    bootstrap_results = []

    for iteration in range(
        n_iterations
    ):
        result = {
            "iteration":
                iteration + 1
        }

        for pair_type in pair_types:
            group_data = (
                pairwise_df[
                    pairwise_df[
                        "pair_type"
                    ] == pair_type
                ]
            )

            sample_index = rng.choice(
                group_data.index,
                size=min_group_size,
                replace=True
            )

            sampled = group_data.loc[
                sample_index
            ]

            values = sampled[
                "soft_cosine_similarity"
            ]

            result[
                f"{pair_type}_mean"
            ] = values.mean()

            result[
                f"{pair_type}_median"
            ] = values.median()

        bootstrap_results.append(
            result
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_results
    )

    summary_rows = []

    for pair_type in pair_types:
        mean_col = (
            f"{pair_type}_mean"
        )

        median_col = (
            f"{pair_type}_median"
        )

        summary_rows.append(
            {
                "pair_type":
                    pair_type,
                "bootstrap_n_per_iteration":
                    min_group_size,
                "iterations":
                    n_iterations,
                "mean_of_bootstrapped_means":
                    bootstrap_df[
                        mean_col
                    ].mean(),
                "median_of_bootstrapped_means":
                    bootstrap_df[
                        mean_col
                    ].median(),
                "lower_95_ci_mean":
                    bootstrap_df[
                        mean_col
                    ].quantile(0.025),
                "upper_95_ci_mean":
                    bootstrap_df[
                        mean_col
                    ].quantile(0.975),
                "mean_of_bootstrapped_medians":
                    bootstrap_df[
                        median_col
                    ].mean(),
                "lower_95_ci_median":
                    bootstrap_df[
                        median_col
                    ].quantile(0.025),
                "upper_95_ci_median":
                    bootstrap_df[
                        median_col
                    ].quantile(0.975),
            }
        )

    bootstrap_summary = pd.DataFrame(
        summary_rows
    )

    return (
        bootstrap_df,
        bootstrap_summary
    )


def main(
    metadata_path,
    corpus_dir,
    output_dir
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data(
        metadata_path,
        corpus_dir
    )

    matrix = (
        build_soft_cosine_matrix(
            df
        )
    )

    pairwise_df = (
        create_pairwise_results(
            df,
            matrix
        )
    )

    summary_df = (
        summarize_pairwise(
            pairwise_df
        )
    )

    bootstrap_df, bootstrap_summary = (
        bootstrap_pairwise(
            pairwise_df
        )
    )

    pairwise_df.to_csv(
        output_dir
        / "pairwise_soft_cosine.csv",
        index=False
    )

    summary_df.to_csv(
        output_dir
        / "soft_cosine_summary.csv",
        index=False
    )

    bootstrap_df.to_csv(
        output_dir
        / "bootstrap_iterations.csv",
        index=False
    )

    bootstrap_summary.to_csv(
        output_dir
        / "bootstrap_summary.csv",
        index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "metadata",
        help="Outlet metadata CSV."
    )

    parser.add_argument(
        "corpus_dir",
        help="Directory containing outlet corpus files."
    )

    parser.add_argument(
        "-o",
        "--output",
        default="results",
        help="Output directory."
    )

    args = parser.parse_args()

    main(
        args.metadata,
        args.corpus_dir,
        args.output
    )
