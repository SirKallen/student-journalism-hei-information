# Estimates rural-urban differences in article localness using logistic regression.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


OUTCOME_COL = "final_locality_binary"
RURAL_COL = "Rural"
OUTLET_COL = "Newspaper_ID"
ARTICLE_ID_COL = "article_id"


def prepare_data(input_csv):
    df = pd.read_csv(input_csv)

    required_cols = [
        ARTICLE_ID_COL,
        OUTCOME_COL,
        RURAL_COL,
        OUTLET_COL
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    df_model = df[required_cols].copy()

    df_model[OUTCOME_COL] = pd.to_numeric(
        df_model[OUTCOME_COL],
        errors="coerce"
    )

    df_model[RURAL_COL] = pd.to_numeric(
        df_model[RURAL_COL],
        errors="coerce"
    )

    df_model = df_model.dropna(
        subset=[
            OUTCOME_COL,
            RURAL_COL,
            OUTLET_COL
        ]
    )

    df_model[OUTCOME_COL] = (
        df_model[OUTCOME_COL].astype(int)
    )

    df_model[RURAL_COL] = (
        df_model[RURAL_COL].astype(int)
    )

    df_model[OUTLET_COL] = (
        df_model[OUTLET_COL].astype(str)
    )

    return df_model


def descriptive_summary(df_model):
    summary = (
        df_model
        .groupby(RURAL_COL)
        .agg(
            n_articles=(ARTICLE_ID_COL, "count"),
            n_outlets=(OUTLET_COL, "nunique"),
            n_locality=(OUTCOME_COL, "sum"),
            locality_rate=(OUTCOME_COL, "mean")
        )
        .reset_index()
    )

    summary["group"] = summary[RURAL_COL].map(
        {
            0: "Urban",
            1: "Town/Rural"
        }
    )

    return summary[
        [
            "group",
            RURAL_COL,
            "n_articles",
            "n_outlets",
            "n_locality",
            "locality_rate"
        ]
    ]


def fit_logit(df_model, clustered=False):
    y = df_model[OUTCOME_COL]

    X = sm.add_constant(
        df_model[[RURAL_COL]]
    )

    model = sm.Logit(
        y,
        X
    )

    if clustered:
        result = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups":
                    df_model[OUTLET_COL]
            },
            disp=False
        )
    else:
        result = model.fit(
            disp=False
        )

    return result


def regression_table(result, clustered=False):
    conf = result.conf_int()

    table = pd.DataFrame(
        {
            "coefficient": result.params,
            "standard_error": result.bse,
            "odds_ratio": np.exp(
                result.params
            ),
            "CI_lower_OR": np.exp(
                conf[0]
            ),
            "CI_upper_OR": np.exp(
                conf[1]
            ),
            "p_value": result.pvalues
        }
    )

    table["model"] = (
        "clustered"
        if clustered
        else "unclustered"
    )

    table["variable"] = (
        table.index
    )

    return table.reset_index(
        drop=True
    )[
        [
            "model",
            "variable",
            "coefficient",
            "standard_error",
            "odds_ratio",
            "CI_lower_OR",
            "CI_upper_OR",
            "p_value"
        ]
    ]


def predicted_probabilities(result):
    pred_df = pd.DataFrame(
        {
            "const": [1, 1],
            RURAL_COL: [0, 1]
        }
    )

    pred_df["group"] = [
        "Urban",
        "Town/Rural"
    ]

    pred_df[
        "predicted_probability_locality"
    ] = result.predict(
        pred_df[
            ["const", RURAL_COL]
        ]
    )

    return pred_df[
        [
            "group",
            RURAL_COL,
            "predicted_probability_locality"
        ]
    ]


def main(input_csv, output_dir):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df_model = prepare_data(
        input_csv
    )

    summary = descriptive_summary(
        df_model
    )

    result_unclustered = fit_logit(
        df_model,
        clustered=False
    )

    result_clustered = fit_logit(
        df_model,
        clustered=True
    )

    regression_results = pd.concat(
        [
            regression_table(
                result_unclustered,
                clustered=False
            ),
            regression_table(
                result_clustered,
                clustered=True
            )
        ],
        ignore_index=True
    )

    predictions = predicted_probabilities(
        result_clustered
    )

    summary.to_csv(
        output_dir
        / "locality_descriptive_summary.csv",
        index=False
    )

    regression_results.to_csv(
        output_dir
        / "locality_logistic_regression.csv",
        index=False
    )

    predictions.to_csv(
        output_dir
        / "locality_predicted_probabilities.csv",
        index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        help="CSV containing article-level locality and rurality variables."
    )

    parser.add_argument(
        "-o",
        "--output",
        default="results",
        help="Output directory."
    )

    args = parser.parse_args()

    main(
        args.input,
        args.output
    )
