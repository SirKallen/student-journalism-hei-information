# Merges institution records with IPEDS locale data to assign a rurality indicator to each HEI.

import argparse
import re
import pandas as pd


def clean_name(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def main(outlet_path, crosswalk_path, output_path):
    outlets = pd.read_csv(outlet_path)
    crosswalk = pd.read_csv(crosswalk_path)

    outlets["HEI_clean"] = outlets["HEI"].apply(clean_name)
    crosswalk["HEI_clean"] = crosswalk["HEI"].apply(clean_name)

    merged = outlets.merge(
        crosswalk,
        on="HEI_clean",
        how="left",
        suffixes=("", "_ipeds")
    )

    if "HEI_ipeds" in merged.columns:
        merged = merged.drop(columns=["HEI_ipeds"])

    final_cols = [
        "Newspaper ID",
        "Newspaper Name",
        "HEI",
        "UNITID",
        "IPEDS_Name",
        "Locale Code",
        "locale_type",
        "town_rural",
        "geo_group",
        "match_method",
        "review_flag"
    ]

    final_cols = [col for col in final_cols if col in merged.columns]

    merged[final_cols].to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outlets")
    parser.add_argument("crosswalk")
    parser.add_argument(
        "-o",
        "--output",
        default="outlet_hei_geo_crosswalk.csv"
    )

    args = parser.parse_args()

    main(
        args.outlets,
        args.crosswalk,
        args.output
    )
