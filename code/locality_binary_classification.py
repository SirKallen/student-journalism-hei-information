# Classifies articles for binary geographic locality using an LLM.

import argparse
import json
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
from openai import OpenAI


MODEL = "gpt-5.4-mini"

ARTICLE_ID_COL = "article_id"
HEADLINE_COL = "headline"
TEXT_PATH_COL = "text_path"

MAX_OUTPUT_TOKENS = 250
TEMPERATURE = 1
TOP_P = 1
SLEEP_SECONDS = 0.2
MIN_ARTICLE_BODY_WORDS = 50


LOCALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "article_id": {
            "type": "string"
        },
        "llm_locality": {
            "type": "integer",
            "enum": [0, 1]
        },
        "llm_locality_excerpt": {
            "type": "string"
        },
        "needs_review": {
            "type": "integer",
            "enum": [0, 1]
        },
        "final_locality": {
            "type": "null"
        }
    },
    "required": [
        "article_id",
        "llm_locality",
        "llm_locality_excerpt",
        "needs_review",
        "final_locality"
    ],
    "additionalProperties": False
}


def normalize_zip_path(path_value):
    path = str(path_value).strip()
    path = path.replace("\\", "/")
    return path.lstrip("./")


def read_article_from_zip(zip_file, text_path):
    target_path = normalize_zip_path(text_path)
    zip_names = zip_file.namelist()

    if target_path in zip_names:
        with zip_file.open(target_path) as file:
            return file.read().decode(
                "utf-8",
                errors="replace"
            )

    matches = [
        name
        for name in zip_names
        if normalize_zip_path(name).endswith(
            target_path
        )
    ]

    if len(matches) == 1:
        with zip_file.open(matches[0]) as file:
            return file.read().decode(
                "utf-8",
                errors="replace"
            )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches found for: {text_path}"
        )

    raise FileNotFoundError(
        f"No match found for: {text_path}"
    )


def split_article_metadata_and_body(raw_text):
    marker = "# ---"

    if marker in raw_text:
        metadata, body = raw_text.split(
            marker,
            1
        )

        return (
            metadata.strip(),
            body.strip()
        )

    return "", raw_text.strip()


def count_words(text):
    if not isinstance(text, str):
        return 0

    return len(text.split())


def fallback_result(
    article_id,
    text_path,
    article_word_count=0
):
    return {
        "article_id": str(article_id),
        "text_path": text_path,
        "llm_locality": 0,
        "llm_locality_excerpt":
            "No substantive locality evidence found",
        "needs_review": 1,
        "article_word_count":
            int(article_word_count),
        "final_locality": None
    }


def build_prompt(
    codebook,
    article_id,
    headline,
    text_path,
    article_metadata,
    article_body,
    article_word_count
):
    return f"""
You are performing a content-analysis classification task.

Classify one article for the locality binary indicator only.

Use the provided codebook exactly. Do not code any other variables or infer rules beyond the codebook.

Use only the headline and article body as substantive evidence.

Do not use the URL, source name, publication date, extraction date, author name, archive label, section label, tags, category labels, text_path, or other metadata as substantive evidence.

The llm_locality_excerpt must come from the article body. If no substantive locality evidence is found, write exactly:
No substantive locality evidence found

If the article body is very short, incomplete, video-only, transcript-free, or does not contain enough substantive text to confidently apply the codebook, set "needs_review": 1. Do not infer locality status from the headline alone when the article body is insufficient.

CODEBOOK:
{codebook}

ARTICLE:
article_id: {article_id}
headline: {headline}
text_path: {text_path}

ARTICLE METADATA:
{article_metadata}

ARTICLE BODY WORD COUNT:
{article_word_count}

ARTICLE BODY:
{article_body}

Return only valid JSON.

Return exactly this structure:

{{
  "article_id": "{article_id}",
  "llm_locality": 0,
  "llm_locality_excerpt": "No substantive locality evidence found",
  "needs_review": 0,
  "final_locality": null
}}

Rules:
- "article_id": copy the provided article_id exactly.
- "llm_locality": use 1 if the article satisfies the locality codebook; otherwise use 0.
- "llm_locality_excerpt": provide a short excerpt from the article body that supports the classification.
- "needs_review": use 1 for ambiguous, borderline, incomplete, internally inconsistent, low-confidence, video-only, transcript-free, or insufficient-text cases; otherwise use 0.
- "final_locality": always return null.
- The article must pass the codebook's inclusion gates to be coded as locality = 1.
- A local place mention alone is not enough.
- A local source alone is not enough.
- A local setting alone is not enough.
- A campus, town, city, neighborhood, local institution, school, school district, local organization, or local government can support locality = 1 only when substantively tied to the article's topic.
- Multiple qualifying local units may collectively support locality = 1 if the article's primary geographic frame remains local under the codebook.
- Do not require one local unit to dominate if multiple local units collectively form the local frame.
- County-level, regional, state-level, national, international, and non-geographic articles should be coded as locality = 0 unless the dominant geographic frame is local under the codebook.
- A broader state, national, international, cultural, policy, or trend topic may still be locality = 1 when the dominant frame is local under the codebook.
- If local and non-local frames are balanced or unclear, set "needs_review": 1.
"""


def classify_article(
    client,
    codebook,
    article_id,
    headline,
    text_path,
    article_metadata,
    article_body,
    article_word_count
):
    prompt = build_prompt(
        codebook,
        article_id,
        headline,
        text_path,
        article_metadata,
        article_body,
        article_word_count
    )

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        text={
            "format": {
                "type": "json_schema",
                "name": "locality_binary_result",
                "strict": True,
                "schema": LOCALITY_SCHEMA
            }
        }
    )

    return json.loads(
        response.output_text
    )


def run_classification(
    codebook_path,
    input_csv,
    article_zip,
    output_csv
):
    api_key = os.environ.get(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set."
        )

    client = OpenAI(
        api_key=api_key
    )

    codebook = Path(
        codebook_path
    ).read_text(
        encoding="utf-8"
    )

    df = pd.read_csv(
        input_csv
    )

    required_cols = [
        ARTICLE_ID_COL,
        HEADLINE_COL,
        TEXT_PATH_COL
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    results = []

    with zipfile.ZipFile(
        article_zip,
        "r"
    ) as zip_file:

        for _, row in df.iterrows():
            article_id = str(
                row[ARTICLE_ID_COL]
            )

            headline = str(
                row[HEADLINE_COL]
            )

            text_path = str(
                row[TEXT_PATH_COL]
            )

            article_word_count = 0

            try:
                raw_text = (
                    read_article_from_zip(
                        zip_file,
                        text_path
                    )
                )

                (
                    article_metadata,
                    article_body
                ) = split_article_metadata_and_body(
                    raw_text
                )

                article_word_count = count_words(
                    article_body
                )

                result = classify_article(
                    client=client,
                    codebook=codebook,
                    article_id=article_id,
                    headline=headline,
                    text_path=text_path,
                    article_metadata=article_metadata,
                    article_body=article_body,
                    article_word_count=article_word_count
                )

                result["article_id"] = str(
                    result["article_id"]
                )

                result["text_path"] = (
                    text_path
                )

                result[
                    "article_word_count"
                ] = int(
                    article_word_count
                )

                if (
                    article_word_count
                    < MIN_ARTICLE_BODY_WORDS
                ):
                    result[
                        "needs_review"
                    ] = 1

            except Exception:
                result = fallback_result(
                    article_id,
                    text_path,
                    article_word_count
                )

            results.append(
                result
            )

            time.sleep(
                SLEEP_SECONDS
            )

    output_df = pd.DataFrame(
        results
    )

    output_df = output_df[
        [
            "article_id",
            "text_path",
            "llm_locality",
            "llm_locality_excerpt",
            "needs_review",
            "article_word_count",
            "final_locality"
        ]
    ]

    output_df.to_csv(
        output_csv,
        index=False
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "codebook",
        help="Path to the locality codebook."
    )

    parser.add_argument(
        "metadata",
        help="Article metadata CSV."
    )

    parser.add_argument(
        "articles",
        help="ZIP archive containing article text files."
    )

    parser.add_argument(
        "-o",
        "--output",
        default="locality_binary_results.csv",
        help="Output CSV."
    )

    args = parser.parse_args()

    run_classification(
        args.codebook,
        args.metadata,
        args.articles,
        args.output
    )


if __name__ == "__main__":
    main()
