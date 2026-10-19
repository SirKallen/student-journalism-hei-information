# LLM Classification Prompts

This file documents the prompts used for the two LLM-assisted article classification tasks in the study:

1. Critical Information Needs (CIN) coverage
2. Geographic locality

Each classification task uses a separate codebook. The codebooks are stored in:

- `cin_codebook.md`
- `locality_codebook.md`

The prompts instruct the model to classify only the variable associated with the relevant codebook and to return a structured JSON response.

---

# Critical Information Needs (CIN) Classification

## Purpose

The CIN classification identifies whether an article substantively addresses a recognized Critical Information Need under the project's CIN codebook.

The output variable is:

`llm_CIN_binary`

where:

- `1` = article satisfies the CIN criteria
- `0` = article does not satisfy the CIN criteria

Cases that are ambiguous, incomplete, or otherwise difficult to classify are separately identified using `needs_review`.

## Classification Prompt

```text
You are assisting with a content-analysis classification task for a research project on HEI-affiliated student-produced news outlets and their role in public information ecosystems.

Your task is to classify one article for the first CIN binary indicator only.

Use the provided codebook exactly. Do not code any other variables. Do not infer rules beyond the codebook.

Classify the article using only the headline and article body as substantive evidence.

Do not treat URL, source name, publication date, extraction date, author name, archive label, section label, tags, category labels, text_path, or other metadata fields as substantive evidence for CIN classification.

The llm_CIN_excerpt must come from the article body, not from metadata. If no substantive CIN evidence is found in the article body, write exactly:

No substantive CIN evidence found

The text_path is provided only for file tracking and must not be used as substantive article evidence.

If the article body is very short, incomplete, video-only, transcript-free, or does not contain enough substantive text to confidently apply the codebook, set "needs_review": 1. Do not infer CIN status from the headline alone when the body text is insufficient.

You will be given:

- a codebook;
- an article ID;
- an article headline;
- article metadata;
- article body word count;
- the article body.

CODEBOOK:
{codebook}

ARTICLE METADATA FROM CSV:
article_id: {article_id}
headline: {headline}
text_path: {text_path}

ARTICLE METADATA FROM TEXT FILE:
{article_metadata}

ARTICLE BODY WORD COUNT:
{article_word_count}

ARTICLE BODY:
{article_body}

Return only valid JSON. Do not include explanation, markdown, comments, or text outside the JSON.

Return exactly this JSON structure:

{
  "article_id": "{article_id}",
  "llm_CIN_binary": 0,
  "llm_CIN_excerpt": "No substantive CIN evidence found",
  "needs_review": 0,
  "final_CIN_binary": null
}

Allowed values:

- "article_id": copy the provided article_id exactly.
- "llm_CIN_binary": use 1 if the article addresses a Critical Information Need under the codebook; use 0 if it does not.
- "llm_CIN_excerpt": provide a short excerpt from the article body that best supports the classification. If no substantive CIN evidence is found, write "No substantive CIN evidence found".
- "needs_review": use 1 if the case is ambiguous, borderline, incomplete, internally inconsistent, lacks enough article body text, appears video-only or transcript-free, or is low-confidence; otherwise use 0.
- "final_CIN_binary": always return null.

Decision reminders:

- A CIN topic alone is not enough.
- Locality alone is not enough.
- Campus relevance alone is not enough.
- Opinion, arts/culture, sports, event, profile, lifestyle, or business articles may count only when they substantively develop a recognized CIN topic.
- If the article plausibly relates to CIN but does not clearly satisfy the codebook, set "needs_review": 1.
```

## CIN Structured Output

The classification returns the following JSON structure:

```json
{
  "article_id": "example_id",
  "llm_CIN_binary": 0,
  "llm_CIN_excerpt": "No substantive CIN evidence found",
  "needs_review": 0,
  "final_CIN_binary": null
}
```

### Output Fields

| Field | Description |
|---|---|
| `article_id` | Article identifier copied from the input data |
| `llm_CIN_binary` | LLM-generated binary CIN classification |
| `llm_CIN_excerpt` | Article-body excerpt supporting the classification |
| `needs_review` | Indicator identifying cases requiring additional review |
| `final_CIN_binary` | Reserved for the final reviewed classification |

---

# Locality Classification

## Purpose

The locality classification identifies whether the article's dominant geographic frame is local under the project's locality codebook.

The output variable is:

`llm_locality`

where:

- `1` = article satisfies the locality criteria
- `0` = article does not satisfy the locality criteria

Cases that are ambiguous, incomplete, or contain balanced local and non-local frames are separately identified using `needs_review`.

## Classification Prompt

```text
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

{
  "article_id": "{article_id}",
  "llm_locality": 0,
  "llm_locality_excerpt": "No substantive locality evidence found",
  "needs_review": 0,
  "final_locality": null
}

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
```

## Locality Structured Output

The classification returns the following JSON structure:

```json
{
  "article_id": "example_id",
  "llm_locality": 0,
  "llm_locality_excerpt": "No substantive locality evidence found",
  "needs_review": 0,
  "final_locality": null
}
```

### Output Fields

| Field | Description |
|---|---|
| `article_id` | Article identifier copied from the input data |
| `llm_locality` | LLM-generated binary locality classification |
| `llm_locality_excerpt` | Article-body excerpt supporting the classification |
| `needs_review` | Indicator identifying cases requiring additional review |
| `final_locality` | Reserved for the final reviewed classification |

---

# Implementation

The prompts are implemented in the corresponding Python scripts:

```text
code/
├── cin_binary_classification.py
└── locality_binary_classification.py
```

The associated codebooks are:

```text
prompts/
├── cin_codebook.md
└── locality_codebook.md
```

The full prompt and codebook workflow is therefore:

```text
cin_codebook.md
        ↓
cin_binary_classification.py
        ↓
llm_CIN_binary
        ↓
manual review where necessary
        ↓
final_CIN_binary


locality_codebook.md
        ↓
locality_binary_classification.py
        ↓
llm_locality
        ↓
manual review where necessary
        ↓
final_locality
```

# Review Flags

For both classification tasks, `needs_review = 1` identifies articles for which the automated classification may require additional human review.

Examples include:

- ambiguous or borderline cases;
- incomplete article text;
- unusually short extracted article bodies;
- video-only or transcript-free content;
- internally inconsistent evidence;
- balanced local and non-local geographic frames;
- cases where the codebook cannot be applied confidently.

The review flag is separate from the substantive binary classification and is intended to support the validation and review process.
