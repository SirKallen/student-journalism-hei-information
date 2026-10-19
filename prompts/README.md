# Prompts and Codebooks

This folder contains the codebooks and LLM classification prompts used for the article-level content analysis in the study:

**Student Journalism and the Informational Role of Higher Education Institutions: A Rural–Urban Comparative Analysis**

The materials in this folder document the rules used to classify articles for two binary measures:

- Critical Information Needs (CIN) coverage
- Geographic locality

These files are intended to make the LLM-assisted coding workflow transparent and reproducible.

## Files

### `cin_codebook.md`

Defines the criteria used to classify whether an article substantively addresses a Critical Information Need.

The codebook includes:

- recognized CIN topic categories;
- a CIN Topic Gate;
- a Local/Public Information Gate;
- a substantiveness standard;
- rules for briefs, updates, and follow-up items;
- guidance for sports, arts, events, profiles, opinion pieces, policy stories, institutional initiatives, service disruptions, research stories, and workforce-training features;
- exclusion criteria; and
- positive and negative examples.

The final binary coding framework is:

```text
CIN_binary = 1
```

when an article satisfies the inclusion criteria, and:

```text
CIN_binary = 0
```

when it does not.

Ambiguous or borderline cases may be flagged for review.

---

### `locality_codebook.md`

Defines the criteria used to classify whether an article's dominant geographic frame is local.

The codebook uses three inclusion gates:

1. Local Unit Gate
2. Local Topic Orientation Gate
3. Primary Focus Gate

Qualifying local units may include:

- towns;
- cities;
- municipalities;
- villages;
- neighborhoods;
- city districts;
- specific campus communities;
- specific local institutions;
- consolidated city-counties;
- island municipalities; and
- specific K–12 schools, school districts, local education agencies, or local public education systems.

The codebook also includes detailed rules for:

- campus communities;
- local institutions;
- local sources and expert commentary;
- profiles;
- opinion articles;
- arts, culture, entertainment, lifestyle, and business coverage;
- sports coverage;
- student government;
- internal campus news;
- multi-campus systems;
- research stories;
- locally grounded trend stories;
- broader issues with local effects;
- campus events;
- county and regional coverage;
- virtual events;
- metadata and datelines;
- consolidated jurisdictions;
- lectures and academic talks;
- first-person articles; and
- non-local geographic scales.

The final binary coding framework is:

```text
locality = 1
```

when an article satisfies all three locality gates, and:

```text
locality = 0
```

when it does not.

Cases with balanced or unclear local and non-local frames may be flagged for review.

---

### `classification_prompts.md`

Documents the full LLM prompts used to apply the CIN and locality codebooks.

The file includes:

- the CIN classification prompt;
- the CIN structured JSON output;
- the locality classification prompt;
- the locality structured JSON output;
- output-field definitions;
- decision reminders;
- review-flag guidance; and
- the relationship between the prompts, codebooks, and Python classification scripts.

The prompts are implemented in:

```text
code/
├── cin_binary_classification.py
└── locality_binary_classification.py
```

## Classification Workflow

The CIN workflow is:

```text
cin_codebook.md
        ↓
classification_prompts.md
        ↓
cin_binary_classification.py
        ↓
llm_CIN_binary
        ↓
manual review where necessary
        ↓
final_CIN_binary
```

The locality workflow is:

```text
locality_codebook.md
        ↓
classification_prompts.md
        ↓
locality_binary_classification.py
        ↓
llm_locality
        ↓
manual review where necessary
        ↓
final_locality
```

## Review Flags

Both classification procedures include a `needs_review` field.

A case may be flagged for review when:

- the article is ambiguous or borderline;
- the extracted text is incomplete;
- the article body is unusually short;
- the article is video-only or transcript-free;
- the evidence is internally inconsistent;
- the model cannot confidently apply the codebook; or
- the locality classification contains balanced local and non-local frames.

The review flag is separate from the binary classification itself.

## Structured Outputs

The CIN classification returns:

```json
{
  "article_id": "example_id",
  "llm_CIN_binary": 0,
  "llm_CIN_excerpt": "No substantive CIN evidence found",
  "needs_review": 0,
  "final_CIN_binary": null
}
```

The locality classification returns:

```json
{
  "article_id": "example_id",
  "llm_locality": 0,
  "llm_locality_excerpt": "No substantive locality evidence found",
  "needs_review": 0,
  "final_locality": null
}
```

The `llm_*` fields contain the model-generated classification, while the `final_*` fields are reserved for the final reviewed coding result.

## Reproducibility

These materials document the operational definitions and prompt structures used in the LLM-assisted content analysis.

Because the underlying article-level dataset is not included in this repository, the classification workflow cannot be reproduced from these files alone.

However, the codebooks and prompts are provided to make the coding framework, decision rules, structured outputs, and review procedure transparent.
