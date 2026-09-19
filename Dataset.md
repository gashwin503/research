# Dataset ID & Versioning Spec — Matched-Condition Problem Sets

**Project:** Separating Mathematical Reasoning from Financial Knowledge in LLMs (Team KHAT)
**Purpose:** Guarantee that the four (or more) condition variants of a single underlying calculation are traceably linked, individually addressable, and immutably referenced across every experiment — overlap analysis, pruning tests, paraphrase checks, and any future re-run.

---

## 1. Design goals

1. A single **Problem Family ID (PFID)** ties together every variant (conditions × paraphrases × versions) of one underlying calculation, so "give me the whole quad for problem X" is a lookup, not a search.
2. Every individual problem instance has a **Full ID** that is globally unique, human-readable, and parseable by a fixed regex — no two different problem texts ever share a Full ID.
3. Once a Full ID + version has been *used in a reported experiment*, its content is **frozen**. Any edit (wording, numbers, token count) creates a new version; nothing is silently overwritten.
4. Token counts are **tokenizer-specific**, so they live in a separate linked table, not baked into the ID.
5. Every experiment run cites an explicit, timestamped **manifest** of exact (Full ID, version) tuples — so a result can always be traced back to the precise problem text used.

---

## 2. ID grammar

```
PFID       = "KHAT" "-" OP_CODE "-" SRC_CODE "-" SEQ
FULL_ID    = PFID "_" COND_CODE [ "-p" PARA_IDX ] "." "v" VER
```

| Field | Format | Meaning |
|---|---|---|
| `KHAT` | literal | team/project namespace — prevents collision if this dataset is ever merged with another team's |
| `OP_CODE` | 2–8 upper-case letters | the mathematical operation/formula family (see table below) |
| `SRC_CODE` | 2–5 upper-case letters | provenance of the underlying numbers/scenario (see table below) |
| `SEQ` | zero-padded 4-digit int | sequence number, unique within `OP_CODE-SRC_CODE` |
| `COND_CODE` | one of `MATH`, `FIN`, `NONFIN`, `CTRL` | which of the four conditions this instance is |
| `PARA_IDX` | 1–2 digit int, optional | paraphrase index within a condition; omitted = the canonical/primary wording (equivalent to `p0`) |
| `VER` | 1+ digit int, starts at 1 | version of this specific (condition, paraphrase) instance |

**Regex:** `^KHAT-[A-Z]{2,8}-[A-Z]{2,5}-\d{4}_(MATH|FIN|NONFIN|CTRL)(-p\d{1,2})?\.v\d+$`

### 2.1 Operation codes (extend as needed, don't repurpose existing codes)

| Code | Meaning |
|---|---|
| `CAGR` | compound annual growth rate |
| `PCTMARGIN` | percentage / profit margin |
| `PV` | present value |
| `RATIO` | financial ratio computation |
| `GENOP` | placeholder for any operation not yet assigned a code — must be replaced before the item leaves draft status |

### 2.2 Source codes

| Code | Meaning |
|---|---|
| `FM` | adapted from FinanceMath |
| `GSM` | adapted from GSM8K |
| `MATHDS` | adapted from the MATH dataset |
| `CUST` | authored in-house by the team (matched non-finance / control wordings will mostly be `CUST`) |

### 2.3 Condition codes

| Code | Meaning |
|---|---|
| `MATH` | pure/symbolic math version |
| `FIN` | financial word-problem version |
| `NONFIN` | non-financial word-problem version |
| `CTRL` | nonsense/control version |

---

## 3. Worked example — one full quad

A CAGR problem adapted from FinanceMath, sequence 7, canonical wording, first version, all four conditions:

| Condition | Full ID |
|---|---|
| Pure math | `KHAT-CAGR-FM-0007_MATH.v1` |
| Financial | `KHAT-CAGR-FM-0007_FIN.v1` |
| Non-financial | `KHAT-CAGR-FM-0007_NONFIN.v1` |
| Control | `KHAT-CAGR-FM-0007_CTRL.v1` |

Shared **PFID**: `KHAT-CAGR-FM-0007` — this is what every overlap/pruning analysis groups by.

If the `FIN` wording is later edited to fix an accidental token-count mismatch, it becomes `KHAT-CAGR-FM-0007_FIN.v2`. The other three conditions are untouched and stay at `.v1` unless they're independently revised. If a second paraphrase of the financial version is added for the robustness check, it's `KHAT-CAGR-FM-0007_FIN-p1.v1` (the original wording is retroactively understood as `p0`, though it's written without the suffix).

---

## 4. Registry schema (single source of truth)

One row per Full ID. Recommend a CSV or JSON Lines file (`registry.jsonl`) checked into version control alongside the codebase — not a spreadsheet that gets hand-edited without a diff trail.

| Field | Type | Notes |
|---|---|---|
| `full_id` | string | primary key, matches the regex above |
| `pfid` | string | derived/redundant but stored explicitly for fast grouping queries |
| `condition` | enum | `MATH` / `FIN` / `NONFIN` / `CTRL` |
| `paraphrase_idx` | int | `0` for canonical wording |
| `version` | int | |
| `status` | enum | `draft`, `frozen`, `deprecated` (see §5) |
| `operation_code` | string | e.g. `CAGR` |
| `source_code` | string | e.g. `FM` |
| `source_ref` | string | original problem ID/page/index in FinanceMath, GSM8K, etc.; `null` for `CUST` |
| `problem_text` | string | exact prompt text, verbatim |
| `numeric_values` | JSON object | all numbers appearing, keyed by role (e.g. `{"principal": 50000, "final_value": 67000, "years": 3}`) — enables automated checking that MATH/FIN/NONFIN/CTRL share identical values |
| `numeric_format` | string | e.g. `plain_no_comma`, `comma_grouped` — must match across a quad per the proposal's standardization rule |
| `correct_answer` | string/number | |
| `created_at` | ISO date | |
| `last_modified_at` | ISO date | |
| `change_note` | string | required whenever version > 1; one line describing what changed and why |
| `superseded_by` | string, nullable | Full ID of the version that replaced this one, if `status = deprecated` |

**Derived/linked table — `token_counts.jsonl`** (kept separate because one problem is tokenized differently per model):

| Field | Type | Notes |
|---|---|---|
| `full_id` | string | foreign key into registry |
| `tokenizer_name` | string | e.g. `Llama-3.1-8B`, `Qwen2.5-7B` |
| `token_count` | int | |
| `measured_at` | ISO date | |

This lets the same problem carry different token counts under different tokenizers without polluting the main registry or the ID itself.

---

## 5. Versioning & freeze policy

- **`draft`** — still being written/reviewed. Can be freely edited in place, no version bump required.
- **`frozen`** — has been used in at least one experiment manifest (§6). From this point on, the row is **immutable**. Any change requires creating a new row with `version + 1` and setting the old row's `status = deprecated`, `superseded_by = <new full_id>`.
- **`deprecated`** — retained forever for reproducibility of past results; never deleted, never reused.
- A version bump is required for **any** change to `problem_text`, `numeric_values`, or `numeric_format` — even a single-character wording fix. Token-count-only changes (e.g., re-measuring under a new tokenizer) do **not** require a version bump, since they're additive rows in `token_counts.jsonl`, not edits to `problem_text`.
- `change_note` is mandatory on every version ≥ 2. No silent diffs.

---

## 6. Experiment manifests (how results cite the registry)

Every experiment run produces its own manifest file, e.g. `manifests/2026-09-20_pruning-run-cagr.json`, containing:

```json
{
  "manifest_id": "2026-09-20_pruning-run-cagr",
  "model": "Llama-3.1-8B",
  "tokenizer": "Llama-3.1-8B",
  "registry_commit": "<git commit hash of registry.jsonl at run time>",
  "problem_ids": [
    "KHAT-CAGR-FM-0007_MATH.v1",
    "KHAT-CAGR-FM-0007_FIN.v1",
    "KHAT-CAGR-FM-0007_NONFIN.v1",
    "KHAT-CAGR-FM-0007_CTRL.v1"
  ],
  "run_date": "2026-09-20",
  "notes": "initial pruning pilot, 5% threshold"
}
```

Reporting a result should always cite the `manifest_id`, never just "the CAGR quad" — since a PFID can span multiple versions over the project's life, the manifest is what pins a result to exact frozen text. This also makes any future "why did overlap change between runs" question answerable by diffing two manifests' registry commits.

---

## 7. Quad-integrity check (run before any analysis)

Before a PFID is eligible for use in overlap or pruning analysis, validate:

1. Exactly one `frozen` row exists for each of `MATH`, `FIN`, `NONFIN`, `CTRL` at `paraphrase_idx = 0` (additional paraphrases are optional, not required).
2. `numeric_values` are identical across all four conditions for that PFID.
3. `numeric_format` is identical across all four conditions.
4. Token counts (per the tokenizer being used in that run) fall within the project's agreed tolerance band across the four conditions — flag, don't silently drop, if a quad fails this.

This check should be a script (`validate_quad.py` or similar) run against `registry.jsonl` + `token_counts.jsonl`, not a manual spot-check — with 4 conditions × paraphrases × multiple operations, manual tracking will drift fast.

---

## 8. Suggested file layout

```
/data
  registry.jsonl              # canonical problem registry (source of truth)
  token_counts.jsonl          # tokenizer-specific token counts, keyed by full_id
  /manifests
    2026-09-20_pruning-run-cagr.json
    2026-09-27_overlap-run-pctmargin.json
  /scripts
    validate_quad.py          # runs the §7 checks
    new_version.py            # enforces the freeze-then-bump workflow in §5
```

Keeping `registry.jsonl` and `manifests/` under git gives you the `registry_commit` field for free and makes "what did this problem look like when we ran experiment X" a one-line `git show` away.
