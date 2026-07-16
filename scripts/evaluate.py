import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from rouge_score import rouge_scorer
from bert_score import scorer as bert_scorer

MODELS = ["gemini", "gpt", "claude"]
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
NA_VALUES = ["N/A"]


def compute_rouge_all(scorer, reference, candidate):
    """
    Calculates every ROUGE metric (rouge1, rouge2, rougeL x precision/recall/fmeasure)
    for a single reference/candidate pair.

    Guard rules are identical to the original single-metric version:
      - both sides omitted/N/A               -> every metric forced to 1.0
      - only one side is a genuinely empty "" -> every metric forced to 0.0
      - otherwise                             -> real rouge_scorer output

    Note: a literal "N/A" on one side (with real text on the other) is NOT caught by
    the second guard (only an actual empty string "" is), so that case still falls
    through to real scoring against the literal text "N/A" -- this mirrors the
    original script's behavior exactly, it isn't a new bug.
    """
    if reference in NA_VALUES and candidate in NA_VALUES:
        return {rt: {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0} for rt in ROUGE_TYPES}
    if not reference or not candidate:
        return {rt: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for rt in ROUGE_TYPES}

    scores = scorer.score(reference, candidate)
    return {
        rt: {
            "precision": float(scores[rt].precision),
            "recall": float(scores[rt].recall),
            "fmeasure": float(scores[rt].fmeasure),
        }
        for rt in ROUGE_TYPES
    }


def compute_bertscore_all(bs_evaluator, references, candidates):
    """
    Calculates full BERTScore (precision, recall, f1) for an entire column at once.

    Same guard rules as the original script: a whole-column shortcut when every
    candidate is omitted/N/A, otherwise a real batch score followed by the same
    row-by-row N/A overwrite (both sides omitted -> 1.0, exactly one side
    omitted -> 0.0, applied with OR -- stricter than the ROUGE guard above,
    exactly like the original).

    Returns three numpy arrays: precision, recall, f1.
    """
    if all(c in NA_VALUES for c in candidates):
        fill = np.array([1.0 if r in NA_VALUES else 0.0 for r in references], dtype="float64")
        return fill.copy(), fill.copy(), fill.copy()

    safe_cands = [c if (c not in NA_VALUES and str(c).strip() != "") else "." for c in candidates]
    safe_refs = [r if (r not in NA_VALUES and str(r).strip() != "") else "." for r in references]

    p, r, f1 = bs_evaluator.score(safe_cands, safe_refs, batch_size=32)
    p_np, r_np, f1_np = p.cpu().numpy(), r.cpu().numpy(), f1.cpu().numpy()

    for i, (ref, cand) in enumerate(zip(references, candidates)):
        if ref in NA_VALUES and cand in NA_VALUES:
            p_np[i] = r_np[i] = f1_np[i] = 1.0
        elif ref in NA_VALUES or cand in NA_VALUES:
            p_np[i] = r_np[i] = f1_np[i] = 0.0

    return p_np, r_np, f1_np


def export_comprehensive_json(detailed_scores, skipped_sheets, file_path, output_path="evaluation_metadata.json"):
    """
    Generates an independent JSON artifact logging every calculation variable,
    hyperparameter matrix, evaluation framework component, AND the full per-row,
    per-model score breakdown (rouge1/rouge2/rougeL x precision/recall/fmeasure,
    bertscore x precision/recall/f1, plus the compared text itself).
    """
    metadata_payload = {
        "execution_context": {
            "timestamp": datetime.now().isoformat(),
            "target_language": "it",
            "target_filename": os.path.basename(file_path),
            "evaluated_models": MODELS,
            "sheets_processed": list(detailed_scores.keys()),
            "sheets_skipped": skipped_sheets,
        },
        "output_split_policy": {
            "excel_file_contains": [
                "{model}_rouge_answer / {model}_rouge_evidence -> rougeL fmeasure only",
                "{model}_bertscore_answer / {model}_bertscore_evidence -> bertscore f1 only"
            ],
            "json_file_contains": (
                "everything else: rouge1/rouge2/rougeL precision & recall (plus fmeasure), "
                "bertscore precision & recall (plus f1), the raw compared text per row/model, "
                "and every hyperparameter used to produce them"
            )
        },
        "rouge_framework_parameters": {
            "library": "google-rouge-score",
            "all_computed_metrics": ROUGE_TYPES,
            "metrics_per_type": ["precision", "recall", "fmeasure"],
            "sheet_target_metric": "rougeL.fmeasure (F1-score)",
            "internal_parameters": {
                "use_stemmer": True,
                "split_summaries": False
            }
        },
        "bertscore_framework_parameters": {
            "library": "bert-score (PyTorch backend)",
            "backbone_model_alignment": "dbmdz/bert-base-italian-xxl-cased",
            "total_layers": 12,
            "layer_selection_policy": "all_layers_averaged_default",
            "all_computed_metrics": ["precision", "recall", "f1"],
            "sheet_target_metric": "f1 (F1-score)",
            "execution_parameters": {
                "batch_size": 32,
                "rescale_with_baseline": False,
                "baseline_value": None,
                "device_allocation": "cuda" if torch.cuda.is_available() else "cpu",
                "use_idf": False
            }
        },
        "detailed_scores": detailed_scores
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Documented comprehensive hyperparameter + score registry -> {output_path}")


def evaluate_excel_workbook(file_path="All_Newspapers_answers_evidence.xlsx",
                             json_output_path="evaluation_metadata.json"):
    if not os.path.exists(file_path):
        print(f"[ERROR] Could not find Excel workbook at: {file_path}")
        return

    bert_model = "dbmdz/bert-base-italian-xxl-cased"

    print("Loading text tokenizers and deep-learning scoring backends...")
    r_scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    bs_evaluator = bert_scorer.BERTScorer(model_type=bert_model, num_layers=12, lang="it")

    # Open the Excel workbook and find all sheet names
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names

    processed_sheets = {}
    detailed_scores = {}
    skipped_sheets = []

    for sheet in sheet_names:
        print(f"\nReading Sheet: [{sheet}]")

        # keep_default_na=False avoids converting string "N/A" to empty float NaN values
        df = pd.read_excel(xl, sheet_name=sheet, keep_default_na=False, na_values=[])

        # Bypass non-data tabs like README
        if "ferrari_answer" not in df.columns or "ferrari_evidence" not in df.columns:
            print(f"  Skipping [{sheet}]: Ground-truth evaluation columns are missing.")
            processed_sheets[sheet] = df
            skipped_sheets.append(sheet)
            continue

        # Strip whitespace and force conversion to string objects
        df["ferrari_answer"] = df["ferrari_answer"].astype(str).str.strip()
        df["ferrari_evidence"] = df["ferrari_evidence"].astype(str).str.strip()

        # Seed one JSON record per row with ground truth; each model gets folded in below
        sheet_rows = [
            {
                "row_index": int(idx),
                "ferrari_answer": row["ferrari_answer"],
                "ferrari_evidence": row["ferrari_evidence"],
                "models": {}
            }
            for idx, row in df.iterrows()
        ]

        for model in MODELS:
            ans_col = f"{model}_answer"
            ev_col = f"{model}_evidence"

            # Destination cells inside the target spreadsheet tabs (F1 only)
            out_rouge_ans = f"{model}_rouge_answer"
            out_rouge_ev = f"{model}_rouge_evidence"
            out_bert_ans = f"{model}_bertscore_answer"
            out_bert_ev = f"{model}_bertscore_evidence"

            if ans_col not in df.columns or ev_col not in df.columns:
                continue

            df[ans_col] = df[ans_col].fillna("").astype(str).str.strip()
            df[ev_col] = df[ev_col].fillna("").astype(str).str.strip()
            print(f"  -> Processing scores for Model: {model}")

            # Ensure evaluation columns exist and are float dtype
            for col in [out_rouge_ans, out_rouge_ev, out_bert_ans, out_bert_ev]:
                if col not in df.columns:
                    df[col] = pd.Series(dtype="float64")
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # --- ROUGE: compute the full metric set per row; Excel keeps rougeL fmeasure only ---
            rouge_ans_full = []
            rouge_ev_full = []
            for idx, row in df.iterrows():
                r_ans = compute_rouge_all(r_scorer, row["ferrari_answer"], row[ans_col])
                r_ev = compute_rouge_all(r_scorer, row["ferrari_evidence"], row[ev_col])
                rouge_ans_full.append(r_ans)
                rouge_ev_full.append(r_ev)
                df.at[idx, out_rouge_ans] = r_ans["rougeL"]["fmeasure"]
                df.at[idx, out_rouge_ev] = r_ev["rougeL"]["fmeasure"]

            # --- BERTSCORE: compute full precision/recall/f1 per column; Excel keeps f1 only ---
            refs_ans = df["ferrari_answer"].tolist()
            cands_ans = df[ans_col].tolist()
            refs_ev = df["ferrari_evidence"].tolist()
            cands_ev = df[ev_col].tolist()

            p_ans, rec_ans, f1_ans = compute_bertscore_all(bs_evaluator, refs_ans, cands_ans)
            p_ev, rec_ev, f1_ev = compute_bertscore_all(bs_evaluator, refs_ev, cands_ev)

            df[out_bert_ans] = f1_ans
            df[out_bert_ev] = f1_ev

            # --- fold the full breakdown (everything Excel does NOT get) into the JSON record ---
            for idx in range(len(df)):
                sheet_rows[idx]["models"][model] = {
                    "answer": {
                        "text": cands_ans[idx],
                        "rouge1": rouge_ans_full[idx]["rouge1"],
                        "rouge2": rouge_ans_full[idx]["rouge2"],
                        "rougeL": rouge_ans_full[idx]["rougeL"],
                        "bertscore": {
                            "precision": float(p_ans[idx]),
                            "recall": float(rec_ans[idx]),
                            "f1": float(f1_ans[idx]),
                        }
                    },
                    "evidence": {
                        "text": cands_ev[idx],
                        "rouge1": rouge_ev_full[idx]["rouge1"],
                        "rouge2": rouge_ev_full[idx]["rouge2"],
                        "rougeL": rouge_ev_full[idx]["rougeL"],
                        "bertscore": {
                            "precision": float(p_ev[idx]),
                            "recall": float(rec_ev[idx]),
                            "f1": float(f1_ev[idx]),
                        }
                    }
                }

        processed_sheets[sheet] = df
        detailed_scores[sheet] = sheet_rows

    # Re-write all tabs straight back into the target .xlsx file layout (F1 columns only)
    print(f"\nWriting updates directly back to Excel tabs...")
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for sheet_name, df_sheet in processed_sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  [UPDATED TAB] -> {sheet_name}")

    # Generate the comprehensive JSON file containing every metric + all metadata
    export_comprehensive_json(
        detailed_scores=detailed_scores,
        skipped_sheets=skipped_sheets,
        file_path=file_path,
        output_path=json_output_path,
    )


if __name__ == "__main__":
    evaluate_excel_workbook()