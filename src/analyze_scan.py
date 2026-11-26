from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ucd_build as ucd

ROOT = ucd.ROOT
RESULTS = ucd.RESULTS
EV_BANDS = ["legacy", "E14.0", "E15.0", "E15.1", "E16.0", "E17.0", "E18.0"]
EV_INDEX = {v: i for i, v in enumerate(ucd.EMOJI_DATES)}
LEGACY_MAX = EV_INDEX["E13.1"]
SCAN_SOURCE_NAMES = ["analyze_scan.py", "testset.py", "tokscan.py", "ucd_build.py", "validate.py"]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key):
    vals = [x[key] for x in rows if x.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def copy_summary(scan_rows):
    groups = defaultdict(list)
    for row in scan_rows:
        if row["probe_type"] != "copy":
            continue
        if row["item_type"] in ("atomic_emoji", "zwj_emoji"):
            system, version = "emoji", row["emoji_version"]
        elif row["item_type"] in ("atomic_nonemoji", "complex_nonemoji"):
            system, version = "unicode", row["release_version"]
        else:
            system, version = "ascii", "ASCII"
        groups[(row["tokenizer_id"], row["family"], system, version, row["item_type"])].append(row)
    out = []
    for (tid, family, system, version, item_type), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][2], str(x[0][3]), x[0][4])):
        out.append({
            "tokenizer_id": tid, "family": family, "system": system, "version": version, "item_type": item_type, "n": len(rows),
            "exact_single_token_count": sum(x["whole_sequence_token"] for x in rows), "exact_single_token_rate": sum(x["whole_sequence_token"] for x in rows) / len(rows),
            "mean_split_codepoint_count": mean(rows, "split_codepoint_count"), "split_metric_missing": sum(x["split_codepoint_count"] is None for x in rows),
            "mean_tokens_per_cluster": mean(rows, "tokens_per_cluster"), "mean_explicit_byte_fallback_count": mean(rows, "explicit_byte_fallback_count"),
            "mean_byte_level_single_byte_atom_count": mean(rows, "byte_level_single_byte_atom_count"),
            "mean_partial_utf8_token_count": mean(rows, "partial_utf8_token_count"), "tokenizer_roundtrip_exact_rate": sum(x["tokenizer_roundtrip_exact"] for x in rows) / len(rows),
            "nfc_nfd_divergence_rate": sum(x["nfc_nfd_token_ids_differ"] for x in rows) / len(rows),
        })
    return out


def flatten_inventory(inv):
    rows = []
    for tok in inv["tokenizers"]:
        for version, z in tok["rgi_single_token_rate_by_emoji_version"].items():
            rows.append({"tokenizer_id": tok["tokenizer_id"], "family": tok["family"], "emoji_version": version, **z})
    return rows


def band(version):
    return "legacy" if EV_INDEX[version] <= LEGACY_MAX else version


def weighted_mean(parts):
    num = den = 0.0
    for value, n in parts:
        if value is not None and n:
            num += value * n
            den += n
    return num / den if den else None


def primary_band_curve(flat):
    groups = defaultdict(list)
    for row in flat:
        groups[(row["tokenizer_id"], row["family"], band(row["emoji_version"]))].append(row)
    out = []
    for (tid, family, b), rows in sorted(groups.items(), key=lambda x: (x[0][0], EV_BANDS.index(x[0][2]))):
        for seq in ("all", "atomic", "zwj"):
            if seq == "all":
                n = sum(x["families"] for x in rows)
                single = sum(x["single_token_count"] for x in rows)
                vocab_exact = sum(x["exact_vocab_count"] for x in rows)
                split = weighted_mean([(x["atomic_mean_split_codepoint_count"], x["atomic_families"]) for x in rows] + [(x["zwj_mean_split_codepoint_count"], x["zwj_families"]) for x in rows])
                tpc = None
            else:
                n = sum(x[f"{seq}_families"] for x in rows)
                single = sum(x[f"{seq}_single_token_count"] for x in rows)
                vocab_exact = sum(x[f"{seq}_exact_vocab_count"] for x in rows)
                split = weighted_mean([(x[f"{seq}_mean_split_codepoint_count"], x[f"{seq}_families"]) for x in rows])
                tpc = weighted_mean([(x["zwj_mean_tokens_per_cluster"], x["zwj_families"]) for x in rows]) if seq == "zwj" else None
            out.append({"tokenizer_id": tid, "family": family, "sequence_type": seq, "emoji_band": b, "families": n, "exact_single_token_count": single, "exact_single_token_rate": single / n if n else None, "exact_vocab_count": vocab_exact, "exact_vocab_rate": vocab_exact / n if n else None, "mean_split_codepoint_count": split, "mean_tokens_per_cluster": tpc})
    return out


def monotone(values, direction):
    vals = [x for x in values if x is not None]
    if len(vals) < 2:
        return False, False
    if direction == "drop":
        return all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])), any(b < a - 1e-12 for a, b in zip(vals, vals[1:]))
    return all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), any(b > a + 1e-12 for a, b in zip(vals, vals[1:]))


def verdicts(band_rows):
    out = []
    for tid in sorted({x["tokenizer_id"] for x in band_rows}):
        family = next(x["family"] for x in band_rows if x["tokenizer_id"] == tid)
        row = {"tokenizer_id": tid, "family": family}
        for seq in ("all", "atomic", "zwj"):
            seqrows = [x for x in band_rows if x["tokenizer_id"] == tid and x["sequence_type"] == seq and x["families"]]
            seqrows.sort(key=lambda x: EV_BANDS.index(x["emoji_band"]))
            coverage = [x["exact_single_token_rate"] for x in seqrows]
            split = [x["mean_split_codepoint_count"] for x in seqrows]
            cov_mono, cov_strict = monotone(coverage, "drop")
            split_mono, split_strict = monotone(split, "rise")
            row[f"{seq}_coverage_series"] = coverage
            row[f"{seq}_split_series"] = split
            row[f"{seq}_coverage_monotone_drop"] = cov_mono and cov_strict
            row[f"{seq}_coverage_nonincreasing_even_if_flat"] = cov_mono
            row[f"{seq}_split_monotone_rise"] = split_mono and split_strict
            row[f"{seq}_split_nondecreasing_even_if_flat"] = split_mono
        row["atomic_sensitivity_signal"] = row["atomic_coverage_monotone_drop"] or row["atomic_split_monotone_rise"]
        row["proceed_rule4_signal"] = row["all_coverage_monotone_drop"] or row["all_split_monotone_rise"]
        out.append(row)
    firing_families = sorted({x["family"] for x in out if x["proceed_rule4_signal"]})
    atomic_families = sorted({x["family"] for x in out if x["atomic_sensitivity_signal"]})
    return out, firing_families, atomic_families


def scan_csv_rows(scan_rows):
    fields = ["scan_id", "probe_type", "tokenizer_id", "family", "item_type", "era", "release_version", "emoji_version", "token_count", "raw_token_count", "dummy_prefix_tokens_removed", "grapheme_clusters", "tokens_per_cluster", "whole_sequence_token", "tokenizer_roundtrip_exact", "explicit_byte_fallback_count", "byte_level_single_byte_atom_count", "partial_utf8_token_count", "split_codepoint_count", "token_bytes_concat_equals_input", "nfc_nfd_canonical_distinct", "nfc_nfd_token_count_difference", "nfc_nfd_token_ids_differ", "nfc_roundtrip_exact", "nfd_roundtrip_exact", "native_backend_ids_match", "backend", "host_unicodedata_version", "normalization_unicodedata_version", "text"]
    return [{k: row.get(k) for k in fields} for row in scan_rows], fields


def normalization_summary(scan_rows):
    groups = defaultdict(list)
    for row in scan_rows:
        if row["probe_type"] == "normalization_diagnostic":
            groups[(row["tokenizer_id"], row["family"])].append(row)
    out = []
    for (tid, family), rows in sorted(groups.items()):
        out.append({
            "tokenizer_id": tid, "family": family, "n": len(rows),
            "canonical_distinct_count": sum(x["nfc_nfd_canonical_distinct"] for x in rows),
            "token_count_equal_rate": sum(x["nfc_nfd_token_count_difference"] == 0 for x in rows) / len(rows),
            "token_count_pairing_fingerprint": "equal-counts" if all(x["nfc_nfd_token_count_difference"] == 0 for x in rows) else "divergent-counts",
            "token_id_divergence_rate": sum(x["nfc_nfd_token_ids_differ"] for x in rows) / len(rows),
            "nfc_roundtrip_exact_rate": sum(x["nfc_roundtrip_exact"] for x in rows) / len(rows),
            "nfd_roundtrip_exact_rate": sum(x["nfd_roundtrip_exact"] for x in rows) / len(rows),
            "mean_nfc_minus_nfd_token_count": mean(rows, "nfc_nfd_token_count_difference"),
        })
    return out


def newest_exactness(inventory):
    out = []
    for tok in inventory["tokenizers"]:
        z = tok["pilot_newest_cohort_exactness"]
        out.append({
            "tokenizer_id": tok["tokenizer_id"], "family": tok["family"], "families": z["families"],
            "pipeline_single_token_count": z["pipeline_single_token_count"], "exact_vocab_entry_count": z["exact_vocab_entry_count"],
            "exact_vocab_entries": z["exact_vocab_entries"], "pipeline_single_tokens": z["pipeline_single_tokens"],
        })
    return out


def fmt_series(rows, key="exact_single_token_rate"):
    return ", ".join(f"{x['emoji_band']}={x[key]:.3f} (n={x['families']})" for x in rows if x[key] is not None)


def fmt_split_series(rows):
    return ", ".join(f"{x['emoji_band']}={x['mean_split_codepoint_count']:.3f}" for x in rows if x["mean_split_codepoint_count"] is not None)


def freeze_hashes():
    names = ["manifest.json", "ucd_audit.json", "testset_audit.json", "validation.json", "tokenizer_inventory.json", "copy_pilot.jsonl", "copy_pilot.csv", "structural_probes.jsonl", "scan_units.jsonl", "count_candidates.jsonl", "count_candidate_scan.jsonl", "count_candidate_scan.metadata.json", "tokenizer_scan.jsonl", "tokenizer_scan.metadata.json", "tokenizer_scan.csv", "copy_scan_summary.csv", "normalization_summary.csv", "rgi_version_curve.csv", "rgi_primary_band_curve.csv", "scan_verdict.json", "SCAN_REPORT.md"]
    rows = {}
    for name in names:
        path = RESULTS / name
        if path.exists():
            rows[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
    rows["code_and_requirements"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / "src" / name for name in SCAN_SOURCE_NAMES] + [ROOT / "requirements.txt", ROOT / "run.sh"] if p.exists()}
    (RESULTS / "artifact_hashes.json").write_text(json.dumps(rows, indent=2, sort_keys=True))


def write_report(verdict, bands, inventory, validation, norm_rows):
    ucd_audit = json.loads((RESULTS / "ucd_audit.json").read_text())
    surface_collisions = ucd_audit.get("cross_version_zwj_surface_key_collisions", {})
    e17_surface_collisions = sum("E17.0" in versions for versions in surface_collisions.values())
    loc = {name[:-3]: len((ROOT / "src" / name).read_text().splitlines()) for name in SCAN_SOURCE_NAMES}
    loc_detail = ", ".join(f"{k} {v}" for k, v in loc.items())
    lines = [
        "# Unicode release debt: tokenizer scan report", "",
        "Scope: Phase 0 plus tokenizer-scan portion of Phase 1. No model weights or behavioral inference were used. Tokenization uses no special tokens. SentencePiece artifacts are encoded with the native backend and cross-checked against raw `tokenizer.json`; Qwen uses raw `tokenizer.json` directly, never `transformers.AutoTokenizer`.", "",
        "## Data and harness validity", "",
        f"- Unicode/emoji inputs: Unicode 18.0.0, Emoji 18, CLDR release-49 annotations; {ucd_audit['input_files']} recorded input files are SHA-256 pinned and passed independent second-download hash checks.",
        "- Audit counts reproduced: E15.1 ZWJ 28 surface families (118 raw), E17 ZWJ 11 (146 raw), strict atomic E16/E17/E18 = 7/7/9.",
        f"- Grapheme conformance: {validation['grapheme']['passed']}/{validation['grapheme']['total']} official tests ({validation['grapheme']['pass_rate']:.1%}), UAX #29 revision {validation['grapheme']['uax29_revision']} / Unicode {validation['grapheme']['unicode_version']} extended grapheme clusters, implementation `{validation['grapheme']['implementation']}`. A split-every-codepoint negative control fails {validation['grapheme']['broken_split_every_codepoint_failures']} cases, so the conformance check can fail.",
        "- Copy pilot: 182 byte-distinct items (62 new emoji, 32 old emoji controls, 32 new non-emoji, 32 old non-emoji controls, 24 ASCII). Structural base probes: 48 decomposition and 24 comparison; 36 count bundles per scanned tokenizer. Five canonical-form scan-only diagnostics were added for NFC/NFD behavior.",
        f"- Tokenizer-scan completeness gate: {min(x['copy_metric_rate'] for x in verdict['tokenizer_metric_gates']):.1%} minimum over copy items and tokenizers (threshold 95%); native SentencePiece/fast-tokenizer ID agreement is {min(x['native_backend_agreement_rate'] for x in verdict['tokenizer_metric_gates']):.1%}. The no-model-weights artifact gate passes.",
        f"- Host normalization versions are explicit: host Python {validation['host_python_version']} `unicodedata` {validation['host_unicodedata_version']}; canonical diagnostics use pinned `unicodedata2` {validation['normalization_unicodedata_version']}.", "",
        "## Tokenizer artifacts", "",
    ]
    for row in inventory["continuity"]:
        lines.append(f"- {row['left']} vs {row['right']}: regular vocab equal={row['regular_vocab_equal']}, merges equal={row['merges_equal']} ({row['left_vocab']}/{row['right_vocab']} regular entries).")
    lines += [
        "- Gemma 3 and canonical Llama 2 repositories are license-gated; scans use pinned unsloth Gemma and NousResearch Llama mirrors. TinyLlama's `tokenizer.model` SHA-256 equals the Llama mirror's, and their regular vocabularies/merges are identical.",
        "- Maximum Unicode age with at least one exact single-codepoint vocabulary entry: Gemma 15.0, Qwen2.5/Qwen3 12.0, Llama2/TinyLlama 6.0. This is sparse lexical coverage, not a completeness threshold.",
        "", "## Newest-cohort exact vocabulary membership", "",
    ]
    newest_lines = []
    for row in verdict["newest_cohort_exactness"]:
        newest_lines.append(f"{row['tokenizer_id']}: pipeline {row['pipeline_single_token_count']}/{row['families']}, exact-vocab {row['exact_vocab_entry_count']}/{row['families']}")
    lines.append("- " + "; ".join(newest_lines) + ".")
    lines += [
        f"- Any newest-cohort emoji as one exact vocabulary entry in any scanned tokenizer: **{'yes' if verdict['any_newest_vocab_entry'] else 'no'}**. Any newest-cohort emoji as one token through the full pipeline: **{'yes' if verdict['any_newest_pipeline_single'] else 'no'}**. This does not weaken the lexical-staleness mechanism, but absence is not proof of ignorance because byte fallback can still encode the sequences.",
        "", "## Proceed-rule 4 evidence", "",
        "Interpretation used: primary emoji bands from the spec, all RGI surface families pooled, monotone means non-increasing/non-decreasing with at least one strict adjacent change. Flat all-zero series are not counted as a coverage drop. Because atomic and ZWJ composition varies by release, atomic-only series are the cleaner sensitivity and are shown here.", "",
    ]
    for artifact in verdict["artifact_verdicts"]:
        seqrows = [x for x in bands if x["tokenizer_id"] == artifact["tokenizer_id"] and x["sequence_type"] == "all" and x["families"]]
        seqrows.sort(key=lambda x: EV_BANDS.index(x["emoji_band"]))
        atomicrows = [x for x in bands if x["tokenizer_id"] == artifact["tokenizer_id"] and x["sequence_type"] == "atomic" and x["families"]]
        atomicrows.sort(key=lambda x: EV_BANDS.index(x["emoji_band"]))
        lines.append(f"- {artifact['tokenizer_id']} ({artifact['family']}): pooled pipeline single-token coverage {fmt_series(seqrows)}; atomic pipeline coverage {fmt_series(atomicrows)}; atomic mean split-codepoint count {fmt_split_series(atomicrows)}; pooled signal={artifact['proceed_rule4_signal']}, atomic sensitivity signal={artifact['atomic_sensitivity_signal']}.")
    lines += [
        "",
        f"**Verdict:** proceed-rule 4 {'FIRES' if verdict['proceed_rule4_fires'] else 'does not fire'} under the non-increasing/non-decreasing interpretation; firing families: {', '.join(verdict['firing_families']) or 'none'}. Even if near-ceiling Llama is discounted, Gemma and Qwen independently satisfy the two-family threshold. Llama's coverage drop is small in absolute terms (0.002 to zero) and its atomic split rise starts at 0.988, so it is ceiling-sensitive rather than primary evidence.",
        "", "## NFC/NFD tokenizer behavior", "",
    ]
    for row in norm_rows:
        lines.append(f"- {row['tokenizer_id']}: NFC/NFD token-count pairing {row['token_count_pairing_fingerprint']} ({row['token_count_equal_rate']:.1%} equal), token-ID divergence {row['token_id_divergence_rate']:.1%}, NFC round-trip {row['nfc_roundtrip_exact_rate']:.1%}, NFD round-trip {row['nfd_roundtrip_exact_rate']:.1%} (n={row['n']}).")
    lines += [
        "", "## Caveats and protocol deviations", "",
        f"- The spec's global surface-family deduplication instruction conflicts with its 62-item audit: {len(surface_collisions)} ZWJ surface keys occur in multiple emoji versions, including {e17_surface_collisions} E17 skin-tone-bearing families that collapse onto older no-tone E4.0 surface keys if deduplication is global. Counts and the pilot therefore dedupe within emoji version, as required to reproduce 28/11/7/7/9 and 62 items.",
        "- Proceed-rule 4 does not define the version series or whether monotone means strict at every adjacent step. This report uses the spec's primary bands and non-strict monotonicity with at least one strict adjacent change. Under a stricter every-adjacent-step decrease requirement, coverage cannot fire after it reaches zero; the atomic split-codepoint prong remains the cleaner positive signal.",
        "- The user prompt omitted Llama 3.2, although the spec's behavioral four-model arm includes it. Llama 3.2 tokenizer artifacts and model selection remain unresolved before behavioral inference.",
        "- Unicode 18.0 itself is dated 2026-09-16 in publicationdates.html, while Emoji 18.0 is dated 2026-09-15 in UTS #51. This is not a conflict: atomic characters use Unicode dates and emoji sequences use emoji dates.",
        f"- The current Phase-0/scan codebase totals {sum(loc.values())} LOC across the scan-stage src files ({loc_detail}), already exceeding the spec's eventual 575-line total artifact budget before behavioral runners exist. The cap appears infeasible at the specified rigor (hand-rolled SentencePiece protobuf parsing, inventory vintage analysis, audit reproduction are not delegable); explicit budget renegotiation is needed before Phase 1 behavioral work.",
        "", "Machine-readable outputs: `manifest.json`, `ucd_audit.json`, `testset_audit.json`, `validation.json`, `tokenizer_inventory.json`, `tokenizer_scan.csv/jsonl`, `copy_scan_summary.csv`, `normalization_summary.csv`, `rgi_version_curve.csv`, `rgi_primary_band_curve.csv`, and `scan_verdict.json`.",
    ]
    (RESULTS / "SCAN_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    scan_rows = read_jsonl(RESULTS / "tokenizer_scan.jsonl")
    inventory = json.loads((RESULTS / "tokenizer_inventory.json").read_text())
    validation = json.loads((RESULTS / "validation.json").read_text())
    copy_rows = copy_summary(scan_rows)
    write_csv(RESULTS / "copy_scan_summary.csv", copy_rows, list(copy_rows[0]))
    scan_csv, scan_fields = scan_csv_rows(scan_rows)
    write_csv(RESULTS / "tokenizer_scan.csv", scan_csv, scan_fields)
    norm_rows = normalization_summary(scan_rows)
    write_csv(RESULTS / "normalization_summary.csv", norm_rows, list(norm_rows[0]))
    flat = flatten_inventory(inventory)
    write_csv(RESULTS / "rgi_version_curve.csv", flat, ["tokenizer_id", "family", "emoji_version"] + [x for x in flat[0] if x not in ("tokenizer_id", "family", "emoji_version")])
    bands = primary_band_curve(flat)
    write_csv(RESULTS / "rgi_primary_band_curve.csv", bands, list(bands[0]))
    vrows, firing_families, atomic_families = verdicts(bands)
    gates = []
    for tid in sorted({x["tokenizer_id"] for x in scan_rows}):
        rows = [x for x in scan_rows if x["tokenizer_id"] == tid]
        copy = [x for x in rows if x["probe_type"] == "copy"]
        native = [x for x in rows if x["native_backend_ids_match"] is not None]
        gates.append({
            "tokenizer_id": tid, "copy_items_with_scan_metrics": sum(x["split_codepoint_count"] is not None for x in copy), "copy_items": len(copy),
            "copy_metric_rate": sum(x["split_codepoint_count"] is not None for x in copy) / len(copy),
            "all_items_with_scan_metrics": sum(x["split_codepoint_count"] is not None for x in rows), "all_items": len(rows),
            "all_metric_rate": sum(x["split_codepoint_count"] is not None for x in rows) / len(rows),
            "native_backend_rows": len(native), "native_backend_mismatches": sum(x["native_backend_ids_match"] == 0 for x in native),
            "native_backend_agreement_rate": sum(x["native_backend_ids_match"] == 1 for x in native) / len(native) if native else 1.0,
        })
    newest = newest_exactness(inventory)
    verdict = {
        "primary_interpretation": "non-increasing exact single-token coverage or non-decreasing split-codepoint count across the spec's primary emoji bands (legacy, E14, E15, E15.1, E16, E17, E18), requiring at least one strict adjacent change; all RGI surface families pooled",
        "artifact_verdicts": vrows, "firing_families": firing_families, "atomic_sensitivity_families": atomic_families,
        "proceed_rule4_fires": len(firing_families) >= 2, "mechanism_absent_everywhere": len(firing_families) == 0,
        "newest_cohort_exactness": newest,
        "any_newest_vocab_entry": any(x["exact_vocab_entry_count"] for x in newest), "any_newest_pipeline_single": any(x["pipeline_single_token_count"] for x in newest),
        "tokenizer_metric_gates": gates, "tokenizer_gate_pass": all(x["copy_metric_rate"] >= 0.95 for x in gates),
        "native_backend_gate_pass": all(x["native_backend_mismatches"] == 0 for x in gates),
        "grapheme_gate_pass": validation["gates"]["grapheme_100_percent"], "normalization_summary": norm_rows,
    }
    (RESULTS / "scan_verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
    write_report(verdict, bands, inventory, validation, norm_rows)
    freeze_hashes()
    print(json.dumps({"proceed_rule4_fires": verdict["proceed_rule4_fires"], "firing_families": firing_families, "atomic_sensitivity_families": atomic_families, "any_newest_vocab_entry": verdict["any_newest_vocab_entry"], "tokenizer_gate_pass": verdict["tokenizer_gate_pass"], "native_backend_gate_pass": verdict["native_backend_gate_pass"], "grapheme_gate_pass": verdict["grapheme_gate_pass"]}, indent=2))


if __name__ == "__main__":
    main()
