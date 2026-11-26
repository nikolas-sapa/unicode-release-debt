from __future__ import annotations

import csv
import hashlib
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ROUND2 = RESULTS / "behavioral_round2"
FIGURES = ROOT / "docs" / "figures"
FONTS = ROOT / "docs" / "assets" / "fonts"

ACCENT = "#006bff"
ACCENT_LIGHT = "#e8f1ff"
INK = "#171717"
TEXT = "#4d4d4d"
MUTED = "#8f8f8f"
GRID = "#e6e6e6"
BORDER = "#d9d9d9"
CARD = "#fafafa"
WHITE = "#ffffff"
TEAL = "#0f766e"
PURPLE = "#7c3aed"
DARK_GRAY = "#4d4d4d"
MID_GRAY = "#a8a8a8"
LIGHT_GRAY = "#c9c9c9"

FAMILY_STYLE = {
    "Gemma": {"color": ACCENT, "marker": "o", "linestyle": "-", "label": "Gemma"},
    "Qwen": {"color": PURPLE, "marker": "s", "linestyle": "--", "label": "Qwen"},
    "Llama": {"color": TEAL, "marker": "^", "linestyle": ":", "label": "Llama"},
}
REPRESENTATIVES = {
    "Gemma": "gemma3-1b-it-mirror",
    "Qwen": "qwen2.5-0.5b-instruct",
    "Llama": "llama2-7b-hf-mirror",
}
BANDS = ["legacy", "E14.0", "E15.0", "E15.1", "E16.0", "E17.0", "E18.0"]
ZWJ_BANDS = ["legacy", "E15.1", "E17.0"]
MODEL_SHORT = {
    "qwen2.5-3b-instruct-bf16": "Qwen2.5-3B",
    "qwen3-4b-instruct-2507-mlx8": "Qwen3-4B",
    "gemma3-4b-it-mlx8-mirror": "Gemma3-4B",
    "tinyllama-1.1b-chat-v1.0-bf16": "TinyLlama-1.1B",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_json(path: Path):
    return json.loads(path.read_text())


def setup_matplotlib():
    for name in ("Geist-Regular.ttf", "Geist-SemiBold.ttf", "GeistMono-Regular.ttf"):
        path = FONTS / name
        if not path.exists():
            raise FileNotFoundError(f"required pinned font missing: {path}")
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": "Geist",
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": BORDER,
        "axes.facecolor": CARD,
        "axes.labelcolor": TEXT,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "svg.fonttype": "path",
        "svg.hashsalt": "unicode-release-debt",
    })


def validate_family_duplicates(rows: list[dict[str, str]]):
    def key(row):
        return (row["sequence_type"], row["emoji_band"], row["families"], row["exact_single_token_rate"], row["mean_split_codepoint_count"], row["mean_tokens_per_cluster"])

    def series(tid):
        return sorted(key(x) for x in rows if x["tokenizer_id"] == tid)

    if series("qwen2.5-0.5b-instruct") != series("qwen3-0.6b"):
        raise ValueError("Qwen2.5/Qwen3 scan rows are no longer exact duplicates; family collapse would be dishonest")
    if series("llama2-7b-hf-mirror") != series("tinyllama-1.1b-chat"):
        raise ValueError("Llama2/TinyLlama scan rows are no longer exact duplicates; family collapse would be dishonest")


def series_by_band(rows: list[dict[str, str]], tokenizer_id: str, sequence_type: str, bands: list[str], field: str):
    values, ns = [], []
    for band in bands:
        row = next((x for x in rows if x["tokenizer_id"] == tokenizer_id and x["sequence_type"] == sequence_type and x["emoji_band"] == band), None)
        if row is None:
            raise ValueError(f"missing row: {tokenizer_id} {sequence_type} {band}")
        values.append(float(row[field]) if row[field] else np.nan)
        ns.append(int(row["families"]))
    return values, ns


def add_heading(fig, eyebrow: str, title: str, subtitle: str):
    fig.text(0.055, 0.955, eyebrow.upper(), fontsize=8.5, color=ACCENT, weight=600)
    fig.text(0.055, 0.905, title, fontsize=20, color=INK, weight=600)
    fig.text(0.055, 0.865, subtitle, fontsize=10.2, color=TEXT)


def add_footer(fig, text: str):
    wrapped = textwrap.fill(text, width=168, break_long_words=False, break_on_hyphens=False)
    fig.text(0.055, 0.026, wrapped, fontsize=7.7, color=MUTED, linespacing=1.45)


def style_axis(ax):
    ax.set_facecolor(CARD)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def save_figure(fig, stem: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES / f"{stem}.png", FIGURES / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=170, facecolor=fig.get_facecolor(), metadata={"Software": "unicode-release-debt src/make_figures.py"})
    fig.savefig(paths[1], facecolor=fig.get_facecolor(), metadata={"Creator": "unicode-release-debt src/make_figures.py", "Date": None})
    plt.close(fig)
    return paths


def percent_tick(x):
    return f"{x:.0f}%"


def figure_coverage(rows: list[dict[str, str]], scan_verdict: dict) -> list[Path]:
    fig = plt.figure(figsize=(11.8, 6.9))
    add_heading(
        fig,
        "Tokenizer scan / mechanism",
        "Exact single-token coverage collapses after legacy emoji",
        "Full-pipeline coverage of skin-tone-stripped RGI surface families; representative artifact per tokenizer family.",
    )
    axes = [fig.add_axes([0.075, 0.205, 0.405, 0.57]), fig.add_axes([0.555, 0.205, 0.405, 0.57])]
    panels = [("all", "All RGI surface families", axes[0]), ("atomic", "Strict atomic sensitivity", axes[1])]
    x = np.arange(len(BANDS))
    for sequence_type, panel_title, ax in panels:
        style_axis(ax)
        ax.axvspan(2.55, 6.45, color=ACCENT_LIGHT, alpha=0.55, zorder=0)
        ns_for_ticks = None
        for family, tid in REPRESENTATIVES.items():
            vals, ns = series_by_band(rows, tid, sequence_type, BANDS, "exact_single_token_rate")
            ns_for_ticks = ns
            st = FAMILY_STYLE[family]
            ax.plot(x, 100 * np.array(vals), color=st["color"], linewidth=2.2, linestyle=st["linestyle"], marker=st["marker"], markersize=5.2, markeredgewidth=0, label=st["label"], zorder=3)
        labels = []
        for band, n in zip(BANDS, ns_for_ticks):
            labels.append(f"{band}\nn={n:,}")
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=8.3)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20), [percent_tick(v) for v in range(0, 101, 20)])
        ax.set_title(panel_title, loc="left", fontsize=12.5, weight=600, color=INK, pad=11)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=TEXT, linewidth=0.8, zorder=2)
    axes[0].set_ylabel("Exact single-token coverage", fontsize=10)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8.8, ncols=1)
    exactness = scan_verdict["newest_cohort_exactness"]
    if not exactness or not all(x["pipeline_single_token_count"] == 0 and x["exact_vocab_entry_count"] == 0 for x in exactness):
        raise ValueError("scan verdict no longer supports the 0/62 newest-cohort callout")
    callout = "Newest copy-pilot cohort\n0/62 exact single-token in every scanned tokenizer"
    axes[0].text(0.985, 0.74, callout, transform=axes[0].transAxes, ha="right", va="top", fontsize=9.2, weight=600, color=ACCENT, bbox={"boxstyle": "round,pad=0.55", "facecolor": WHITE, "edgecolor": ACCENT, "linewidth": 1.1})
    axes[1].text(0.50, 0.52, "E15.1 contributes no strict atomic families\nin the Unicode 18 snapshot", transform=axes[1].transAxes, ha="center", va="center", fontsize=9, color=MUTED, bbox={"boxstyle": "round,pad=0.45", "facecolor": WHITE, "edgecolor": BORDER, "linewidth": 0.8})
    add_footer(
        fig,
        "Source: results/rgi_primary_band_curve.csv and results/scan_verdict.json. Coverage is exact full-pipeline single-token encoding, not semantic knowledge. "
        "Qwen2.5/Qwen3 scan rows are exactly identical, as are Llama2/TinyLlama rows; one representative artifact is plotted per family. "
        "Cells count skin-tone-stripped surface families deduplicated within emoji version.",
    )
    return save_figure(fig, "f1_tokenizer_coverage_collapse")


def figure_fragmentation(rows: list[dict[str, str]]) -> list[Path]:
    fig = plt.figure(figsize=(10.8, 6.6))
    add_heading(
        fig,
        "Tokenizer scan / fragmentation",
        "Post-legacy ZWJ cohorts fragment into more tokens per cluster",
        "Mean tokens per extended grapheme cluster for RGI ZWJ surface families in sufficiently populated release bands.",
    )
    ax = fig.add_axes([0.09, 0.20, 0.76, 0.58])
    style_axis(ax)
    x = np.arange(len(ZWJ_BANDS))
    ns_for_ticks = None
    max_seen = 0.0
    endpoints = []
    for family, tid in REPRESENTATIVES.items():
        vals, ns = series_by_band(rows, tid, "zwj", ZWJ_BANDS, "mean_tokens_per_cluster")
        ns_for_ticks = ns
        max_seen = max(max_seen, max(vals))
        st = FAMILY_STYLE[family]
        ax.plot(x, vals, color=st["color"], linewidth=2.5, linestyle=st["linestyle"], marker=st["marker"], markersize=7, markeredgewidth=0, label=st["label"], zorder=3)
        endpoints.append((vals[-1], family, st["color"]))
    labels = []
    pretty = {"legacy": "Legacy control", "E15.1": "Emoji 15.1", "E17.0": "Emoji 17.0"}
    for band, n in zip(ZWJ_BANDS, ns_for_ticks):
        labels.append(f"{pretty[band]}\nn={n:,} families")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=9)
    upper = int(np.ceil(max_seen + 1.0))
    ax.set_ylim(0, upper)
    ax.set_yticks(range(0, upper + 1, 3))
    ax.set_ylabel("Mean tokens per grapheme cluster", fontsize=10)
    ax.set_xlim(-0.25, 2.72)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color=TEXT, linewidth=0.8)
    for value, family, color in sorted(endpoints):
        ax.text(2.08, value, f"{value:.1f}  {family}", va="center", ha="left", fontsize=9.3, weight=600, color=color)
    ax.text(0.015, 0.94, "Legacy-to-E17 mean increase", transform=ax.transAxes, fontsize=9, color=MUTED, weight=600)
    for family, tid in REPRESENTATIVES.items():
        vals, _ = series_by_band(rows, tid, "zwj", ZWJ_BANDS, "mean_tokens_per_cluster")
        increase = vals[-1] - vals[0]
        ax.text(0.015, 0.88 - 0.055 * list(REPRESENTATIVES).index(family), f"{family}: +{increase:.1f} tokens/cluster", transform=ax.transAxes, fontsize=9, color=FAMILY_STYLE[family]["color"])
    add_footer(
        fig,
        "Source: results/rgi_primary_band_curve.csv. Descriptive tokenizer mechanism, not adjusted for sequence complexity. "
        "Singleton E14.0/E15.0 ZWJ cells are omitted from the visual trend; E16.0 and E18.0 contribute no fully-qualified ZWJ surface families in this snapshot. "
        "Qwen2.5/Qwen3 and Llama2/TinyLlama duplicate exactly; representatives are plotted.",
    )
    return save_figure(fig, "f2_zwj_fragmentation_gradient")


def exact_count_text(num: str | int, den: str | int) -> str:
    return f"{num}/{den}"


def figure_central_contrast() -> list[Path]:
    verdict = load_json(ROUND2 / "round2_verdict.json")
    copy_rows = read_csv(ROUND2 / "round2_copy_newest_old_summary.csv")
    count_rows = [x for x in read_csv(ROUND2 / "round2_count_fragmentation_summary.csv") if x["scope"] == "pooled"]
    valid_models = verdict["valid_models"]
    if len(valid_models) != 4:
        raise ValueError("figure expects four gate-valid models from the frozen round-2 verdict")
    if verdict["invalid_models"] != ["qwen2.5-7b-instruct-mlx8"]:
        raise ValueError("invalid-model identity changed; figure annotation must be revisited")
    copy_by_model = {x["model_id"]: x for x in copy_rows}
    count_by_model = {x["model_id"]: x for x in count_rows}
    missing = [m for m in valid_models if m not in copy_by_model or m not in count_by_model]
    if missing:
        raise ValueError(f"missing summary rows for {missing}")

    fig = plt.figure(figsize=(12.4, 7.1))
    add_heading(
        fig,
        "Behavioral round 2 / central contrast",
        "Byte copying stays at ceiling; grapheme operations do not",
        "Gate-valid models only. Exact rates use total failure scoring: format failure and content failure both count as failure.",
    )
    ax_copy = fig.add_axes([0.075, 0.215, 0.29, 0.56])
    ax_count = fig.add_axes([0.455, 0.215, 0.505, 0.56])
    for ax in (ax_copy, ax_count):
        style_axis(ax)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25), [percent_tick(v) for v in range(0, 101, 25)])
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=TEXT, linewidth=0.8)
    x = np.arange(len(valid_models))
    labels = [MODEL_SHORT[m] for m in valid_models]

    ax_copy.set_title("A  Copy exact-match rate", loc="left", fontsize=12.5, weight=600, pad=11)
    for i, model in enumerate(valid_models):
        row = copy_by_model[model]
        newest = 100 * float(row["newest_exact_rate"])
        old = 100 * float(row["old_control_exact_rate"])
        ax_copy.scatter(i - 0.09, old, s=56, color=MID_GRAY, marker="o", zorder=3, label="Old controls" if i == 0 else None)
        ax_copy.scatter(i + 0.09, newest, s=62, color=ACCENT, marker="o", zorder=3, label="Newest cohort" if i == 0 else None)
    ax_copy.set_xticks(x, labels)
    ax_copy.tick_params(axis="x", labelsize=8.6)
    ax_copy.set_ylabel("Exact rate", fontsize=10)
    ax_copy.legend(loc="lower right", frameon=False, fontsize=8.7)
    ax_copy.text(0.03, 0.89, "Every valid model:\n94/94 newest, 64/64 old", transform=ax_copy.transAxes, fontsize=9.4, weight=600, color=ACCENT, va="top", bbox={"boxstyle": "round,pad=0.55", "facecolor": WHITE, "edgecolor": ACCENT, "linewidth": 1.0})

    ax_count.set_title("B  Grapheme-count exact rate by fragmentation", loc="left", fontsize=12.5, weight=600, pad=11)
    for i, model in enumerate(valid_models):
        row = count_by_model[model]
        low = 100 * (1.0 - float(row["low_failure_rate"]))
        high = 100 * (1.0 - float(row["high_failure_rate"]))
        ax_count.plot([i, i], [low, high], color=LIGHT_GRAY, linewidth=2.2, zorder=2)
        ax_count.scatter(i - 0.075, low, s=58, color=DARK_GRAY, marker="o", zorder=3, label="Low fragmentation" if i == 0 else None)
        ax_count.scatter(i + 0.075, high, s=66, color=ACCENT, marker="D", zorder=3, label="High fragmentation" if i == 0 else None)
        gap = float(row["high_minus_low_failure_pp"])
        ytext = max(low, high) + 8 if max(low, high) < 82 else max(low, high) - 13
        ax_count.text(i, ytext, f"+{gap:.1f} pp\nfailure", ha="center", va="center", fontsize=8.4, color=ACCENT if gap >= 15 else MUTED, weight=600 if gap >= 15 else 400)
    ax_count.set_xticks(x, labels)
    ax_count.tick_params(axis="x", labelsize=8.6)
    ax_count.legend(loc="upper right", frameon=False, fontsize=8.7)
    rule2_models = verdict["behavioral_proceed_rule2_models"]
    ax_count.text(0.015, 0.945, f"Proceed rule 2 fires in {len(rule2_models)}/4 valid models", transform=ax_count.transAxes, fontsize=9.2, weight=600, color=ACCENT, va="top")
    ax_count.text(0.015, 0.875, "Decomposition floor: 0/48 exact in every valid model", transform=ax_count.transAxes, fontsize=8.8, color=INK, va="top", bbox={"boxstyle": "round,pad=0.42", "facecolor": WHITE, "edgecolor": BORDER, "linewidth": 0.8})

    invalid = verdict["validity_gates"]["qwen2.5-7b-instruct-mlx8"]["wrapper"]
    fig.text(0.955, 0.835, f"Excluded from rule evaluation: qwen2.5-7b\nwrapper failures {invalid['format_failures']}/{invalid['n']} ({100 * invalid['format_failure_rate']:.1f}%)", ha="right", va="top", fontsize=8.7, color=TEXT, bbox={"boxstyle": "round,pad=0.55", "facecolor": CARD, "edgecolor": BORDER, "linewidth": 0.9})
    add_footer(
        fig,
        "Source: results/behavioral_round2/round2_verdict.json, round2_copy_newest_old_summary.csv, and round2_count_fragmentation_summary.csv. "
        "Count rates pool legacy and newest bundles (18 low-fragmentation and 18 high-fragmentation rows per model). "
        "TinyLlama emits no valid structural answer format; its structural zeros are therefore format-floor results, not representation evidence. "
        "Copy gates: ASCII at least 23/24 and copy-wrapper failure at most 5%.",
    )
    return save_figure(fig, "f3_copy_vs_structural_contrast")


def box(ax, xywh, number, title, body, accent=False):
    x, y, w, h = xywh
    fill = ACCENT_LIGHT if accent else CARD
    edge = ACCENT if accent else BORDER
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", transform=ax.transAxes, facecolor=fill, edgecolor=edge, linewidth=1.1, zorder=2)
    ax.add_patch(patch)
    circle = plt.Circle((x + 0.025, y + h - 0.045), 0.016, transform=ax.transAxes, facecolor=ACCENT, edgecolor="none", zorder=3)
    ax.add_patch(circle)
    ax.text(x + 0.025, y + h - 0.045, number, transform=ax.transAxes, ha="center", va="center", fontsize=6.7, color=WHITE, weight=600, zorder=4)
    ax.text(x + 0.047, y + h - 0.045, title, transform=ax.transAxes, ha="left", va="center", fontsize=7.8, color=INK, weight=600, zorder=4)
    ax.text(x + 0.022, y + h - 0.091, body, transform=ax.transAxes, ha="left", va="top", fontsize=7.75, color=TEXT, linespacing=1.42, zorder=4)


def arrow(ax, start, end, dashed=False):
    arr = FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=11, linewidth=1.3, color=ACCENT if not dashed else MID_GRAY, linestyle="--" if dashed else "-", shrinkA=3, shrinkB=3, zorder=5)
    ax.add_patch(arr)


def figure_pipeline(scan_verdict: dict, round2_verdict: dict) -> list[Path]:
    manifest = load_json(RESULTS / "manifest.json")
    input_count = len(manifest["inputs"])
    completeness = round2_verdict["completeness"]
    rows_per_model = completeness[0]["expected_rows"] if completeness else 0
    model_count = len(completeness)
    total_rows = sum(x["rows"] for x in completeness)
    if input_count != 32 or rows_per_model != 290 or total_rows != 1450:
        raise ValueError("pipeline figure constants no longer match artifact manifests")
    if not scan_verdict["proceed_rule4_fires"] or not round2_verdict["hard_kill_operational_arm"] or not round2_verdict["behavioral_proceed_rule2_fires"]:
        raise ValueError("pipeline verdict annotations no longer match machine verdicts")

    fig = plt.figure(figsize=(12.6, 6.9))
    add_heading(
        fig,
        "Artifact / measurement pipeline",
        "A gated measurement pipeline, not a prompt cherry-pick",
        "Frozen public inputs and tokenizer artifacts feed independent scan and behavior arms; interpretation occurs only behind validity gates.",
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    panel = FancyBboxPatch((0.045, 0.085), 0.91, 0.67, boxstyle="round,pad=0.012,rounding_size=0.018", transform=ax.transAxes, facecolor=WHITE, edgecolor=BORDER, linewidth=1.0, zorder=1)
    ax.add_patch(panel)

    xs = [0.065, 0.247, 0.429, 0.611, 0.793]
    w, h = 0.15, 0.19
    y_top, y_bottom = 0.50, 0.17
    box(ax, (xs[0], y_top, w, h), "01", "UNICODE INPUTS", f"Unicode 18.0 / Emoji 18\nCLDR 49; {input_count} pinned files\nindependent hash replication")
    box(ax, (xs[1], y_top, w, h), "03", "TEST UNIVERSE", "182 copy items per model\n108 structural rows per model\nASCII and legacy controls")
    box(ax, (xs[2], y_top, w, h), "05", "BEHAVIORAL PROBES", f"copy / count / decompose\ncompare; greedy local inference\n{model_count} x {rows_per_model} rows = {total_rows:,}\nappend-only raw JSONL")
    box(ax, (xs[3], y_top, w, h), "06", "VALIDITY GATES", "ASCII copy at least 23/24\nwrapper failure at most 5%\nmutation-tested extraction")
    box(ax, (xs[4], y_top, w, h), "07", "VERDICTS", "scan rule 4 fires\ncopy arm hard-killed\nstructural rule 2 proceeds", accent=True)
    box(ax, (xs[0], y_bottom, w, h), "02", "PINNED TOKENIZERS", "five artifacts, three families\nvocabulary and merge continuity\nhash-pinned tokenizer files")
    box(ax, (xs[1], y_bottom, w, h), "04", "TOKENIZER SCAN", "exact single-token coverage\ntokens per cluster\nsplit codepoints; NFC/NFD")

    arrow(ax, (xs[0] + w, y_top + h / 2), (xs[1], y_top + h / 2))
    arrow(ax, (xs[1] + w, y_top + h / 2), (xs[2], y_top + h / 2))
    arrow(ax, (xs[2] + w, y_top + h / 2), (xs[3], y_top + h / 2))
    arrow(ax, (xs[3] + w, y_top + h / 2), (xs[4], y_top + h / 2))
    arrow(ax, (xs[0] + w, y_bottom + h / 2), (xs[1], y_bottom + h / 2))
    arrow(ax, (xs[1] + w, y_bottom + h / 2), (xs[2] + 0.025, y_top - 0.01))
    ax.text(xs[2] + 0.026, 0.425, "scan covariates join behavior by item and tokenizer", transform=ax.transAxes, fontsize=7.8, color=MUTED, ha="left")

    add_footer(
        fig,
        "Counts are validated against results/manifest.json, results/scan_verdict.json, and results/behavioral_round2/round2_verdict.json at figure-generation time. "
        "Raw responses remain append-only; superseded wrong-tokenizer control rows are retained with explicit superseded filenames.",
    )
    return save_figure(fig, "f4_measurement_pipeline")


def write_manifest(output_paths: list[Path]):
    inputs = [
        RESULTS / "rgi_primary_band_curve.csv",
        RESULTS / "scan_verdict.json",
        RESULTS / "manifest.json",
        ROUND2 / "round2_verdict.json",
        ROUND2 / "round2_copy_newest_old_summary.csv",
        ROUND2 / "round2_count_fragmentation_summary.csv",
    ]
    data = {
        "generator": "src/make_figures.py",
        "deterministic_metadata": "SVG Date metadata removed; fixed svg.hashsalt; no wall-clock timestamp embedded by this manifest",
        "matplotlib_version": matplotlib.__version__,
        "python_version": sys.version.split()[0],
        "palette_constraints": {"primary_accent": ACCENT, "orange_used": False, "emoji_glyphs_used_in_figure_text": False},
        "fonts": {p.name: sha256(p) for p in sorted(FONTS.glob("*.ttf"))},
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "outputs": {str(p.relative_to(ROOT)): {"sha256": sha256(p), "size": p.stat().st_size} for p in sorted(output_paths)},
    }
    path = FIGURES / "figure_manifest.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    setup_matplotlib()
    rows = read_csv(RESULTS / "rgi_primary_band_curve.csv")
    validate_family_duplicates(rows)
    scan_verdict = load_json(RESULTS / "scan_verdict.json")
    round2_verdict = load_json(ROUND2 / "round2_verdict.json")
    outputs = []
    outputs += figure_coverage(rows, scan_verdict)
    outputs += figure_fragmentation(rows)
    outputs += figure_central_contrast()
    outputs += figure_pipeline(scan_verdict, round2_verdict)
    write_manifest(outputs)
    print(json.dumps({"outputs": [str(p.relative_to(ROOT)) for p in outputs], "manifest": "docs/figures/figure_manifest.json"}, indent=2))


if __name__ == "__main__":
    main()
