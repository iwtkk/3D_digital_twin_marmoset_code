"""
I/O ledger: make_table   (run with .venv)

Purpose:
    Measure a .blend in Blender (headless) using the definitions in config/environment_table.yaml,
    and output the environment accuracy table (geometric accuracy of the environment model
    compared with the blueprint values) and summary statistics.

Inputs:
    config:
        type: file (YAML)
        path_source: config/environment_table.yaml
        content: blender_bin, blends{name: path}, default_blend, output_dir, match_tol_mm,
                 rows[{label, paper_no, part, blueprint_mm, method, ...}]
    blend:
        type: file (.blend)
        path_source: config.blends[--blend]
        note: read only (Blender does not save it)

Outputs (output_dir/{blend_name}/):
    table.csv:
        rows: same order and count as config.rows (measured with by_blend[{blend name}] if present)
        columns: label(int|str), paper_no(int|empty), part(str), blueprint_mm(float|empty = not filled in), model_mm(float),
                 error_mm(float, = model − blueprint), abs_error_mm(float),
                 ratio(float, = model / blueprint, signed), match(bool),
                 objects(str), geometry(str, evaluated|mesh for bbox_extent),
                 hidden_objects(str, those with hide_render), animated_objects(str, those with an action),
                 stale_objects(str, those whose saved matrix_world is stale), error(str)
    table.md:
        content: the same table as table.csv + summary statistics (Markdown)
    table.tex:
        content: LaTeX table (booktabs) in the same format as the manuscript table. Columns = No., blueprint,
                 3D model, |difference|, ratio (signed). Ordered by paper_no, with avg./max./min. rows at the end
                 Digits: mm with 2 decimals (<10^{-3} below 10^-3), ratio with 3 decimals
        note: requires the booktabs package. Include it in the manuscript with \input
    summary.json:
        content: n_rows, n_measured, n_no_blueprint, n_match, n_deviate, n_larger, n_smaller, mae_all_mm, mae_deviate_mm,
                 max_abs_error_mm, ratio_mean, ratio_min, ratio_max, mean_abs_rel_error_pct,
                 blend, blend_md5, blender_version, match_tol_mm
    measured_raw.json:
        content: the raw output of measure_blend.py

Usage:
    .venv/bin/python scripts/06_validation/environment_accuracy/make_table.py                    # = --blend digitaltwin
    .venv/bin/python scripts/06_validation/environment_accuracy/make_table.py --blend v250430
    .venv/bin/python scripts/06_validation/environment_accuracy/make_table.py --blend dataset
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "config" / "environment_table.yaml"

COLUMNS = ["label", "paper_no", "part", "blueprint_mm", "model_mm", "error_mm",
           "abs_error_mm", "ratio", "match", "objects", "geometry", "hidden_objects", "stale_objects",
           "animated_objects", "error"]


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def resolve_rows(rows, blend_name):
    """If rows[i].by_blend[blend_name] exists, override the row with the definition for that version."""
    out = []
    for row in rows:
        r = {k: v for k, v in row.items() if k != "by_blend"}
        r.update((row.get("by_blend") or {}).get(blend_name, {}))
        out.append(r)
    return out


def run_blender(blender_bin, blend, rows, workdir):
    spec = workdir / "spec.json"
    out = workdir / "measured.json"
    spec.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    cmd = [blender_bin, "-b", str(blend), "--python", str(HERE / "measure_blend.py"),
           "--", str(spec), str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise RuntimeError(f"blender failed (returncode={proc.returncode})")
    return json.loads(out.read_text(encoding="utf-8"))


def build_table(rows, measured, tol):
    by_label = {m["label"]: m for m in measured["rows"]}
    table = []
    for row in rows:
        m = by_label[row["label"]]
        bp = None if row["blueprint_mm"] is None else float(row["blueprint_mm"])
        rec = {"label": row["label"], "paper_no": row.get("paper_no") or "",
               "part": row["part"], "blueprint_mm": bp, "model_mm": m["model_mm"],
               "error_mm": None, "abs_error_mm": None, "ratio": None, "match": None,
               "objects": ";".join(o["name"] for o in m["objects"]),
               "hidden_objects": ";".join(o["name"] for o in m["objects"] if o["hide_render"]),
               "stale_objects": ";".join(o["name"] for o in m["objects"] if o["matrix_world_stale"]),
               "animated_objects": ";".join(o["name"] for o in m["objects"] if o["animated"]),
               "geometry": row.get("geometry", "evaluated") if row["method"] == "bbox_extent" else "",
               "error": m["error"] or ""}
        if m["model_mm"] is not None and bp is not None:
            err = m["model_mm"] - bp
            rec.update(error_mm=err, abs_error_mm=abs(err), ratio=m["model_mm"] / bp,
                       match=abs(err) < tol)
        table.append(rec)
    return table


def summarize(table, tol):
    # Rows without a blueprint value (blueprint_mm: null) appear in the table but not in the summary
    ok = [r for r in table if r["error_mm"] is not None]
    dev = [r for r in ok if not r["match"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {
        "n_rows": len(table),
        "n_measured": len(ok),
        "n_no_blueprint": sum(r["blueprint_mm"] is None for r in table),
        "n_match": len(ok) - len(dev),
        "n_deviate": len(dev),
        "n_larger": sum(r["error_mm"] >= tol for r in ok),
        "n_smaller": sum(r["error_mm"] <= -tol for r in ok),
        "mae_all_mm": mean([r["abs_error_mm"] for r in ok]),
        "mae_deviate_mm": mean([r["abs_error_mm"] for r in dev]),
        "max_abs_error_mm": max((r["abs_error_mm"] for r in ok), default=None),
        "ratio_mean": mean([r["ratio"] for r in ok]),
        "ratio_min": min((r["ratio"] for r in ok), default=None),
        "ratio_max": max((r["ratio"] for r in ok), default=None),
        "mean_abs_rel_error_pct": mean([abs(r["ratio"] - 1) * 100 for r in ok]),
        "match_tol_mm": tol,
    }


def fmt(v, nd):
    return "" if v is None else f"{v:.{nd}f}"


# Digits in the LaTeX table. Blueprint values are integer mm and Blender coordinates are float32
# (about 0.0002 mm steps at 2 m), so show down to 0.01 mm. The ratio uses 3 decimals so that
# 0.989 and 1.010 can be told apart (signed = model / blueprint)
TEX_MM_DIGITS = 2
TEX_RATIO_DIGITS = 3


def tex_mm(v):
    """A mm value with 2 decimals. Below 10^-3 mm (matched items) it becomes "$<10^{-3}$" as in the manuscript."""
    if v is None:
        return "--"
    if abs(v) < 1e-3:
        return r"$<10^{-3}$"
    return f"{v:.{TEX_MM_DIGITS}f}".replace("-", "$-$")


def tex_ratio(v):
    return "--" if v is None else f"{v:.{TEX_RATIO_DIGITS}f}"


def write_tex(path, table, summary):
    """LaTeX table (booktabs) in the same format as the manuscript table.
    If every row has a paper_no, rows are ordered by paper_no. If even one is missing, all rows use the
    xlsx label in config order (mixing them would make numbers collide, e.g. xlsx #13 vs paper No.13).
    Differences from the manuscript: unified digits (mm with 2 decimals, ratio with 3), and the ratio
    is signed instead of 1+|1-ratio|."""
    use_paper_no = all(r["paper_no"] != "" for r in table)
    rows = sorted(table, key=lambda r: int(r["paper_no"])) if use_paper_no else table
    ok = [r for r in rows if r["abs_error_mm"] is not None]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{Blueprint vs.\ 3D model size comparison}",
             r"\label{tab:blueprint_3Dsize}",
             r"\begin{tabular}{r p{25mm} p{25mm} p{25mm} p{30mm}}", r"\toprule",
             r"\textbf{Blueprint object No.} &",
             r"\textbf{MarmoCage blueprint [mm]} &",
             r"\textbf{3D model size [mm]} &",
             r"\textbf{Absolute difference [mm]} &",
             r"\textbf{Ratio (3D/blueprint)} \\", r"\midrule"]
    for r in rows:
        no = r["paper_no"] if use_paper_no else r["label"]
        bp = "--" if r["blueprint_mm"] is None else f"{r['blueprint_mm']:.0f}"
        lines.append(f"{no} & {bp} & {r['model_mm']:.{TEX_MM_DIGITS}f} & "
                     f"{tex_mm(r['abs_error_mm'])} & {tex_ratio(r['ratio'])} \\\\")
    if ok:
        err = [r["abs_error_mm"] for r in ok]
        rat = [r["ratio"] for r in ok]
        lines += [r"\midrule",
                  rf"\textbf{{avg.}} &  &  & {tex_mm(sum(err) / len(err))} & {tex_ratio(sum(rat) / len(rat))} \\",
                  rf"\textbf{{max.}} &  &  & {tex_mm(max(err))} & {tex_ratio(max(rat))} \\",
                  rf"\textbf{{min.}} &  &  & {tex_mm(min(err))} & {tex_ratio(min(rat))} \\"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(out_dir, table, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(table)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    lines = [f"# Environment accuracy table — {summary['blend']}", "",
             f"- md5: `{summary['blend_md5']}`",
             f"- Blender {summary['blender_version']}", "",
             "| label | paper No. | part | object | blueprint [mm] | model [mm] | error [mm] | ratio | match | note |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        note = r["error"] or "; ".join(
            f"{k}: {r[c]}" for k, c in [("hidden", "hidden_objects"), ("matrix_world stale", "stale_objects"),
                                         ("animated", "animated_objects")]
            if r[c])
        match = "" if r["match"] is None else ("✓" if r["match"] else "")
        obj = r["objects"] + (" (mesh)" if r["geometry"] == "mesh" else "")
        bp = "?" if r["blueprint_mm"] is None else f"{r['blueprint_mm']:g}"
        lines.append(f"| {r['label']} | {r['paper_no']} | {r['part']} | {obj} | {bp} | "
                     f"{fmt(r['model_mm'], 3)} | {fmt(r['error_mm'], 3)} | {fmt(r['ratio'], 3)} | "
                     f"{match} | {note} |")
    s = summary
    lines += ["", "## Summary", "",
              f"- match (|error| < {s['match_tol_mm']} mm): {s['n_match']}/{s['n_measured']}",
              f"- deviating: {s['n_deviate']} (larger {s['n_larger']}, smaller {s['n_smaller']})",
              f"- MAE (all): {fmt(s['mae_all_mm'], 2)} mm",
              f"- MAE (deviating only): {fmt(s['mae_deviate_mm'], 2)} mm",
              f"- max |error|: {fmt(s['max_abs_error_mm'], 2)} mm",
              f"- ratio (signed) mean {fmt(s['ratio_mean'], 4)}, "
              f"range {fmt(s['ratio_min'], 3)}–{fmt(s['ratio_max'], 3)}",
              f"- mean |relative error|: {fmt(s['mean_abs_rel_error_pct'], 2)} %"]
    if s["n_no_blueprint"]:
        lines.append(f"- **{s['n_no_blueprint']} row(s) have no blueprint value** (not in summary)")
    n_fail = s["n_rows"] - s["n_measured"] - s["n_no_blueprint"]
    if n_fail:
        lines.append(f"- **{n_fail} row(s) could not be measured** (see note)")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_tex(out_dir / "table.tex", table, summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3].strip())
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--blend", default=None,
                    help="A key of blends in the config, or a path to a .blend (default: default_blend)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Output directory (default: config.output_dir/{blend name})")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    key = args.blend or cfg["default_blend"]
    if key in cfg["blends"]:
        blend, name = Path(cfg["blends"][key]), key
    else:
        blend, name = Path(key), Path(key).stem
    if not blend.exists():
        sys.exit(f"blend not found: {blend}")
    out_dir = args.output_dir or (REPO / cfg["output_dir"] / name)
    tol = float(cfg["match_tol_mm"])

    rows = resolve_rows(cfg["rows"], name)
    with tempfile.TemporaryDirectory() as tmp:
        measured = run_blender(cfg["blender_bin"], blend, rows, Path(tmp))

    table = build_table(rows, measured, tol)
    summary = summarize(table, tol)
    summary.update(blend=str(blend), blend_md5=md5sum(blend),
                   blender_version=measured["blender_version"])
    write_outputs(out_dir, table, summary)
    (out_dir / "measured_raw.json").write_text(json.dumps(measured, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print((out_dir / "table.md").read_text(encoding="utf-8"))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
