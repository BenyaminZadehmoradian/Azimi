"""Compare PHAX / baseline result ZIPs (or extracted folders) on the quantities that test E0-E3.

usage: python compare_runs.py RUN1 [RUN2 ...]   (each a results .zip or a folder with summary.json)
Writes comparison.csv and prints a table. Test numbers are reported, never used for selection.
"""
import io, json, sys, zipfile
from pathlib import Path
import pandas as pd


def load(path):
    p = Path(path); files = {}
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                files[Path(n).name] = z.read(n)
    else:
        for f in p.iterdir():
            if f.is_file(): files[f.name] = f.read_bytes()
    return files


def csv(files, name):
    return pd.read_csv(io.BytesIO(files[name])) if name in files else None


def row_for(path):
    f = load(path); s = json.loads(f["summary.json"])
    r = {"run": Path(path).stem, "artifact": s.get("artifact"),
         "experiment": s.get("experiment", s.get("phases_mode", "baseline")),
         "val_acc": s["validation"]["accuracy"], "test_acc": s["test"]["accuracy"],
         "train_full_acc": (s.get("train_full") or {}).get("accuracy"),
         "optimizer_epochs": s.get("optimizer_epochs"),
         "selected_path_min": (s.get("selected_path_train_sec") or 0) / 60 or None}
    t = s.get("timing", {})
    r["all_training_min"] = (s.get("actual_all_training_sec_including_local") or s.get("actual_training_sec")
                             or t.get("train_loop_sec", 0)) / 60
    r["total_wall_min"] = t.get("total_wall_sec", 0) / 60
    r["compile_warmup_min"] = t.get("compile_warmup_sec", 0) / 60
    ph = csv(f, "phase_summary.csv")
    if ph is not None and "phase" in ph and (ph.phase == "v12_endpoint2_float_refine").any():
        # PHAX: hybrid accuracy at the end of endpoint2 (before any S3+ replacement)
        for name, key in (("v12_endpoint2_float_refine", "e2_float_val"), ("v12_endpoint2_shift_qat", "e2_after_qat_val")):
            v = ph.loc[ph.phase == name, "validation"].iloc[0]
            r[key] = eval(v)["accuracy"] if isinstance(v, str) else v["accuracy"]
        r["final_minus_e2_pp"] = 100 * (r["val_acc"] - r["e2_after_qat_val"])
    led = csv(f, "information_transition_ledger.csv")
    if led is not None:
        s10 = led[led.event == "replace_internal_digital_output_by_physical_S10"]
        if len(s10): r["S10_transition_pp"] = float(s10.delta_pp.iloc[0])
        r["sum_negative_transitions_pp"] = float(led.delta_pp[led.delta_pp < 0].sum())
    bn = csv(f, "bn_final_per_channel.csv")
    if bn is not None:
        r["bn_bias_changed_fraction"] = float((bn.bias_final != bn.bias_initial).mean())
    loc = csv(f, "local_timing.csv")
    if loc is not None:
        r["local_to_best_min"] = loc.train_to_best_sec.sum() / 60
    return r


if __name__ == "__main__":
    rows = [row_for(p) for p in sys.argv[1:]]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(df.round(4).T.to_string())
    df.to_csv("comparison.csv", index=False)
