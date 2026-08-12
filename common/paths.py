"""
Central path configuration.

Every data / checkpoint / output location used anywhere in this repository is read from an
environment variable through `P()`. Nothing is hard-coded to a particular machine.

Resolution order for each variable:

  1. the process environment  (`export EVAL_ROOT=/my/eval`)
  2. `env/paths.local.sh` at the repository root, if it exists

`env/paths.local.sh` is **gitignored** — it is where the authors' cluster paths live, and where you
put yours. Copy the template to create it:

    cp env/paths.example.sh env/paths.local.sh
    $EDITOR env/paths.local.sh

The same file is `source`-able from bash, so the `.sh` run scripts and the Python scripts read
exactly the same values. See `docs/PATHS.md` for what each variable means.

Usage:

    from paths import P

    EV  = P("EVAL_ROOT")                                    # the root itself
    IDD = P("RISK_ROOT", "csv_train_valid_test_individual_id_disease")   # a path under it

`P()` raises a clear error naming the missing variable rather than failing later with a confusing
"file not found", so a misconfigured environment is obvious immediately.
"""
import os
import re
from pathlib import Path

__all__ = ["P", "VARS", "resolve", "missing"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_FILE = _REPO_ROOT / "env" / "paths.local.sh"

# variable -> human-readable description (kept in sync with env/paths.example.sh and docs/PATHS.md)
VARS = {
    "UKB_MRI_DIR":     "UK Biobank cardiac MRI, per subject: vst_2ch.npy / vst_4ch.npy / vst_sa.npy",
    "UKB_PHENO_DIR":   "UKB phenotype CSVs: MRI_train/, MRI_valid_new/, MRI_test_new/, one <outcome>.csv each",
    "UKB_ECG_ROOT":    "UKB ECG: ECG_manifest*/, ECG_label*/, ECG_label_surv/<outcome>/, stage1/",
    "MRI_SPLITS":      "subject-ID lists for MAE pretraining: train/ and val/",
    "CHS_ECG_ROOT":    "CHS ECG: ECG_manifest/, ECG_label_surv/<outcome>/",
    "MESA_ECG_ROOT":   "MESA ECG: ECG_manifest/, ECG_label_surv/<outcome>/",
    "CHS_MESA_ROOT":   "parent of the CHS/MESA cohort directories",
    "RISK_ROOT":       "risk-score inputs and outputs: CHARGE-PREVENT/, computed/, csv_HR/, csv_train_valid_test_individual_id_disease/",
    "MESA_TABLES":     "MESA measured-CMR and disease tables (MESA_CMR_features.csv, MESA_disease.csv)",
    "FEWSHOT_RESULTS": "external few-shot result tables",
    "EXT_RESULTS":     "external aggregate result tables",
    "CKPT_ROOT":       "all model checkpoints (MAE, stage-1, downstream)",
    "LOG_DIR":         "training log directory",
    "ECG_CKPT":        "the ECG-FM backbone file itself (see weights/README.md)",
    "EVAL_ROOT":       "prediction/eval outputs consumed by the figure scripts",
    "ECGFOUNDER_REPO": "checkout of the ECGFounder repository (baselines/ only)",
}

_local_cache = None
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


def _load_local():
    """Parse env/paths.local.sh into a dict. Tolerates `export`, quotes, comments, blank lines."""
    global _local_cache
    if _local_cache is not None:
        return _local_cache
    _local_cache = {}
    if _LOCAL_FILE.exists():
        for line in _LOCAL_FILE.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = _LINE.match(line)
            if not m:
                continue
            val = m.group(2)
            if val and val[0] == val[-1] and val[0] in "\"'" and len(val) > 1:
                val = val[1:-1]
            # expand references to variables defined earlier in the same file
            val = re.sub(r"\$\{?([A-Z_][A-Z0-9_]*)\}?",
                         lambda mm: _local_cache.get(mm.group(1), os.environ.get(mm.group(1), "")),
                         val)
            if val:
                _local_cache[m.group(1)] = val
    return _local_cache


def resolve(name):
    """Return the configured value for `name`, or None if it is not set anywhere."""
    return os.environ.get(name) or _load_local().get(name)


def missing(names=None):
    """Return the subset of `names` (default: all VARS) that are not configured."""
    return [n for n in (names or VARS) if not resolve(n)]


def P(name, *parts):
    """
    Resolve path variable `name`, optionally joined with `parts`.

    Raises KeyError with an actionable message if the variable is not configured.
    """
    base = resolve(name)
    if not base:
        hint = VARS.get(name, "")
        raise KeyError(
            f"Path variable {name!r} is not set"
            + (f" ({hint})" if hint else "")
            + f".\n  Set it in the environment, or add it to {_LOCAL_FILE}."
            + f"\n  Start from the template: cp env/paths.example.sh env/paths.local.sh"
            + f"\n  See docs/PATHS.md for what each variable means."
        )
    return os.path.join(base, *parts) if parts else base


if __name__ == "__main__":  # `python common/paths.py` prints the current configuration
    width = max(len(v) for v in VARS)
    unset = []
    for var, desc in VARS.items():
        val = resolve(var)
        if val:
            print(f"  {var:<{width}}  {val}")
        else:
            unset.append(var)
            print(f"  {var:<{width}}  <UNSET>   -- {desc}")
    print(f"\nlocal defaults file: {_LOCAL_FILE} ({'found' if _LOCAL_FILE.exists() else 'ABSENT'})")
    print(f"{len(VARS) - len(unset)}/{len(VARS)} configured")
