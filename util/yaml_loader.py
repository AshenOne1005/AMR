# --------------------------------------------------------
# Lightweight YAML config with `_base_` inheritance + deep merge.
# --------------------------------------------------------

import copy
import os

import yaml


def _deep_merge(base, override):
    """Recursively merge ``override`` into ``base``.

    Nested dicts are merged field-by-field; non-dict values in ``override``
    overwrite ``base``. Returns a new dict; inputs are not mutated.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_base(base_path, here):
    """Resolve a `_base_` path relative to the file, then cwd, then repo root."""
    candidates = [
        os.path.join(here, base_path),          # relative to current file's dir
        base_path,                              # relative to cwd
        os.path.join(_repo_root(), base_path),  # relative to repo root
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f'_base_ not found: {base_path!r} (from {here!r})')


def _repo_root():
    """Best-effort repo root (this file lives at <root>/util/yaml_loader.py)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml_config(path, cfg_name=None):
    """Load a YAML config, resolving `_base_` inheritance with deep merge.

    A top-level `_base_: <relative/path.yaml>` key loads the referenced file
    first, then the current file's keys are deep-merged on top of it (so a
    child only needs to list the fields it overrides). `_base_` itself is
    popped before returning, so the result contains only real argparse keys.

    If ``cfg_name`` is given, the file is treated as a collection of *named
    configs*: each top-level key (besides `_base_`) names a dataset entry whose
    dict is deep-merged onto the file-level `_base_`. Other named entries are
    ignored (only the selected one is applied). This lets one file hold all
    datasets for a protocol, selected via ``--cfg-name <dataset>``. Without
    ``cfg_name`` the file is loaded flat (backward compatible).

    Returns a flat dict suitable for ``parser.set_defaults(**result)``.
    """
    with open(path, 'r') as f:
        raw = yaml.load(f, yaml.FullLoader) or {}
    here = os.path.dirname(os.path.abspath(path))
    base_path = raw.pop('_base_', None)

    if cfg_name is not None:
        if cfg_name not in raw:
            raise KeyError(
                f'cfg-name {cfg_name!r} not found in {path!r}; '
                f'available: {list(raw.keys())}')
        entry = raw.pop(cfg_name)
        base_cfg = {}
        if base_path:
            base_cfg = load_yaml_config(_resolve_base(base_path, here))
        # only the selected entry is merged onto the base; the other named
        # entries left in `raw` are not applied.
        return _deep_merge(base_cfg, entry)

    # backward-compat flat load
    if base_path:
        base_cfg = load_yaml_config(_resolve_base(base_path, here))
        return _deep_merge(base_cfg, raw)
    return raw
