# lift-shield

**Is a Little Slip Worth It? Certified Fast Lifting for Residual Policies Under a Static-to-Kinetic Friction Drop**
Clement Aidoo, CANLABS (Paper 7). MuJoCo simulation, pre-registered.

- `lift_env.py`: rig, probe, model controller, slip bound (Proposition 1) and grip/time shields
- `validate.py`: rig validation (V1–V4) → `results/validation.json`
- `PROTOCOL.md`: frozen protocol (committed before training) with dated deviations
- `train_es.py`, `train_all.sh`: evolution-strategies training, 5 seeds × {residual, shielded residual}; weights in `weights/`
- `evaluate.py test|confirm`, `analyze.py`, `analyze_confirm.py`: every number in the paper
- `explore_allowance.py`: exploratory slip-allowance analysis (D3)
- `controlled.py`, `controlled_analysis.py`: Addendum A controlled study (19 changes + 4 diagnostics) → `results/controlled/`
- `extended.py`: safety map, bound tightness and failure traces → `results/extended/`; `render.py`: MuJoCo renders
- `figs.py`, `figs_ext.py`, `make_tables.py`, `make_listings.py`, `paper/`: figures, tables and LaTeX source (long form: `paper/main.tex`; first short draft: `paper/short_v1.tex`)

Reproduce: `python validate.py && ./train_all.sh && python evaluate.py test && python analyze.py test && python evaluate.py confirm && python analyze_confirm.py && python explore_allowance.py && python figs.py`

License: MIT.
