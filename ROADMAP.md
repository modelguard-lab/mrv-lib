# mrv-lib Roadmap

## Current -- Released

**Status:** Production-ready for SR 26-2 / OCC Bulletin 2026-13 model validation workflows.
SR 26-2 / OCC Bulletin 2026-13 (effective 2026-04-17) supersedes SR 11-7 (Federal Reserve, 2011).

Key capabilities shipped:

- Representation Invariance validator (Paper 1): `validate_rep`, `rep_invariance_validator`
  (CLI: `mrv run config.yaml rep`)
- Resolution Invariance validator (Paper 2): `validate_res`, `res_invariance_validator`
  (labels-first via the Python API; fit your own models at each frequency)
- Business impact function (`impact_fn`) and disagreement attribution (LOO / frequency-pair / temporal)
- Specification-invariance report generation (`report()`: result JSON to PDF via LaTeX), covering
  both the representation and resolution tests
- GitHub Actions CI on Linux (Python 3.9 / 3.10 / 3.12)

## Planned

- Sphinx documentation site.
- `examples` extras group (`pip install mrv-lib[examples]`).
- Structured logging adapter for enterprise SIEM integration.
- REST API wrapper for production ML monitoring deployments.
- Zenodo DOI and formal academic citation support.
