# Contributing to mrv-lib

Issues and pull requests are welcome.

## Requirements

- Python 3.9+
- Install dev dependencies: `pip install -e ".[dev]"`

## Running tests

```bash
pytest tests/ -q
```

All pull requests must pass the full test suite before review.

## Code style

```bash
ruff check src/ tests/
```

Configuration is in `pyproject.toml` (`[tool.ruff]`).

## Contributor License Agreement

mrv-lib is dual-licensed (AGPL-3.0-or-later plus a commercial license). Before
your first contribution can be merged you must accept the
[Contributor License Agreement](CLA.md). This grants the right to relicense
your contribution under the commercial license; you keep your copyright. A DCO
sign-off alone is not sufficient for a dual-licensed project.

Add this line to the pull request description (substitute your details):

```text
mrv-lib-CLA-1.0 accepted by: Full Name <email>
```

Contributions on behalf of an organization require a corporate CLA; contact
kai.zheng@mrv-lib.org first.

## Submitting a pull request

1. Fork the repository and create a branch from `main`.
2. Add tests for any new public API.
3. Ensure `pytest tests/ -q` exits 0.
4. Accept the CLA in the PR description (see above).
5. Open a pull request with a short description of the change and which finding or
   feature it addresses.

## Governance scope

mrv-lib is a pure model risk validation library. Contributions that introduce
alpha-generation claims, trading signals, or fabricated regulatory citations
will not be accepted. See `README.md` for the honest-scope statement.

## Contact

For consulting and enterprise enquiries: kai.zheng@mrv-lib.org
