"""Bundled contest data shipped with the package.

This subpackage exists so :mod:`importlib.resources` can locate the
CSV files distributed inside the wheel:

- ``test_compounds.csv`` — the canonical list of held-out compounds
  contestants must predict. Keyed on ``user_compound_id`` (numeric
  LIMS ID as a string) under the ``compound`` column; the original
  chemistry name is preserved as ``compound_name`` for display.
- ``gene_filter.csv`` — the canonical gene set the leaderboard server
  scores on (rebuilt from the combined VCPI 1/2/3 training release
  via ``build_gene_filter(min_mean_cpm=1.0)``).

The per-compound Mejia weight matrix (``weights.parquet``, distributed
via a GitHub Release asset rather than bundled in the wheel) shares the
``user_compound_id`` column key.
"""
