# vcpi-prediction-contest

Scoring code and held-out test query set for the **VCPI compound
expression-prediction contest**.

Version: 0.1.

---

## The task

Given a compound's chemical structure (SMILES + the chemistry table
shipped with the training data), predict the **per-gene average
expression** that compound induces in THP-1 cells after 24 hours at
10 µM. For every held-out test compound `c` and every gene `g` in the
scored gene set, predict a single number: `expression(c, g)`.
"Expression" here is the **mean of** `log2(CPM + 1)` **across replicates of**
`(c, g)`, where CPM is `1e6 * count_g / total_counts_sample`.

## Setup

To run the examples below, you'll need to install this package and
[vcpi-client](https://github.com/virtualcell-vcpi/vcpi-client). See the client's
README for information on setting the token. You can install them in a
[uv](https://docs.astral.sh/uv/) project as follows:

```shell
# Make a folder with a pyproject.toml and `cd` into it
uv init --bare my-vcpi-repo && cd my-vcpi-repo
# Install the two packages from Github
uv add git+https://github.com/virtualcell-vcpi/vcpi-prediction-contest-2026.git
uv add git+https://github.com/virtualcell-vcpi/vcpi-client.git
```

## Scoring metric

- **wMSE** — weighted per-gene-weighted mean squared error between
predicted and true expression vectors. Lower is better, 0 is
perfect.

### Weights

Per-compound, per-gene weights are used to reward algorithms that more accurately predict the expression of genes that are different compared to the average of the rest of the training data for a given compound. They are calculated following [Meija’s preprint](https://arxiv.org/abs/2506.22641). Briefly, we calculate the t-statistic vs the rest of the compounds for each gene for each compound, minmax it, square it and renormalize so the column sums to 1. We provide these for each (gene, compound) pair.

### wMSE results on simple models

Here we show the `wMSE` metric results on a variety of simple models on the training data, using a validation split of 200 compounds to score the models.

| rank | baseline                         | wMSE mean | what it is                                                          |
| ---- | -------------------------------- | --------- | ------------------------------------------------------------------- |
| 1    | `truth_itself`                   |     0.000 | the held-out truth itself (sanity check; wMSE must be 0)            |
| 2    | `truth_plus_noise_small`         |     0.010 | near-perfect: `truth + N(0, 0.1)` (technical-replicate analog)      |
| 3    | `per_gene_mean_of_train`         |     0.507 | mode-collapse: per-gene mean across the training compounds          |
| 5    | `random_train_compound`          |     0.701 | predict a random training compound's expression vector              |
| 6    | `truth_plus_noise_large`         |     1.003 | `truth + N(0, 1.0)`; expected wMSE ≈ σ² = 1.0                       |
| 7    | `global_constant_at_mean`        |     5.785 | wMSE-optimal global constant: scalar `mean(truth_train)`            |
| 11   | `global_constant_5.0`            |     8.345 | scalar 5.0 everywhere (off-optimal constant)                        |
| 12   | `random_gaussian`                |    10.493 | pure noise: `N(mean, std)` of `truth_train`                         |

## Submission format

A single parquet or CSV file with three columns:

| column                 | dtype  | notes                                                                                                              |
| ---------------------- | ------ | ------------------------------------------------------------------------------------------------------------------ |
| `compound`             | string | the `user_compound_id` (numeric LIMS ID as a string), matching the `compound` column of the bundled `test_compounds.csv` and `metadata.user_compound_id` from `vcpi-client` |
| `gene_id`              | string | must match the bundled `gene_filter.csv`                                                                           |
| `predicted_expression` | float  | non-negative; on the `log2(CPM + 1)` scale                                                                         |

**The leaderboard server scores on the genes in** `gene_filter.csv`**.**
You must provide one row for every `(compound, gene_id)` pair — every
test-set compound times every scored gene — so the submission frame
has exactly `len(test_compounds) × len(gene_filter)` rows.

## Getting the training data

The training set is the union of three VCPI releases — `tvc-bhr-009`,
`tvc-kdl-010`, `tvc-qnu-012` — delivered via the
[vcpi-client](https://github.com/virtualcell-vcpi/vcpi-client) Python
package (install, set `TVC_TOKEN`). The recipe below downloads the
Mejia weight matrix and pulls each release filtered to the contest
condition (THP-1, 24 h, library at 10 µM + DMSO controls), then merges
into three parquet artifacts. Peak RAM ≈ 22 GB; wall-clock ≈ 3 min on a
fast network.

```python
import pandas as pd
import polars as pl
import vcpi
from vcpi_prediction_contest import load_weights_matrix

load_weights_matrix().to_parquet("weights.parquet")  # 375 MB, (12,995 genes x 14,031 compounds)

JOBS = ["tvc-bhr-009", "tvc-kdl-010", "tvc-qnu-012"]
counts_pieces, metadata_pieces, chemistry_pieces = [], [], []
for job in JOBS:
    exp = vcpi.load_experiment(job)
    meta = exp["metadata"].filter(
        (pl.col("cell_line") == "THP-1")
        & (pl.col("timepoint") == "24h")
        & (
            # vcpi stores 10 µM as 10000 nM
            ((pl.col("compound_concentration") == 10_000)
             & (pl.col("compound_concentration_unit") == "nM"))
            | (pl.col("user_compound_id") == "DMSO")
        )
    )
    keep = set(meta["sequenced_id"].cast(pl.Utf8).to_list())
    data = exp["data"].select(
        ["gene_id", *[c for c in exp["data"].columns if c != "gene_id" and c in keep]]
    )
    counts_pieces.append(data.to_pandas().set_index("gene_id"))
    metadata_pieces.append(meta.to_pandas())
    chemistry_pieces.append(exp["chemistry"].to_pandas())
    del exp, meta, data

counts = (
    pd.concat(counts_pieces, axis=1, join="outer")
    .fillna(0)
    .astype("int32")
    .reset_index()
)
metadata = pd.concat(metadata_pieces, ignore_index=True)
chemistry = (
    pd.concat(chemistry_pieces, ignore_index=True)
    .drop_duplicates(subset=["compound"])
    .reset_index(drop=True)
)

counts.to_parquet("train_counts.parquet")        # 1.5 GB, 78,778 genes x 32,500 samples
metadata.to_parquet("train_metadata.parquet")    #   3 MB, 32,500 samples x 26 columns
chemistry.to_parquet("train_chemistry.parquet")  # 1.5 MB, 14,031 unique compounds x 13 columns
```

`metadata.user_compound_id` (numeric LIMS ID as a string, e.g.
`"9251300"`) is the canonical contest join key — same value used in
`W.columns` from `load_weights_matrix()` and in the `compound` column
of `test_compounds.csv` and submissions. To skip the weights download
on a machine that already has a copy, set
`VCPI_WEIGHTS_PATH=/path/to/weights.parquet`.

### Bundled Information

This repository includes
`test_compounds.csv` and `gene_filter.csv`, loadable via
`load_test_compounds()` and `load_gene_filter()` . The `test_compounds.csv` are the hold-out compounds consisting of 6 plates of DRUG-seq data of > 1,000 different compounds, hundreds of which are active to varying degrees. The `gene_filter.csv` file are the genes that are used for scoring.

```python
from vcpi_prediction_contest import (
    gene_filter_path,
    load_gene_filter,
    load_test_compounds,
    test_compounds_path,
)

compounds = load_test_compounds()      # DataFrame
genes = load_gene_filter()             # list[str] of scored gene_ids
test_compounds_path()                  # filesystem Path to test_compounds.csv
gene_filter_path()                     # filesystem Path to gene_filter.csv
```

`test_compounds.csv` has columns:

| column                        | notes                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `compound`                    | `user_compound_id` (numeric LIMS ID as a string). The canonical join key — same value as `metadata.user_compound_id` from `vcpi-client`. |
| `compound_name`               | human-readable chemistry name. Display only — do not use for joins (vcpi-client does not expose chemistry names).                           |
| `compound_concentration`      | always 10                                                                                                                                   |
| `compound_concentration_unit` | always `"uM"`                                                                                                                               |
| `cell_line`                   | always `"THP-1"`                                                                                                                            |
| `timepoint`                   | always `"24h"`                                                                                                                              |
| `smiles`                      | canonical SMILES                                                                                                                            |
| `inchi_key`                   | computed InChIKey                                                                                                                           |

`gene_filter.csv` has a single `gene_id` column listing the Ensembl
IDs of the scored genes. These are just the genes with mean CPM >=1 across the training samples.

### Converting raw counts to expression values

`vcpi-client` returns raw UMI counts per replicate. The leaderboard
scores in `log2(CPM + 1)`-mean space, so convert counts to that before scoring.
There is a `counts_to_expression` function that will do that for you.
representation before training or self-scoring — `counts_to_expression()`.

### Control compounds

`vcpi-client`'s `metadata` table includes several control compounds
alongside the library compounds:

| `compound` / `user_compound_id` | concentration | purpose                 |
| ------------------------------- | ------------- | ----------------------- |
| DMSO                            | 0.15%         | inert vehicle control   |
| Staurosporine                   | 10 uM         | cell-death control      |
| Brefeldin A                     | 10 uM         | transcriptional control |
| Rigosertib                      | 10 uM         | transcriptional control |
| Trichostatin A                  | 10 uM         | transcriptional control |

Filter on `is_control`, `user_compound_id` (human-readable), or `compound` (UUID) to identify them in metadata.

### Fetching the canonical scoring weights

The package ships `fetch_weights()` and `load_weights_matrix()` for
the per-compound Mejia weight matrix the leaderboard server scores
against. The first call streams the file from a public GitHub Release
asset and caches it under `~/.cache/vcpi-prediction-contest/`;
subsequent calls hit the cache directly.

```python
from vcpi_prediction_contest import load_weights_matrix

# (12,995 genes x 14,031 training compounds), float16 on disk;
# numpy / pandas upcast to float32 at use-time.
W = load_weights_matrix()
# W.columns are training-compound `user_compound_id` strings
# (numeric LIMS IDs) — same key as `metadata.user_compound_id`
# from vcpi-client, so:
#
#     train_meta = exp["metadata"].to_pandas()
#     overlap = set(W.columns) & set(train_meta["user_compound_id"])
#
# returns the set of training compounds present in both. No
# chemistry-name <-> id bridge required.
```

The file is ~365 MB on first download, then loads from cache.

For offline / CI / contestants who already have a local copy, set
`$VCPI_WEIGHTS_PATH` to short-circuit the download:

```bash
export VCPI_WEIGHTS_PATH=/path/to/weights.parquet
```

## Library use

In every contest frame — `truth`, `pred`, `test_compounds`, and
`W.columns` of the weights matrix — `compound` is the
`user_compound_id` (numeric LIMS ID as a string).

```python
import pandas as pd
from vcpi_prediction_contest import (
    score_compounds,
    aggregate_leaderboards,
    load_gene_filter,
    load_weights_matrix,
)

truth = pd.read_parquet("my_holdout.parquet")           # compound (user_compound_id), gene_id, expression
pred  = pd.read_parquet("my_predictions.parquet")       # compound (user_compound_id), gene_id, predicted_expression
genes = load_gene_filter()                              # bundled scored gene set
weights = load_weights_matrix()                         # canonical Mejia (n_genes x n_train_compounds), columns are user_compound_id

per_compound = score_compounds(
    truth,
    pred,
    gene_filter=genes,
    weights=weights,
)
print(per_compound.head())
```

If you omit `weights`, the function derives them from the truth
file itself (variance-of-truth weights). If you want to score the same way the leaderboard does at the end, pass in the Meija pre-calculated weights.

### Example Scoring

Run this after the code in "Getting the training data" to score a dummy set of predictions based on the real counts.

```python
import pandas as pd
from numpy.random import default_rng
from vcpi_prediction_contest import (
    aggregate_leaderboards,
    counts_to_expression,
    load_gene_filter,
    score_compounds,
)

metadata = pd.read_parquet('train_metadata.parquet')
# Sample 10 compounds
compounds = (
    metadata.query('~is_control').user_compound_id.drop_duplicates().sample(10, random_state=42)
)
# Read in the counts for those compounds
counts = pd.read_parquet(
    'train_counts.parquet',
    columns=[
        'gene_id',
        *metadata.query('user_compound_id in @compounds')['sequenced_id'].astype(str).tolist(),
    ],
)

# Generate ground truth expression data with a subset of samples
expression = counts_to_expression(counts, metadata, compound_col='user_compound_id')
truth = expression.rename(columns={'user_compound_id': 'compound'})

# Loop over different levels of noise and score predictions
weights = pd.read_parquet('weights.parquet')
genes = load_gene_filter()  # bundled scored gene set
mean = truth['expression'].mean()
std = truth['expression'].std()

noise_factors = [0, 1, 2]  # no noise, noise, 2*noise

for factor in noise_factors:
    pred = truth.rename(columns={'expression': 'predicted_expression'}).copy()
    noise = default_rng().uniform(-std, std, size=len(pred)) * factor if factor > 0 else 0
    pred['predicted_expression'] = pred['predicted_expression'] + noise
    per_compound = score_compounds(truth, pred, gene_filter=genes, weights=weights)
    board = aggregate_leaderboards(per_compound)
    print(f'Noise factor {factor}, score: {board["wmse_mean"]:.4f}')
```
