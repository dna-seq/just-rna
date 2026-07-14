# just-rna

Type-safe transcriptomic aging clocks for Python. The first implementation is
the Gladyshev Lab tAge clock.

## Design

- Public inputs and outputs are immutable typed values.
- Scientific transforms form a functional core: they take explicit inputs,
  return new values, and do not read files, download assets, or mutate data.
- File, network, `.env`, and model-loading effects are kept at explicit API
  boundaries.
- Polars is used for expression matrices and derived tabular views.
- Paths are always represented by `pathlib.Path`.

## Predict with tAge

```python
from pathlib import Path

from just_rna import ExpressionScale
from just_rna import GeneIdType
from just_rna import Species
from just_rna import predict
from just_rna import read_expression
from just_rna.clocks.tage import TAgeConfig

dataset = read_expression(
    Path("counts.tsv"),
    metadata=Path("metadata.csv"),
    scale=ExpressionScale.RAW_COUNTS,
    gene_id_type=GeneIdType.SYMBOL,
)
config = TAgeConfig(
    species=Species.MOUSE,
    control_group_column="age_group",
    control_group_label="young",
)
result = predict(dataset, species=Species.MOUSE, config=config)

sample = result.one()
print(sample.prediction, sample.unit)
print(result.by_sample())
print(result.to_polars())
```

The default model is
`EN_Chronoage_Multispecies_Multitissue_scaleddiff`. Its pinned Zenodo asset is
downloaded, checksum-verified, and cached on first use.

tAge accepts raw counts and explicitly declared normalized counts. For raw
counts it runs RLE normalization. For already-normalized counts it skips RLE
and returns a structured warning because such predictions may not be directly
comparable with the published raw-count pipeline. TPM, CPM, and FPKM are
rejected rather than silently treated as counts.

## Salmon and transcript quantifiers

Use Salmon's `NumReads`, not `TPM`. `NumReads` are fractional estimated counts,
so just-rna types them as `ExpressionScale.ESTIMATED_COUNTS` and applies RLE
normalization. Transcript-level quantifications must first be summed to genes
using a transcript-to-gene table from the same GTF annotation used to build the
Salmon index.

The mapping can be CSV/TSV or a Polars frame and must contain `transcript` and
`gene` columns:

```python
from pathlib import Path

from just_rna import GeneIdType
from just_rna import Species
from just_rna import predict
from just_rna import read_salmon
from just_rna.clocks.tage import TAgeConfig

dataset = read_salmon(
    {
        "mouse_6m_1": Path("data/input/salmon/mouse_6m_1"),
        "mouse_24m_1": Path("data/input/salmon/mouse_24m_1"),
    },
    Path("data/input/annotation/transcript_to_gene.tsv"),
    gene_id_type=GeneIdType.ENSEMBL,
    metadata=Path("data/input/salmon/metadata.csv"),
)

config = TAgeConfig(
    species=Species.MOUSE,
    control_group_column="age_group",
    control_group_label="young",
)
result = predict(dataset, species=Species.MOUSE, config=config)
result.write_csv(Path("data/output/salmon_tage_predictions.csv"))
```

Each sample path may point to its Salmon directory or directly to `quant.sf`.
Ensembl version suffixes such as `.12` are removed by default. For Kallisto or
another compatible quantifier, use `read_transcript_quantifications(...)` and
specify its transcript-ID and estimated-count column names.

The default `scaled_diff` model requires a real control group. The metadata CSV
must contain `sample` plus the configured control column; its sample IDs must
match the quantification mapping keys.

## Environment and cache

`python-dotenv` is loaded only at effectful boundaries, never when
`just_rna` is imported. Put this in `.env` to override the asset cache:

```dotenv
JUST_RNA_CACHE_DIR=.cache/just-rna
```

Python callers can load an explicit file with
`load_settings(Path("settings.env"))`. CLI commands accept `--dotenv`.

## CLI

```bash
uv run predict counts.tsv \
  --metadata metadata.csv \
  --species mouse \
  --gene-id Gene.Symbol \
  --output predictions.parquet

uv run models list
uv run geo GSE225576 --destination data/input/geo
uv run age
uv run age --csv
```

The standalone command defaults to `data/input/mouse-aging-atlas` for downloaded
matrices and `data/output/mouse_aging_atlas_tage_predictions.parquet` for predictions.
Both generated-data directories are gitignored. Use `--input-directory` and
`--output` to override them. The atlas command also writes a metrics CSV next
to the predictions.

### Run real test predictions

```bash
uv run age --csv
```

This downloads the real 128-sample mouse aging atlas and the real tAge model,
then writes prediction and evaluation CSVs under `data/output/`. To run the
same workflow as an integration assertion:

```bash
uv run pytest tests/test_integration_gse225576.py
```

## Evaluation

`ClockResult.to_polars()` adds `ground_truth` and `ground_truth_unit` whenever
sample metadata contains chronological age. Evaluate any compatible frame:

```python
from pathlib import Path

from just_rna.evaluation import evaluate_predictions

report = evaluate_predictions(
    result.to_polars(),
    group_by=("tissue",),
)
print(report.overall.mae, report.overall.rmse, report.overall.r_squared)
report.write_csv(Path("data/output/metrics.csv"))
```

Metrics include MAE, RMSE, median absolute error, mean error (bias), R²,
Pearson correlation, and Spearman correlation. For the atlas workflow,
`ground_truth` is months since the 6-month control baseline; raw `age_months`
remains in the prediction table.

## GSE225576 integration dataset

The curated adapter downloads all eight mouse tissues and verifies pinned file
sizes and MD5 checksums. GEO metadata provides tissue and age for 112 samples.
The 16 aorta records have an empty GEO `age` characteristic, so their ages are
derived from GEO's `sample_name` metadata; `age_source` records this distinction.
The published matrices contain fractional normalized counts, not raw integer
counts, and are typed as `ExpressionScale.NORMALIZED_COUNTS`. The workflow runs
each tissue independently and uses its 6-month samples as the required
`scaled_diff` control group.

## Development

```bash
uv sync
uv run pytest                       # includes real downloads and predictions
uv run pytest -m "not integration"  # fast numerical/unit checks only
uv run pytest -m integration
uv run ruff check .
uv run pyright
```

Integration tests make real requests, download real public data and models,
run tAge without mocks, and save results under pytest's temporary directory.
