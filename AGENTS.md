# just-rna Engineering Rules

## Architecture

- Prefer a functional core with a thin imperative shell.
- Scientific transforms must take explicit inputs, return new values, and avoid I/O or mutation.
- Keep network, filesystem, environment, model loading, and serialization at explicit boundaries.
- Do not perform network, disk, `.env`, registration, or heavy dependency work during import.
- Use immutable public models and values where practical.
- Remove unused legacy APIs instead of maintaining compatibility layers without a requirement.

## Python

- Support Python 3.14 and use complete type hints.
- Run Pyright in strict mode; do not weaken checks to hide errors.
- Use `pathlib.Path` for paths and `python-dotenv` only at explicit configuration boundaries.
- Use Pydantic v2, Polars, and Typer. Do not use Pydantic v1 or pandas in public APIs.
- Prefer enums, value objects, generics, and discriminated unions over untyped strings or dictionaries.
- Use absolute imports; never use relative imports.
- Never hardcode the package version in `__init__.py`; use `importlib.metadata`.
- Avoid broad or nested `try`/`except`; catch failures only at justified effect boundaries.

## Dependencies and commands

- This is a uv project: use `uv sync`, `uv add`, and `uv run`; never use `uv pip install`.
- Routine workflows need standalone, one-word commands such as `age`, `predict`, `geo`, and `models`.
- Do not prefix standalone commands with `rna-`, `just-rna`, or another redundant project name.
- Never expose accessions, internal IDs, dataset codes, or implementation details in command names.
- Keep the umbrella `just-rna` CLI only as an optional package-level interface.
- Canonical workflows should run without required path arguments by using sensible `data/` defaults.
- README commands must exactly match `project.scripts`; execute the documented command before claiming it works.
- Put generated inputs under `data/input/` and outputs under `data/output/`; keep generated data gitignored.
- Do not put placeholder paths such as `/my/custom/path/` in executable code.

## Scientific tests

- Integration tests use real requests, real public data, and real clock models; do not mock them.
- Integration tests run in the default `uv run pytest` invocation.
- Assert scientific behavior: pinned reference predictions, numerical parity, age signal, and saved/in-memory equality.
- Schema or field-presence assertions alone are insufficient.
- Synthetic fixtures are acceptable only for isolated numerical transforms, validation, and failure cases.
- Save generated integration outputs under pytest temporary paths, not in Git.

## Verification

- Run `uv run pytest`, `uv run pyright`, and `uv run ruff check .` after substantive changes.
- Treat normalized counts, raw counts, TPM, and other scales as distinct typed inputs; never silently reinterpret them.
