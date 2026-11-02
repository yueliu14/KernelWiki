# Kernel source-bundle format

The [KernelWiki API](README.md) stores complete implementations that can be called, tested and
benchmarked. A local collection uses `<collection-root>/<operator>/<backend>/<implementation>/`.
Supply your own source collection or use a remote store. Textual explanations and lessons live
under [knowledge](../knowledge/README.md).

## Required bundle contents

- Actual kernel source and an explicit callable entry point.
- Runtime configuration and declared dependencies, including version requirements.
- A numerical reference and correctness test program covering relevant boundary cases.
- A benchmark program with an identified baseline and timing protocol.
- `README.md` with inputs, outputs, layouts, side effects, limitations and run instructions.
- `metadata.yaml`, provenance and license information.

Files are explicitly declared in metadata. Cache files, environment files and undeclared run outputs are not uploaded. Runtime dependencies may be external libraries; an implementation must state them. A wrapper that only points to an unavailable private source tree is not a complete kernel implementation.

## Metadata

Each source bundle declares the following metadata in `metadata.yaml`.

| Field | Meaning |
| --- | --- |
| `schema_version` | Currently `1` |
| `implementation_id` | Stable `operator/backend/name`, matching those three fields |
| `status` | `reference`, `validated` or `best_known` |
| `description`, `license`, `provenance` | Purpose, applicable license, original source/run and adaptations |
| `entry_point` | Declared source file and callable symbol |
| `dependencies` | Runtime packages and version requirements |
| `files` | Lists named `source`, `tests`, `benchmarks`, optional `config` and `support` |
| `commands` | Argument lists for running `test` and `benchmark` from the bundle directory |
| `supports` | GPU architectures, dtypes, shape constraints and human-readable layout/semantic restrictions |
| `validation` | `not_run` or `passed`; passed results include `source_digest` and evidence references |
| `benchmarks` | Distinct `measured` and `upstream_reported` evidence |
| `best_for` | Exact measured cases for a `best_known` recommendation |

Shape constraints support `min`, `max`, `multiple_of` and `values` for each dimension. A query checks the dimensions it supplies; an unlisted dimension does not match. Layout, aliasing and semantic restrictions in the README still need to be checked by the caller.

## Source identity and evidence

`source_digest` hashes the declared source files, runtime configuration, entry point and dependency declarations. Correctness and measured-performance records must identify that digest. A source/configuration/dependency change invalidates their old claims. `revision` hashes every declared bundle file, including metadata, tests and documentation. Changing evidence can therefore produce a new bundle revision without changing the kernel source digest.

A measured benchmark records `gfx`, `dtype`, every shape dimension, `bench_key`, baseline identity, `latency_us`, `baseline_latency_us`, correctness, environment, `source_digest` and evidence references. A benchmark key must identify the same baseline and timing protocol in the same environment. Uploading source does not run it or certify these claims.

Evidence may use an external HTTP/KB/object-store/run reference. A local evidence file must be listed in `files.support` so it travels with the bundle. Runtime requirements/build configuration should be declared in `files.config` so changes affect the source digest.

`best_known` requires passed validation and at least one `best_for` case with a measured improvement. Each case specifies `gfx`, `dtype`, full `shape` and `bench_key`. A recommendation applies to those cases and the named comparison set. Upstream ratios alone cannot promote an extracted implementation to `best_known`.

## Curating and retrieving code

Use Git to review and evolve your source collection. The [KernelWiki API](README.md) can add or fetch local entries, and upload/download complete candidates through the existing GEAK KB service. Local `put()` is idempotent for identical content and refuses to overwrite a different existing implementation. Remote revisions coexist and can be retrieved by their full digest.

Link a curated entry from its operator/backend knowledge page. Preserve useful lessons from unsuccessful runs as scoped corpus conclusions, with public evidence summaries; keep raw code/measurement run histories in a private store.
