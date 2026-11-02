# Kernel source API

`KernelClient` provides `put`, `search` and `fetch` for complete source bundles. Use your own local
source collection or the existing GEAK KB service. See the [source-bundle format](source_bundles.md).

## Install and choose a store

```bash
python -m pip install -e .
```

```python
from kernelwiki import KernelClient, SourceBundle

local = KernelClient.local("/path/to/your/source-collection")
```

For remote storage:

```bash
export GEAK_KB_URL="https://your-existing-kb-service"
# Use your existing GEAK_KB_STORE_TOKEN / KB_STORE_TOKEN credentials.
```

```python
remote = KernelClient.from_env()
```

URL precedence is `GEAK_KB_URL`, `GEAK_KB_STORE_URL`, then `KB_STORE_URL`; blank values are ignored. Token precedence is `GEAK_KB_STORE_TOKEN`, then `KB_STORE_TOKEN`. `GEAK_KB_READ_ONLY=1` blocks `put()` on both stores. Search and fetch remain available.

## Inspect and put

```python
path = "/path/to/your/source-bundle"
bundle = SourceBundle.load(path)
print(bundle.implementation_id, bundle.source_digest, bundle.revision)

# Run this only when you intend to publish the source to your configured store.
reference = remote.put(path)
```

`put(source_bundle, metadata=None)` uses the directory's `metadata.yaml` by default. An explicit metadata object replaces it in the stored snapshot without changing the input directory. Only declared files are included. Validation, a temporary snapshot and digest checks complete before upload.

The returned object identifies the implementation, source digest and immutable bundle revision. A retry of the same bundle reuses its remote session. Local `put()` adds `operator/backend/name/` under the configured root, accepts identical existing content and rejects replacement of different content. Use a normal Git edit/review to evolve curated local source.

Remote source revisions are stored under `geak:kernel_code:<operator>:<backend>:<name>`. The operator page `geak:kernel_code:<operator>` contains discovery records referencing those revisions. Files have one home on the implementation page. The service's scalar-pagination mechanism uses a constant schema field to enumerate source records; it is not a fabricated performance score. This also makes unmeasured reference implementations discoverable.

Publication is not a transaction across the implementation and operator pages. A failure reports completed identities and the source revision to retry. It does not silently report a partial publication as success. No champion promotion occurs during upload.

## Search applicability

```python
result = local.search(
    "rmsnorm", backend="triton", gfx="gfx950", dtype="bf16",
    shape={"M": 16384, "N": 128}, top_k=5,
)
for item in result["implementations"]:
    print(item["implementation_id"], item["revision"], item["metadata"]["status"])
```

`operator` is required. Backend, architecture, dtype and shape filter the declared applicability. Without a benchmark context, results are ordered by implementation ID and revision, with `comparable=False`. A source reference is useful even when it has no measured speedup.

To compare measured candidates, supply `gfx`, `dtype`, a complete shape and `bench_key`. Search then keeps matching measurements and orders by latency relative to their common baseline. Reusing a key across incompatible baselines, environments or recorded timing protocols is rejected. `best_only=True` additionally requires an explicit `best_for` recommendation for that exact case.

`matched_count` counts matches within the scan. Remote `scan` limits examined sessions per operator (default 200), and `scan_truncated=True` means the result may be incomplete. An absent remote operator page is a normal miss. A missing local collection, transport failure or malformed source record raises an error rather than pretending there are no candidates.

## Fetch an exact implementation

```python
item = result["implementations"][0]
directory = local.fetch(
    item["implementation_id"], revision=item["revision"],
    destination="downloaded_rmsnorm",
)
```

The destination must not already exist. It receives `kernel.py`, `entry.py`, tests, configuration, metadata and the other declared files directly, without a patch-application step. Downloads are staged and verified against the recorded manifest, metadata, bundle revision and source digest before publishing the destination. Fetch does not import, execute or benchmark the kernel.

Pin `revision` for reproducibility. A local checkout only contains its checked-out revision. Without a revision, remote fetch uses the service-selected record for that implementation, which can be its champion or most recently updated candidate; it does not imply that the result is fastest for the caller's workload.

## Migration from version 0.1

The `kernel_patch.PatchClient` interface has been removed. New source records use `carrier: kernel_source` and a separate `geak:kernel_code:...` namespace. Existing patch records in the service remain historical artifacts; search does not relabel or return them as source bundles. A historical winning patch must be materialized into complete source and supplied with its entry point, dependencies and evidence before publication through this API.

The protocol SDK in `_store/` is adapted from GEAK. Core identity, store, curation and read-only regression tests are retained. Source-bundle tests cover local/remote round trips, applicability, exact revisions, evidence consistency, corrupt downloads, partial publication and idempotent retries. No test writes to the live shared KB service.
