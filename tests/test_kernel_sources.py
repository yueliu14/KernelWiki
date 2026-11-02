import copy
from pathlib import Path

import pytest
import yaml

from kernelwiki import BundleError, KernelClient, SourceBundle, KBStoreError
from kernelwiki.client import INDEX_METRIC, implementation_address, operator_address


class MemoryService:
    """Record/file protocol fake; each identity has an independent manifest."""

    def __init__(self):
        self.records = {}
        self.files = {}
        self.writes = []
        self.fail_id = None
        self.corrupt_download = False

    def put_files(self, cid, sid, items):
        self.writes.append(("files", cid, sid))
        self.files[cid, sid] = {rel: Path(path).read_bytes() for rel, path, *_ in items}
        return {rel: "kb://test/" + rel for rel in self.files[cid, sid]}

    def put_knowledge(self, cid, document, *, session_id, mode):
        if cid == self.fail_id:
            raise KBStoreError("index unavailable")
        self.writes.append(("knowledge", cid, session_id))
        self.records[cid, session_id] = copy.deepcopy(document)

    def get_top_sessions(self, cid, *, metric, limit, offset):
        assert metric == INDEX_METRIC
        rows = [{"session_id": sid} for (key, sid), doc in self.records.items()
                if key == cid and metric in doc]
        rows.sort(key=lambda x: x["session_id"])
        return {"sessions": rows[offset:offset + limit]}

    def get_session(self, cid, sid):
        doc = self.records.get((cid, sid))
        return {"session_id": sid, "knowledge": copy.deepcopy(doc)} if doc is not None else None

    def get_best_record(self, cid):
        for kind, key, sid in reversed(self.writes):
            if kind == "knowledge" and key == cid:
                return self.get_session(cid, sid)
        return None

    def download_session(self, cid, sid, destination, *, include_values):
        assert include_values is False
        for rel, data in self.files.get((cid, sid), {}).items():
            target = Path(destination) / "files" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.corrupt_download and rel == "kernel.py":
                data += b"\n# modified\n"
            target.write_bytes(data)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    contents = {"kernel.py": "def run(x):\n    return x + 1\n", "test.py": "# test fixture\n",
                "benchmark.py": "# benchmark fixture\n", "launch.json": '{"tile": 64}\n',
                "README.md": "# A synthetic bundle used for protocol tests\n"}
    for name, text in contents.items():
        (root / name).write_text(text)
    metadata = {"schema_version": 1, "operator": "rmsnorm", "backend": "triton", "name": "fixture",
                "implementation_id": "rmsnorm/triton/fixture", "status": "reference",
                "description": "Synthetic protocol-test source", "license": "MIT", "dependencies": [],
                "entry_point": {"file": "kernel.py", "symbol": "run"},
                "files": {"source": ["kernel.py"], "tests": ["test.py"], "benchmarks": ["benchmark.py"],
                          "config": ["launch.json"], "support": ["README.md"]},
                "commands": {"test": ["python", "test.py"], "benchmark": ["python", "benchmark.py"]},
                "supports": {"gfx": ["gfx942", "gfx950"], "dtypes": ["bf16"],
                             "shape": {"M": {"min": 1, "max": 16384}, "N": {"values": [64, 128]}}},
                "validation": {"status": "not_run"}, "provenance": {"provider": "test"}}
    save(root, metadata)
    return root


def save(root, metadata):
    (root / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))


def metadata(root):
    return yaml.safe_load((root / "metadata.yaml").read_text())


def measured(root, *, name=None, latency=1.0, best=False):
    data = metadata(root)
    if name:
        data["name"] = name
        data["implementation_id"] = "rmsnorm/triton/" + name
    code = SourceBundle.load(root).source_digest
    data["validation"] = {"status": "passed", "source_digest": code, "evidence": ["test-run.json"]}
    case = {"gfx": "gfx950", "dtype": "bf16", "shape": {"M": 16384, "N": 128}, "bench_key": "frozen-baseline"}
    data["benchmarks"] = [{**case, "kind": "measured", "baseline": "reference-v1",
                           "latency_us": latency, "baseline_latency_us": 4.0,
                           "correctness": "passed", "environment": {"gpu_model": "test-device", "rocm": "test"},
                           "source_digest": code, "evidence": ["measurements.json"]}]
    data["status"] = "best_known" if best else "validated"
    if best:
        data["best_for"] = [case]
    for name in ("test-run.json", "measurements.json"):
        (root / name).write_text('{"fixture": true}\n')
        if name not in data["files"]["support"]:
            data["files"]["support"].append(name)
    save(root, data)
    return data


@pytest.mark.parametrize("remote", [False, True])
def test_source_roundtrip_without_gpu_imports(source, tmp_path, remote):
    service = MemoryService()
    client = KernelClient(store=service) if remote else KernelClient.local(tmp_path / "collection")
    expected = SourceBundle.load(source)
    reference = client.put(source)
    results = client.search("rmsnorm", backend="triton", gfx="gfx950", dtype="bf16", shape={"M": 16384, "N": 128})
    item, = results["implementations"]
    assert item["revision"] == expected.revision == reference["revision"]
    assert not item["comparable"] and not item["best_known_for_query"]
    result = client.fetch(reference["implementation_id"], revision=reference["revision"], destination=tmp_path / "download")
    assert (result / "kernel.py").read_bytes() == (source / "kernel.py").read_bytes()
    assert (result / "metadata.yaml").is_file()
    assert SourceBundle.load(result).revision == expected.revision
    if remote:
        assert len(service.files) == 1
        assert len(service.records) == 2
        assert implementation_address(expected.implementation_id) != operator_address("rmsnorm")


def test_operator_search_does_not_need_kernel_symbol_or_fake_speedup(source):
    service = MemoryService()
    client = KernelClient(store=service)
    client.put(source)
    assert len(client.search("rmsnorm")["implementations"]) == 1
    assert not any("speedup" in doc for doc in service.records.values())


def test_unlisted_files_are_not_uploaded(source):
    (source / ".env").write_text("TEST_ONLY=not-an-artifact\n")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/cache.pyc").write_bytes(b"cache")
    service = MemoryService()
    KernelClient(store=service).put(source)
    names = next(iter(service.files.values()))
    assert ".env" not in names and "__pycache__/cache.pyc" not in names


def test_local_put_is_idempotent_but_never_overwrites_different_source(source, tmp_path):
    client = KernelClient.local(tmp_path / "collection")
    first = client.put(source)
    assert client.put(source) == first
    (source / "kernel.py").write_text("def run(x):\n    return x + 2\n")
    with pytest.raises(FileExistsError):
        client.put(source)
    assert SourceBundle.load(first["path"]).revision == first["revision"]


def test_remote_revisions_coexist_and_explicit_revision_retrieves_original(source, tmp_path):
    client = KernelClient(store=MemoryService())
    first = client.put(source)
    (source / "kernel.py").write_text("def run(x):\n    return x + 2\n")
    second = client.put(source)
    assert first["revision"] != second["revision"]
    old = client.fetch(first["implementation_id"], revision=first["revision"], destination=tmp_path / "old")
    latest = client.fetch(first["implementation_id"], destination=tmp_path / "latest")
    assert "x + 1" in (old / "kernel.py").read_text()
    assert "x + 2" in (latest / "kernel.py").read_text()


@pytest.mark.parametrize("query", [
    {"gfx": "gfx1250"}, {"dtype": "fp32"}, {"shape": {"M": 20000}},
    {"shape": {"N": 127}}, {"shape": {"K": 128}}, {"backend": "hip"},
])
def test_applicability_filters(source, tmp_path, query):
    client = KernelClient.local(tmp_path / "collection")
    client.put(source)
    assert not client.search("rmsnorm", **query)["implementations"]


def test_best_known_is_case_specific_and_upstream_reports_are_not_ranked(source):
    service = MemoryService()
    client = KernelClient(store=service)
    data = measured(source, best=True)
    client.put(source)
    query = data["best_for"][0]
    result = client.search("rmsnorm", best_only=True, **query)
    assert result["implementations"][0]["best_known_for_query"]
    assert not client.search("rmsnorm")["implementations"][0]["best_known_for_query"]
    assert not client.search("rmsnorm", best_only=True, **{**query, "shape": {"M": 8192, "N": 128}})["implementations"]
    with pytest.raises(BundleError, match="require"):
        client.search("rmsnorm", best_only=True)


def test_measurement_order_uses_matching_benchmark_case(source):
    client = KernelClient(store=MemoryService())
    data = measured(source, name="slow", latency=3)
    client.put(source)
    data = measured(source, name="fast", latency=1)
    client.put(source)
    query = {k: data["benchmarks"][0][k] for k in ("gfx", "dtype", "shape", "bench_key")}
    results = client.search("rmsnorm", **query)["implementations"]
    assert [r["metadata"]["name"] for r in results] == ["fast", "slow"]


def test_reused_benchmark_key_cannot_mix_environments(source):
    client = KernelClient(store=MemoryService())
    data = measured(source, name="one")
    client.put(source)
    data = measured(source, name="two")
    data["benchmarks"][0]["environment"]["rocm"] = "different"
    save(source, data)
    client.put(source)
    query = {k: data["benchmarks"][0][k] for k in ("gfx", "dtype", "shape", "bench_key")}
    with pytest.raises(BundleError, match="incompatible"):
        client.search("rmsnorm", **query)


@pytest.mark.parametrize("file", ["kernel.py", "launch.json"])
def test_changed_runtime_invalidates_old_measurements(source, file):
    measured(source, best=True)
    (source / file).write_text((source / file).read_text() + "\n")
    with pytest.raises(BundleError, match="different source"):
        SourceBundle.load(source)


def test_best_known_without_passed_validation_is_rejected(source):
    data = metadata(source)
    data["status"] = "best_known"
    save(source, data)
    with pytest.raises(BundleError, match="passed validation"):
        SourceBundle.load(source)


def test_regression_cannot_be_declared_best(source):
    measured(source, latency=5, best=True)
    with pytest.raises(BundleError, match="measured improvement"):
        SourceBundle.load(source)


@pytest.mark.parametrize("name", ["../outside.py", "/absolute.py", "nested/../outside.py", "patch.diff"])
def test_invalid_source_paths_fail_before_remote_write(source, name):
    data = metadata(source)
    data["files"]["source"].append(name)
    save(source, data)
    service = MemoryService()
    with pytest.raises(BundleError):
        KernelClient(store=service).put(source)
    assert not service.writes


def test_missing_reproduction_file_is_rejected(source):
    (source / "test.py").unlink()
    with pytest.raises(BundleError, match="missing bundle"):
        SourceBundle.load(source)


def test_local_evidence_cannot_be_left_out_of_downloaded_bundle(source):
    data = measured(source)
    data["files"]["support"].remove("measurements.json")
    save(source, data)
    with pytest.raises(BundleError, match="local evidence must be a declared"):
        SourceBundle.load(source)


def test_symlink_source_is_rejected(source, tmp_path):
    (source / "kernel.py").unlink()
    outside = tmp_path / "outside.py"
    outside.write_text("def run(): pass\n")
    (source / "kernel.py").symlink_to(outside)
    with pytest.raises(BundleError, match="symlinks"):
        SourceBundle.load(source)


def test_duplicate_yaml_keys_are_rejected(source):
    with (source / "metadata.yaml").open("a") as handle:
        handle.write("operator: gemm\n")
    with pytest.raises(BundleError, match="unique"):
        SourceBundle.load(source)


def test_remote_corruption_never_publishes_destination(source, tmp_path):
    service = MemoryService()
    client = KernelClient(store=service)
    ref = client.put(source)
    service.corrupt_download = True
    target = tmp_path / "result"
    with pytest.raises(BundleError, match="does not match"):
        client.fetch(ref["implementation_id"], destination=target)
    assert not target.exists()


def test_fetch_rejects_wrong_revision_and_existing_destination(source, tmp_path):
    client = KernelClient.local(tmp_path / "collection")
    ref = client.put(source)
    with pytest.raises(BundleError, match="not in this checkout"):
        client.fetch(ref["implementation_id"], revision="0" * 64, destination=tmp_path / "wrong")
    with pytest.raises(FileExistsError):
        client.fetch(ref["implementation_id"], destination=source)


def test_index_failure_is_explicit_and_retry_reuses_source_revision(source):
    service = MemoryService()
    service.fail_id = operator_address("rmsnorm")
    client = KernelClient(store=service)
    with pytest.raises(KBStoreError, match="completed identities"):
        client.put(source)
    assert len(service.files) == 1 and len(service.records) == 1
    service.fail_id = None
    client.put(source)
    assert len(service.files) == 1 and len(service.records) == 2


def test_read_only_blocks_local_and_remote_put(source, tmp_path, monkeypatch):
    monkeypatch.setenv("GEAK_KB_READ_ONLY", "1")
    for client in (KernelClient.local(tmp_path / "collection"), KernelClient(store=MemoryService())):
        with pytest.raises(KBStoreError, match="READ_ONLY"):
            client.put(source)


def test_metadata_override_is_packaged_without_mutating_source(source, tmp_path):
    client = KernelClient.local(tmp_path / "collection")
    original = metadata(source)
    override = copy.deepcopy(original)
    override["description"] = "An explicit metadata revision"
    ref = client.put(source, metadata=override)
    assert SourceBundle.load(ref["path"]).metadata["description"] == override["description"]
    assert metadata(source) == original


def test_remote_pagination_and_bound_are_visible(source):
    service = MemoryService()
    client = KernelClient(store=service)
    ref = client.put(source)
    cid = operator_address("rmsnorm")
    for n in range(205):
        service.records[cid, f"a-{n:04}"] = {INDEX_METRIC: 1, "value": {"carrier": "patch"}}
    assert client.search("rmsnorm", scan=300)["matched_count"] == 1
    bounded = client.search("rmsnorm", scan=100)
    assert bounded["scan_truncated"] and not bounded["implementations"]
