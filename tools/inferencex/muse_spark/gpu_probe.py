#!/usr/bin/env python3
"""Bounded TEP8 c1 reproduction; keep profiler timing separate from E2E timing."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def request(path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request("http://127.0.0.1:18173" + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    job = os.environ["SLURM_JOB_ID"]
    metadata = {"job_id": job, "hostname": socket.gethostname(), "started_at": stamp(),
                "scheduler_env": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}}
    write(args.output / "allocation.json", metadata)
    for port in (18173, 18174):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is occupied")
    # Check cached snapshots on the allocated host: some links target node-local storage.
    hub = Path("/shared_nfs/yueliu14/muse_spark_opt_1021/e2e_ab/hf_hub")
    revisions = {}
    for repo in ("models--amd--Muse-Spark-Open-816B-A42B-Quark-MXFP4",
                 "models--RadixArk--Muse-Spark-Open-DFlash2"):
        revision = (hub / repo / "refs/main").read_text().strip()
        snapshot = hub / repo / "snapshots" / revision
        missing = [str(p) for p in snapshot.iterdir() if not p.exists()]
        if missing:
            raise RuntimeError(f"Incomplete cached {repo}: {missing[:3]}")
        shards = list(snapshot.glob("*.safetensors"))
        if not shards:
            raise RuntimeError(f"No weights for {repo}")
        for index in snapshot.glob("*.safetensors.index.json"):
            for shard in set(json.loads(index.read_text())["weight_map"].values()):
                if not (snapshot / shard).is_file():
                    raise RuntimeError(f"Missing shard {shard}")
        revisions[repo] = {"revision": revision, "path": str(snapshot),
                           "weight_files": len(shards), "weight_bytes": sum(p.stat().st_size for p in shards)}
    write(args.output / "model_cache.json", revisions)
    for endpoint in json.loads(args.endpoints.read_text()):
        out = args.output / endpoint["label"]
        out.mkdir(exist_ok=True)
        name = f"muse_verify_{job}_{endpoint['label']}_yueliu14"
        created = False
        try:
            with (out / "pull.log").open("w") as stream:
                run(["docker", "pull", endpoint["image"]], stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
            with (out / "image_inspect.json").open("w") as stream:
                run(["docker", "image", "inspect", endpoint["image"]], stdout=stream)
            env = dict(endpoint["environment"], HF_HUB_CACHE=str(hub),
                       HEAD_NODE_IP=socket.gethostbyname(socket.gethostname()))
            command = ["docker", "run", "-d", "--name", name, "--network", "host", "--ipc", "host",
                       "--shm-size", "64g", "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video",
                       "--cap-add", "SYS_PTRACE", "--security-opt", "seccomp=unconfined",
                       "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
                       "-v", "/mnt/hf_cache:/mnt/hf_cache:ro", "-v", "/shared_nfs:/shared_nfs"]
            for key, value in env.items():
                command.extend(["-e", f"{key}={value}"])
            command.extend([endpoint["image"], *endpoint["command"]])
            write(out / "launch.json", {"argv": command, "started_at": stamp(), "historical_job": endpoint["historical_job"]})
            result = run(command, capture_output=True)
            created = True
            (out / "container.id").write_text(result.stdout)
            # Eight allocated GPUs are required; do not reuse another task's container.
            with (out / "gpu_environment.txt").open("w") as stream:
                run(["docker", "exec", name, "bash", "-lc",
                     "rocminfo | head -100; rocm-smi; python3 -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count()); assert torch.cuda.device_count() == 8'"],
                    stdout=stream, stderr=subprocess.STDOUT, timeout=60)
            deadline = time.monotonic() + 1500
            while time.monotonic() < deadline:
                state = run(["docker", "inspect", "-f", "{{.State.Running}}", name], capture_output=True).stdout.strip()
                if state != "true":
                    raise RuntimeError("Server exited during startup")
                try:
                    request("/health", timeout=3)
                    break
                except Exception:
                    time.sleep(10)
            else:
                raise TimeoutError("Server startup exceeded 25 minutes")
            write(out / "server_info.json", request("/get_server_info"))
            # A deterministic shape probe, not the historical variable-length AgentX replay.
            text = ("Explain how a processor executes matrix multiplication, including tiling, data reuse, "
                    "synchronization, and the difference between memory traffic and arithmetic work. " * 16)
            payload = {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": 256, "ignore_eos": True}, "stream": False}
            write(out / "workload.json", payload)
            samples = []
            for phase in ("warmup", "unprofiled_1", "unprofiled_2", "unprofiled_3", "profiled"):
                # Empty radix cache for every request, outside the timed/profiled region.
                request("/flush_cache", {})
                if phase == "profiled":
                    profile = {"output_dir": str(out / "traces"), "activities": ["CPU", "GPU"],
                               "with_stack": False, "record_shapes": True, "detailed_annotations": True,
                               "profile_prefix": endpoint["label"], "profile_id": "muse_c1", "merge_profiles": False}
                    write(out / "profile_config.json", profile)
                    write(out / "profile_start.json", {"response": request("/start_profile", profile), "time": stamp()})
                started = stamp()
                start = time.perf_counter()
                result = request("/generate", payload)
                elapsed = time.perf_counter() - start
                samples.append({"phase": phase, "started_at": started, "elapsed_seconds": elapsed, "meta_info": result.get("meta_info"), "finished_at": stamp()})
                write(out / "samples.json", samples)
                if phase == "profiled":
                    write(out / "profile_stop.json", {"response": request("/stop_profile", {}), "time": stamp()})
            write(out / "status.json", {"status": "completed", "time": stamp()})
        except Exception as exc:
            write(out / "status.json", {"status": "failed", "error": repr(exc), "time": stamp()})
            print(endpoint["label"], repr(exc), flush=True)
        finally:
            if created:
                with (out / "server.log").open("w") as stream:
                    subprocess.run(["docker", "logs", name], stdout=stream, stderr=subprocess.STDOUT)
                subprocess.run(["docker", "rm", "-f", name], check=False)
    metadata["finished_at"] = stamp()
    write(args.output / "allocation.json", metadata)


if __name__ == "__main__":
    main()
