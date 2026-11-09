# Third-party notices

## GEAK

Knowledge and bundled workflow resources are derived from AMD-AGI/GEAK, commit `02bfbb0b72746941be3ab3ad48fb21a284ae2fde`.
Copyright © 2026 Advanced Micro Devices, Inc. All rights reserved. Distributed under the Apache License 2.0 in [LICENSE.md](LICENSE.md).

KernelWiki reorganized documents, adjusted links and distilled reusable conclusions into corpus decisions. Per-document migration provenance identifies original paths. Nested resources retain their own notices where present.

## SGLang-AMD-Mono workload references

The public workload cases and derived fusion decisions reference
[sgl-project/sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) at commit
`7a4a552db119d1c3689165a182f151388a636272`. The source index links the reviewed files and
selected line ranges; those workload references do not vendor upstream kernels or code objects.
Copyright 2026 SGLang contributors, Apache License 2.0; see the pinned upstream
[LICENSE](https://github.com/sgl-project/sgl-amd-mono/blob/7a4a552db119d1c3689165a182f151388a636272/LICENSE)
and [NOTICE](https://github.com/sgl-project/sgl-amd-mono/blob/7a4a552db119d1c3689165a182f151388a636272/NOTICE).
The reviewed files also retain vLLM contributor notices and, for adapted common/DeepSeek
components, ROCm/ATOM's AMD attribution under MIT. KernelWiki summarizes their contracts
and adds scoped decisions; source inspection does not confer upstream validation or measured gains.

## InferenceX container source snapshots

`knowledge/top_model/muse_spark/kernel_sources/blobs/` contains selected, unmodified source
files exported from the public `lmsysorg/sglang-rocm` images identified in the adjacent
manifest. These reading snapshots cover the cited SGLang, AITER and SGLang-AMD-Mono
kernel implementations and dispatch paths. They are deduplicated by file content;
container paths, image digests and SHA256 values are retained in the manifest. They are
not a full working copy, and their inclusion does not establish upstream validation.

Existing per-file copyright and SPDX notices are preserved byte for byte. SGLang and
SGLang-AMD-Mono code is covered by Apache License 2.0, included in [LICENSE.md](LICENSE.md),
with SGLang and vLLM contributor notices where present. The AITER MIT notice below also
applies to its included files, subject to their per-file notices. These are container
snapshots and must not be assumed identical to a named upstream commit.
The public SGLang-AMD-Mono [NOTICE](knowledge/top_model/muse_spark/kernel_sources/notices/sgl-amd-mono-NOTICE.txt)
is also retained from the public reference commit cited above; it is attribution material,
not a claim that the container code matches that revision.

## AITER

The structured corpus contains attributed source excerpts: the retained GEMM snapshot is
`a6bb499375849eec45d68c5ccaebc8865fd422c0`, and the operator expansion uses
`b3ad6c461703110dc36286265c16e453ef0e2dd2`. The corpus catalog records repositories and full
commits; operator-local evidence records identify their snapshot, and generated pages link to exact
source ranges. The extraction, curation and rendering
changes do not confer any upstream benchmark or validation status on these excerpts.

Reviewed PR provenance under `knowledge/_sources/aiter/pull_requests/` includes short
before/after code excerpts at the full change and parent commits recorded in each YAML.
The AITER MIT notice applies subject to each file's retained notices. The Triton prefill
attention source additionally carries `Copyright (C) 2023-2026 SGLang Team` and the
Apache License 2.0 notice, whose text is included in [LICENSE.md](LICENSE.md); both
attributions are retained here. Attributed PR descriptions and comments are quoted with their source URLs,
capture/update times and excerpt fingerprints. Reported performance remains attributed
to its upstream author and is not presented as reproduced KernelWiki validation.

Source references and selected documentation excerpts use ROCm/aiter, commit `e9995854abbd935f0190b19abebe9938b6b90cca`. Additional native-pipeline and workload-variant reviews reference commit `b3ad6c461703110dc36286265c16e453ef0e2dd2`. Each page or added section identifies its source revision. KernelWiki does not distribute AITER code objects or a full working copy of AITER.

AITER's general MIT notice is reproduced below. Per-file SPDX notices take precedence for
individual excerpts. In particular,
[`aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py`](https://github.com/ROCm/aiter/blob/b3ad6c461703110dc36286265c16e453ef0e2dd2/aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/config.py)
declares `SPDX-License-Identifier: Apache-2.0`; its excerpts in the shared-expert corpus retain
that license, whose text is included in [LICENSE.md](LICENSE.md).

```text
Copyright © Advanced Micro Devices, Inc. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
