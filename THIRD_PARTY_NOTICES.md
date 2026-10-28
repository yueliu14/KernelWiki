# Third-party notices

## GEAK

Knowledge and the `kernelwiki/_store` protocol implementation are derived from AMD-AGI/GEAK, commit `02bfbb0b72746941be3ab3ad48fb21a284ae2fde`.
Copyright © 2026 Advanced Micro Devices, Inc. All rights reserved. Distributed under the Apache License 2.0 in [LICENSE.md](LICENSE.md).

KernelWiki reorganized documents, adjusted links and Python package imports, added the GEAK_KB_URL alias, and added source-bundle validation, storage and retrieval. Per-document migration provenance identifies original paths. Nested resources retain their own notices where present.

## AITER

The structured corpus contains attributed source excerpts: the retained GEMM snapshot is
`a6bb499375849eec45d68c5ccaebc8865fd422c0`, and the attention/MoE/indexer expansion uses
`b3ad6c461703110dc36286265c16e453ef0e2dd2`. Evidence files record the source repository and
commit; generated pages link to the exact source ranges. The extraction, curation and rendering
changes do not confer any upstream benchmark or validation status on these excerpts.

Source references, selected documentation excerpts and the kernel body/naming helper in `kernels/rmsnorm/triton/aiter_row_tiled/` use ROCm/aiter, commit `e9995854abbd935f0190b19abebe9938b6b90cca`. Additional native-pipeline and workload-variant reviews reference commit `b3ad6c461703110dc36286265c16e453ef0e2dd2`. Each page or added section identifies its source revision. The source bundle preserves its MIT license and documents the standalone wrapper and packaging adaptations. KernelWiki does not distribute AITER code objects or a full working copy of AITER. The original license for AITER material follows.

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
