# Bonsai 2 4B on a native CPU

OMG can run Prism's Ternary Bonsai 2 4B GGUF through the native llama.cpp
backend. The model is a Qwen3.5 hybrid (attention plus recurrent layers),
with Prism's Hadamard activation transforms and packed ternary kernels.
The backend is pinned to Prism's `prism-b10828-a153697`, rather than stock
llama.cpp; see [the vendored wrapper notes](../vendor/llama-cpp-sys-2/README.omg.md).

## Smallest official packing

Use **PTQ1_0**, 949,549,920 bytes (906 MiB), instead of PQ2_0,
1,146,465,120 bytes. This saves 196,915,200 bytes (17.18%). These are
different packings of the same official ternary weights, not a new trained
or vocabulary-pruned checkpoint. The full vocabulary and tied output
embedding are preserved. `tools/prune_vocab.py` supports Gemma only and
rejects Bonsai explicitly.

Download with Python 3 (standard library only):

```sh
python tools/download_bonsai.py models
```

The downloader pins Hugging Face revision
`8d6557168168f91ea6697a4a49ed2ef622b53adb`, checks the byte size and SHA-256
`56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60`,
and installs the file atomically. An already verified file is reused.

Sources: [official model repository](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf),
[Prism backend audit](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md).

## Build and run

Install Rust stable and the native build prerequisites: a C/C++ compiler,
CMake, and libclang for bindgen. On Windows use the MSVC Rust toolchain,
Visual Studio Build Tools with Desktop development with C++, CMake, and
LLVM/libclang (set `LIBCLANG_PATH` if bindgen cannot locate its DLL).
No CUDA SDK is required.

From the repository root, in PowerShell:

```powershell
git submodule update --init --recursive
cargo build -p omg-cli --release --no-default-features
python tools/download_bonsai.py models
.\target\release\omg.exe --cpu-low-memory probe --model models/Ternary-Bonsai-2-4B-PTQ1_0.gguf --request examples/ticket-ja.json --mode check
.\target\release\omg.exe --cpu-low-memory serve --model models/Ternary-Bonsai-2-4B-PTQ1_0.gguf
```

On macOS/Linux use `./target/release/omg` in place of the `.exe` path.
The global `--cpu-low-memory` preset applies to native GGUF models and
rejects checkpoint directories. It caps context at 2,048 tokens, batch and
microbatch at 64, and sequences at eight; it uses four CPU threads,
zero GPU layers, no KQV/op offload, and no serialized RAM/disk state cache.
Smaller command-specific context/batch values remain smaller. The resident
prefix is still reused. At most seven branches fit in a packed request with
one prefix. For larger requests use `probe --mode separate` or the server's
`/v1/systemone/separate` endpoint, or reduce questions/option orders. The planner does not
split a single oversized request into smaller packed requests. Option orders
and calibration can add branches/passes.

This bounds runtime allocations for the intended i5 / 8 GB machine; it is
not a Windows latency or total system memory guarantee. macOS may initialize
the Metal backend through the Rust dependency even in a CPU build, but the
preset keeps weights, cache, and compute on the CPU.

Bonsai's label renderer uses its GGUF ChatML template without BOS, including
the empty `<think>` block for `enable_thinking=false`. Choice, Noul and Score
use the existing next-token label readout. A Gemma pointer head cannot be
reused: a head must be trained with the matching `chatml_label_pointer`
layout. Probabilities remain zero-shot and uncalibrated by default.

Branches within a request share their prefix and remain isolated. For
hybrid/recurrent models, prefix and individual branches use separate decode
calls; `evaluate_many` also serializes unrelated requests even when
cross-request batching is requested. The pinned backend produced different
logits when mixing those states in one decode. These models therefore reuse
the prefix but do not run all questions in one native decode. Attention-only
models retain the existing batching behavior. On decode failure, resident
state is cleared before another request can reuse it.

## Verification

Ordinary CI checks include native CPU compilation/tests on Linux and Windows.
The real model regression is opt-in, so CI does not download weights:

```powershell
$env:OMG_BONSAI_MODEL = (Resolve-Path models/Ternary-Bonsai-2-4B-PTQ1_0.gguf)
cargo test -p omg-llama --release --no-default-features bonsai_cpu_branch_isolation_and_prefix_cache -- --ignored --nocapture
```

It compares all vocabulary logits for packed/individual branches, resident
reuse, RAM and disk restoration, cold recomputation, and request ordering,
with a small batch forcing multiple native decode calls. RAM/disk caches
are enabled in this test to exercise them; the low-memory preset disables
them in normal use.

### Local CPU smoke measurements

One `probe --mode check` run with `examples/ticket-ja.json` (five Japanese
questions), Apple M4 / 16 GiB macOS, four CPU threads, zero GPU layers,
context 2,048 and batch/microbatch 64:

| Packing | File bytes | Cold five-question request | Five individual questions, prefix resident | Peak process RSS |
| --- | ---: | ---: | ---: | ---: |
| PTQ1_0 | 949,549,920 | 18.95 s | 14.46 s | 1.47 GiB |
| PQ2_0 | 1,146,465,120 | 12.52 s | 9.81 s | 1.64 GiB |

The first column of timings includes prefix decoding; the second reuses that
prefix in the same process. These single runs are smoke measurements, not a
speedup comparison or a Windows/i5 benchmark. PTQ1_0 is smaller but was slower
on this host; CPU architecture and kernels affect this tradeoff. Both packings
selected the same top choices in this example, but their probabilities differ;
this is not a quality evaluation. PTQ1_0's maximum absolute probability
difference between grouped and individual evaluation was 0.000013 after
isolating native decode boundaries. The CLI's reported `1 pass` counts an OMG
evaluation, which contains multiple native decode calls for this hybrid model.
Actual Windows/i5 performance and memory use still need measurement on the
target machine.
