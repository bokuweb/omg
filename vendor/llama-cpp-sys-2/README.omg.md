# OMG's native Bonsai backend

The Rust FFI wrapper is `llama-cpp-sys-2` **0.1.156**, from
[utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs), licensed
MIT OR Apache-2.0 (license texts included). The root Cargo patch applies this
copy to both OMG and `llama-cpp-2` 0.1.156; the Rust API is unchanged.

The `llama.cpp` submodule is PrismML's fork, pinned to
[`prism-b10828-a153697`](https://github.com/PrismML-Eng/llama.cpp/tree/a15369706017f177cae8aa40b6b8592259a48806).
It retains its own MIT license. Initialize it with `git submodule update --init
--recursive`. No generated binaries or model weights are vendored.

This fork supplies Bonsai's Hadamard activation graph and native packed
PQ2_0/PTQ1_0 CPU kernels. Merely renaming or converting tensor types in the
upstream backend would not preserve this computation. See the
[Prism backend audit](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md).

Local wrapper change: default features are empty rather than `common`.
OMG uses only the model/context API, and the optional upstream common helper
wrapper does not match Prism's common API. The `common`/`mtmd` helpers are not
supported by this patch; keep those features disabled. GPU features remain
available for existing users, but the Bonsai CPU preset offloads zero layers.

This patch affects all native GGUF models in the workspace. Re-evaluate other
models before deploying a backend upgrade. To update the fork, update the
submodule pin and re-run the ignored Bonsai model test, as well as the ordinary
workspace checks. The patch is workspace-local and is not a publishable
replacement sys crate.
