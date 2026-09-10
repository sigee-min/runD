# Residency Backends

This directory owns backend capability admission and native execution facts.
It does not own cache policy, page selection, or publication.

1. [CPU](./cpu.md) owns the Host-only route and compile-time isolation.
2. [Common](./common.md) owns backend-neutral accelerator admission.
3. [Metal](./metal.md) owns MTL4 terminal and bounded-window behavior.
4. [Vulkan](./vulkan.md) owns timeline, MoltenVK, and fallback behavior.

Cache and frame state remain in [State](../state.md). Recurrent scheduling and
Host-service joins remain in [Execution](../execution/README.md).
