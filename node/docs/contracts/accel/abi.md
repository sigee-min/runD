# Native ABI Boundaries

## Authority

Repository docs route to the Node build graph and Accel runtime contracts.
Private native headers own their namespaces, direct type dependencies, and SDK
conditions. A header is not a fragment injected inside a consumer's namespace.
This work first closes the Vulkan Kernel resource boundary, then the seven
Metal Pipeline parameter layouts, then the durable verification route.

## Metal Pipeline Parameters

The seven parameter records exposed by `metal/kernel/pipeline/abi.hpp`
have one field schema each under `metal/kernel/pipeline/abi/schema/`.
`records.def` assembles those schemas for host declarations and verification;
`abi/source/begin.def` and `end.def` bound their production MSL projection.
The `.def` files are deliberate macro inputs, not standalone headers.
Each schema owns host and shader record names,
field widths and order, host initialization, fixed array extents, byte offsets,
record size, and alignment. Host declarations and Metal source declarations
are projections of those schemas, not separately maintained layouts.

The host's four policy words remain one `std::array`; the shader's existing
four scalar policy names remain unchanged. Their common schema explicitly
describes this representation mapping. It does not change arithmetic, shader
entry points, bindings, parameter values, or native resource ownership.

Production shader bytes and existing golden hashes must remain unchanged.
Host compile-time checks prove every declared offset, size, and alignment.
Native Metal verification must also compile layout assertions and execute
field-level host-to-GPU-to-host transfers for all seven records. Merely copying
the same untyped byte range does not prove that both sides interpret fields
identically.

## Verification Closure

The existing repository verification route must include standalone header
compilation with SDKs enabled and disabled, multi-translation-unit link checks,
and the native ABI contract. Checks must exercise the actual headers and schema
projections. They may not introduce a second implementation or a second list of
production translation-unit ownership.

`tools/test/run --fresh tools.native-headers` owns the private-header check.
Its scope is every Vulkan Kernel `.hpp`, the Vulkan Capture, Dispatch,
Timestamp, and Buffer Create declaration headers, and the Metal Pipeline ABI
header. It uses the configured compile database, checks repeated standalone
includes, and compiles forward/reverse include orders into separate objects
before a relocatable link. SDK availability comes from the configured native
component definitions; an unavailable SDK is reported, not claimed as tested.
Negative fixtures exercise missing dependencies, SDK-conditional failures, and
duplicate strong definitions. The check does not execute native work.

The same route also compiles the configured Metal and Vulkan Pipeline
SDK-disabled implementation owners, links a complete host executable against
their declared interfaces, and executes their unavailable-backend contract.
This verifies that residency readiness, sliding submission/abort, persistent
preparation/service, and recurrence diagnostics remain link-complete without
either SDK. Rejected submission cannot invoke a completion callback; diagnostic
queries clear prior observations; unavailable persistent preparation publishes
no lowering, request cell, ticket, or callable service. The two implementation
rows come from the configured compile database. The probe creates no native
device and does not count an unavailable path as native execution evidence.

`tools/test/run --fresh accel.kernel-core` owns the native ABI check in
`node/tests/contract/accel/kernel/metal/abi.cpp`, alongside the existing exact
production-source length/hash contract. Each field receives a distinct offset
seed, including nonzero upper 32 bits for 64-bit fields; the GPU reads and XORs
typed fields and the host compares typed results without comparing padding.
SDK-off builds still verify host layouts and initial values. SDK-on builds
require a device and seven successful GPU roundtrips; missing device, shader
compilation, queue creation, or dispatch completion is a test failure.

The three steps are not complete until the old injected header paths and
handwritten parameter-layout mirrors are removed and the registered checks
pass. Other backend records and algorithms remain with their existing owners;
this contract does not introduce a generic backend abstraction.
