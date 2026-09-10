#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>

#include <node/accel/context.hpp>

#include "../backend/ops/table.hpp"
#include "admission.hpp"
#include "capability.hpp"
#include "shared.hpp"
namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelContext RejectContext(const char *reason);

[[nodiscard]] rund::AccelBuffer RejectBuffer(const rund::AccelBufferDesc &desc,
                                             const rund::Buffer &buffer,
                                             const char *reason);

[[nodiscard]] rund::AccelBuffer CreateAccelBufferWithInitialization(
    const rund::AccelContext &context, rund::AccelBufferDesc desc,
    BackendBufferInitialization initialization,
    BackendBufferMemory memory = BackendBufferMemory::DeviceLocal,
    std::uint64_t exact_storage_bytes = 0u);

// Authenticates one existing AccelBuffer capability and mints a semantic
// typed view over the same backend allocation. Width/count may be retyped
// together, but the descriptor must cover the source's exact admitted extent;
// its requested role must remain compatible with canonical backend usage.
[[nodiscard]] rund::AccelBuffer
ProjectAccelBufferView(const rund::AccelContext &context,
                       const rund::AccelBuffer &buffer,
                       rund::AccelBufferDesc desc);

} // namespace rund::node::accel::detail
