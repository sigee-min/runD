#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>

#include <node/accel/context.hpp>

#include "admission.hpp"
#include "../backend/ops/table.hpp"
#include "capability.hpp"
#include "shared.hpp"
namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelContext RejectContext(const char *reason);

[[nodiscard]] rund::AccelBuffer RejectBuffer(const rund::AccelBufferDesc &desc,
                                             const rund::Buffer &buffer,
                                             const char *reason);

[[nodiscard]] rund::AccelBuffer CreateAccelBufferWithInitialization(
    const rund::AccelContext &context, rund::AccelBufferDesc desc,
    BackendBufferInitialization initialization);

} // namespace rund::node::accel::detail
