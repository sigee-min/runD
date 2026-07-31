#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>

#include "../../backend/resource.hpp"
#include "../admission.hpp"
#include "../shared.hpp"
#include "reference.hpp"
#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] bool
ContextMatchesAdmission(const rund::AccelContext &context,
                        const ContextAdmission &admission) noexcept;
[[nodiscard]] bool
AccelBufferOwnerMatches(const ContextAdmission &admission,
                        const rund::AccelBuffer &buffer) noexcept;
[[nodiscard]] bool BufferOwnerMatches(const ContextAdmission &admission,
                                      const rund::AccelContext &context,
                                      const rund::Buffer &buffer) noexcept;

} // namespace rund::node::accel::detail
