#pragma once

#include <accel/buffer.hpp>
#include <accel/check.hpp>

#include <node/accel/context.hpp>

#include "../backend/match.hpp"
#include "../backend/usage.hpp"

#include <kernel/core/checked.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck RejectAccelCheck(const char *reason) noexcept;

[[nodiscard]] rund::AccelCheck OkAccelCheck() noexcept;

} // namespace rund::node::accel::detail
