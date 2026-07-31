#pragma once

#include <kernel/program/compute/cpu.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] rund::kernel::CpuCaps DetectCpuCaps() noexcept;

} // namespace rund::node::accel::detail
