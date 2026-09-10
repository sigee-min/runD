#pragma once

#include "../local.hpp"

#include <rund/compute/abi/primitive.hpp>

#include <cstdint>

namespace rund::compute::detail {

void record_cpu_algebra_status(CpuGraphRun &, Primitive, std::uint64_t failed,
                               std::uint32_t first) noexcept;

} // namespace rund::compute::detail
