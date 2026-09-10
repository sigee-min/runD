#pragma once

#include <accel/device.hpp>

#include "execute.hpp"

#include <array>
#include <cstddef>

namespace node_accel_contract::stencil {

template <typename T, std::size_t Count>
[[nodiscard]] bool MatchesDirectAndForcedRangePath(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ComputeDomain domain, const rund::kernel::StencilOp op,
    const rund::kernel::StencilElement element, const rund::kernel::u64 radius,
    const std::array<T, Count> &input,
    const match_detail::ForcedRangePath path) {
  return MatchesForcedPathReference(pick, scalar, domain, op, element, radius,
                                    input,
                                    match_detail::ForcedRangePath::Direct) &&
         MatchesForcedPathReference(pick, scalar, domain, op, element, radius,
                                    input, path);
}

} // namespace node_accel_contract::stencil
