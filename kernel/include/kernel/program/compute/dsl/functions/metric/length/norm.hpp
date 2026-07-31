#pragma once

#include <kernel/program/compute/dsl/norm.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue len(Norm::L1Tag, const ComputeValue x,
                                      const ComputeValue y) noexcept {
  return add_sat(detail::StorageQuantize(abs(x)),
                 detail::StorageQuantize(abs(y)));
}

[[nodiscard]] inline ComputeValue len(Norm::L1Tag norm, const ComputeValue x,
                                      const ComputeValue y,
                                      const ComputeValue z) noexcept {
  return add_sat(len(norm, x, y), detail::StorageQuantize(abs(z)));
}

[[nodiscard]] inline ComputeValue len(Norm::LInfTag, const ComputeValue x,
                                      const ComputeValue y) noexcept {
  return max(abs(x), abs(y));
}

[[nodiscard]] inline ComputeValue len(Norm::LInfTag norm, const ComputeValue x,
                                      const ComputeValue y,
                                      const ComputeValue z) noexcept {
  return max(len(norm, x, y), abs(z));
}

} // namespace rund::compute_dsl
