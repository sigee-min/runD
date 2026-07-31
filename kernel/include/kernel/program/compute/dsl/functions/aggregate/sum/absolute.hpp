#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue sum(const SumOpAbs, const ComputeValue lhs,
                                      const ComputeValue rhs) noexcept {
  return sum(detail::StorageQuantize(abs(lhs)),
             detail::StorageQuantize(abs(rhs)));
}

[[nodiscard]] inline ComputeValue sum(const SumOpAbs, const ComputeValue a,
                                      const ComputeValue b,
                                      const ComputeValue c) noexcept {
  return sum(detail::StorageQuantize(abs(a)),
             detail::StorageQuantize(abs(b)),
             detail::StorageQuantize(abs(c)));
}

[[nodiscard]] inline ComputeValue
sum(const SumOpAbs, const ComputeValue a, const ComputeValue b,
    const ComputeValue c, const ComputeValue d) noexcept {
  return sum(detail::StorageQuantize(abs(a)),
             detail::StorageQuantize(abs(b)),
             detail::StorageQuantize(abs(c)),
             detail::StorageQuantize(abs(d)));
}

} // namespace rund::compute_dsl
