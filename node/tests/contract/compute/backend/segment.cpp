#include "model.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

namespace rund_node_backend_contract {

template <class T> [[nodiscard]] bool CheckSegmentedDomainMatrix() {
  constexpr std::array<std::uint32_t, 6> heads{1u, 0u, 0u, 1u, 0u, 0u};
  PrimitiveEvidence scan{};
  PrimitiveEvidence reduce{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!CheckTwoInputPrimitive<T>(
            backend, "backend-segmented-scan", heads,
            [](auto values, const std::size_t count) {
              return std::move(values).segmented_scan(
                  count, rund::compute::Scan::InclusiveSum);
            },
            scan) ||
        !CheckTwoInputPrimitive<T>(
            backend, "backend-segmented-reduce", heads,
            [](auto values, const std::size_t count) {
              return std::move(values).segmented_reduce(
                  count, rund::compute::Reduce::Sum);
            },
            reduce)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckSegments() {
  return CheckSegmentedDomainMatrix<std::int32_t>() &&
         CheckSegmentedDomainMatrix<std::uint32_t>() &&
         CheckSegmentedDomainMatrix<std::int64_t>() &&
         CheckSegmentedDomainMatrix<std::uint64_t>() &&
         CheckSegmentedDomainMatrix<rund::compute::Fixed<1, 31>>() &&
         CheckSegmentedDomainMatrix<rund::compute::Fixed<1, 63>>();
}

} // namespace rund_node_backend_contract
