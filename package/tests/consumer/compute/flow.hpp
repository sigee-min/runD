#pragma once

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace package_compute {

inline int Flow() {
  constexpr std::size_t count = 64u * 1024u + 17u;
  std::vector<std::uint32_t> values(count);
  for (std::size_t index = 0u; index < count; ++index) {
    values[index] = static_cast<std::uint32_t>(index % 7u);
  }
  auto prefix = rund::compute::on(rund::compute::Target::cpu(), values)
                    .map("adjust", [](auto x) { return x + 1; })
                    .scan(rund::compute::Scan::InclusiveSum)
                    .collect();
  if (!prefix) {
    return prefix.exit_code();
  }
  if (prefix->size() != count) {
    return 2;
  }
  std::uint32_t sum = 0u;
  for (std::size_t index = 0u; index < count; ++index) {
    sum += values[index] + 1u;
    if ((*prefix)[index] != sum) {
      return 2;
    }
  }
  using Fixed = rund::compute::Fixed<1, 31>;
  const std::array fixed{Fixed::from_raw(1 << 28), Fixed::from_raw(1 << 29)};
  auto functional = rund::compute::on(rund::compute::Target::cpu(), fixed)
                        .map("functional",
                             [](auto value) {
                               const auto quarter = rund::compute::fixed(
                                   rund::compute::FixedOp::Quarter, value);
                               const auto soft = rund::compute::quantize<
                                   Fixed, rund::compute::Rounding::NearestEven,
                                   rund::compute::Overflow::Saturate,
                                   rund::compute::Approximation::Deterministic>(
                                   rund::compute::softsign(value));
                               const auto robust = rund::compute::quantize<
                                   Fixed, rund::compute::Rounding::NearestEven,
                                   rund::compute::Overflow::Saturate,
                                   rund::compute::Approximation::Deterministic>(
                                   rund::compute::huber(value, quarter));
                               return rund::compute::quantize<
                                   Fixed, rund::compute::Rounding::NearestEven,
                                   rund::compute::Overflow::Saturate,
                                   rund::compute::Approximation::Deterministic>(
                                   rund::compute::add_sat(soft, robust));
                             })
                        .collect();
  if (!functional) {
    std::fprintf(stderr, "package fixed functional failed: %.*s\n",
                 static_cast<int>(functional.error().size()),
                 functional.error().data());
    return functional.exit_code();
  }
  if (functional->size() != fixed.size()) {
    return 2;
  }
  return 0;
}

} // namespace package_compute
