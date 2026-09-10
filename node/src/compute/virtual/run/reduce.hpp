#pragma once

#include "projection.hpp"

#include "../../../hash/fnv.hpp"
#include "../../device/residency/registry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

struct VirtualReduction final {
  __uint128_t total{};
  std::array<std::byte, sizeof(std::uint64_t)> value{};
  std::uint32_t operation{};
  Type type{Type::U32};
  bool has_value{};
};

[[nodiscard]] Status
begin_virtual_reduction(const VirtualRunProjection &run,
                        VirtualReduction &reduction) noexcept;

[[nodiscard]] Status
consume_virtual_reduction(const VirtualRunProjection &run,
                          residency::EpochLease lease,
                          VirtualReduction &reduction) noexcept;

[[nodiscard]] Status
finish_virtual_reduction(VirtualBacking &output,
                         const VirtualRunProjection &run,
                         VirtualReduction &reduction, ResidencyStats &stats,
                         ::rund::node::hash_detail::Fnv &output_hash) noexcept;

} // namespace rund::compute::detail
