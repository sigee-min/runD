#pragma once

#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/metadata.hpp>

#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

enum InputUse : std::uint8_t {
  InputUseDirect = 1u << 0u,
  InputUseUniform = 1u << 1u,
  InputUseIndexedSource = 1u << 2u,
  InputUseIndex = 1u << 3u,
};

struct InputWindowPlan final {
  rund::kernel::u64 required_count{};
  std::uint8_t uses{};

  [[nodiscard]] constexpr bool base_anchored() const noexcept {
    return (uses & (InputUseUniform | InputUseIndexedSource)) != 0u;
  }

  [[nodiscard]] constexpr bool windowed() const noexcept {
    return (uses & (InputUseDirect | InputUseIndex)) != 0u;
  }
};

[[nodiscard]] inline bool
FreezeInputWindowPlans(const rund::kernel::ExecutionMetadata &metadata,
                       const rund::kernel::u64 tile_count,
                       const std::span<InputWindowPlan> plans) noexcept {
  if (!metadata || tile_count == 0u || metadata.read_count > 64u ||
      metadata.read_count != plans.size()) {
    return false;
  }
  rund::kernel::u64 indexed_source_mask = 0u;
  rund::kernel::u64 index_mask = 0u;
  for (const rund::kernel::ReadRoute route : metadata.read_routes) {
    if (route.source >= metadata.read_count ||
        route.index >= metadata.read_count) {
      return false;
    }
    indexed_source_mask |= rund::kernel::u64{1u} << route.source;
    index_mask |= rund::kernel::u64{1u} << route.index;
  }
  for (std::size_t index = 0u; index < plans.size(); ++index) {
    const rund::kernel::u64 binding = static_cast<rund::kernel::u64>(index);
    const rund::kernel::u64 bit = rund::kernel::u64{1u} << binding;
    std::uint8_t uses{};
    if ((metadata.direct_read_mask & bit) != 0u) {
      uses |= InputUseDirect;
    }
    if ((metadata.uniform_read_mask & bit) != 0u) {
      uses |= InputUseUniform;
    }
    if ((indexed_source_mask & bit) != 0u) {
      uses |= InputUseIndexedSource;
    }
    if ((index_mask & bit) != 0u) {
      uses |= InputUseIndex;
    }
    const rund::kernel::u64 required =
        rund::kernel::RequiredInputCount(metadata, binding, tile_count);
    const InputWindowPlan plan{required, uses};
    if (uses == 0u || required == 0u ||
        (plan.base_anchored() && plan.windowed())) {
      return false;
    }
    plans[index] = plan;
  }
  return true;
}

[[nodiscard]] inline rund::kernel::ComputeDispatchWindow
InputWindow(const InputWindowPlan plan,
            const rund::kernel::ComputeDispatchWindow window) noexcept {
  return plan.base_anchored()
             ? rund::kernel::ComputeDispatchWindow{
                   .begin_sequence = 0u,
                   .tile_count = plan.required_count,
               }
             : window;
}

} // namespace rund::node::accel::detail
