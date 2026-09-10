#include "../upper.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] std::uint64_t
VulkanDecimalDigitCount(std::uint64_t value) noexcept {
  std::uint64_t digits = 1u;
  while (value >= 10u) {
    value /= 10u;
    ++digits;
  }
  return digits;
}

[[nodiscard]] bool
VulkanControlledMapSourceUpperBytes(const rund::kernel::ComputePlan &plan,
                                    const std::uint64_t specialized,
                                    std::uint64_t &upper) noexcept {
  using namespace vulkan_controlled_map_source_detail;
  std::uint64_t binding = 0u;
  if (!rund::kernel::checked::add(plan.input_buffer_count,
                                  plan.output_buffer_count, binding) ||
      !rund::kernel::checked::add(binding, 1u, binding)) {
    return false;
  }
  const std::uint64_t growth =
      DeclarationPrefix.size() + VulkanDecimalDigitCount(binding) +
      DeclarationSuffix.size() + ControlledGuard.size() - Guard.size() +
      ControlledVariant.size() - CanonicalVariant.size();
  return rund::kernel::checked::add(specialized, growth, upper);
}

} // namespace rund::node::accel::detail
