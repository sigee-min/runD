#include "../record.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

template <class T>
[[nodiscard]] std::uint64_t CapacityBytes(const std::vector<T> &values) {
  return ::rund::detail::counter::SaturatingMultiply(
      static_cast<std::uint64_t>(values.capacity()), sizeof(T));
}

} // namespace

std::uint64_t VulkanPipelineRecordHostBytes(
    const VulkanPipelineRecordRecipe &recipe) noexcept {
  std::uint64_t bytes = sizeof(VulkanPipelineRecordRecipe);
  for (const std::uint64_t current :
       {CapacityBytes(recipe.entries), CapacityBytes(recipe.barriers),
        CapacityBytes(recipe.transducer_ready), CapacityBytes(recipe.canonical),
        CapacityBytes(recipe.status_steps),
        CapacityBytes(recipe.telemetry_steps)}) {
    bytes = ::rund::detail::counter::SaturatingAdd(bytes, current);
  }
  return bytes;
}

bool VulkanPipelineRecordHostBytes(
    const PreparedKernelPipelineReservation &reservation,
    std::uint64_t &bytes) noexcept {
  bytes = sizeof(VulkanPipelineRecordRecipe);
  const auto add_extent = [&bytes](const std::uint64_t count,
                                   const std::uint64_t element_bytes) noexcept {
    std::uint64_t extent = 0u;
    return rund::kernel::checked::mul(count, element_bytes, extent) &&
           rund::kernel::checked::add(bytes, extent, bytes);
  };
  return add_extent(reservation.occurrence_count,
                    sizeof(VulkanPipelineRecordEntry)) &&
         add_extent(reservation.occurrence_count, sizeof(std::uint8_t)) &&
         add_extent(reservation.nested_group_count, sizeof(std::uint8_t)) &&
         add_extent(reservation.backend_status_source_count,
                    sizeof(VulkanPipelineCanonicalStatus)) &&
         add_extent(reservation.backend_step_description_count,
                    sizeof(PreparedProgramStatusSlice)) &&
         add_extent(reservation.backend_step_description_count,
                    sizeof(PreparedProgramStatusSlice));
}

#endif

} // namespace rund::node::accel::detail
