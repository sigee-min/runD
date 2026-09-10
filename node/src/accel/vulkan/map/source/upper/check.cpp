#include "../upper.hpp"
#include "check_fragments.hpp"

#include <kernel/core/checked.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
namespace rund::node::accel::detail {

[[nodiscard]] bool VulkanMapCheckSourceUpperBytes(
    const std::uint64_t check_count, const std::uint64_t offset_digit_bytes,
    const std::uint64_t stride_digit_bytes,
    const std::uint64_t limit_digit_bytes, std::uint64_t &upper) noexcept {
  using rund::kernel::checked::add;
  using rund::kernel::checked::mul;
  if (check_count > std::numeric_limits<std::uint64_t>::max() - 2u) {
    return false;
  }
  std::uint64_t bytes = vulkan_map_source_detail::CheckPrefix.size();
  std::uint64_t item = 0u;
  if (!mul(check_count,
           vulkan_map_source_detail::BindingPrefix.size() +
               vulkan_map_source_detail::BindingIndex.size() +
               vulkan_map_source_detail::BindingWords.size() +
               vulkan_map_source_detail::BindingSuffix.size() +
               vulkan_map_source_detail::CheckLinePrefix.size() +
               vulkan_map_source_detail::CheckLineOffset.size() +
               vulkan_map_source_detail::CheckLineStride.size() +
               vulkan_map_source_detail::CheckLineLimit.size() +
               vulkan_map_source_detail::CheckLineSuffix.size(),
           item) ||
      !add(bytes, item, bytes) ||
      !add(bytes, vulkan_map_source_detail::BindingPrefix.size(), bytes) ||
      !add(bytes, VulkanDecimalDigitCount(check_count + 2u), bytes) ||
      !add(bytes, vulkan_map_source_detail::StatusMiddle.size(), bytes) ||
      !add(bytes, vulkan_map_source_detail::CheckBody.size(), bytes) ||
      !add(bytes, vulkan_map_source_detail::CheckTail.size(), bytes) ||
      !add(bytes, offset_digit_bytes, bytes) ||
      !add(bytes, stride_digit_bytes, bytes) ||
      !add(bytes, limit_digit_bytes, bytes)) {
    return false;
  }
  for (std::uint64_t index = 0u; index < check_count; ++index) {
    if (!add(bytes, 3u * VulkanDecimalDigitCount(index), bytes) ||
        !add(bytes, VulkanDecimalDigitCount(index + 2u), bytes)) {
      return false;
    }
  }
  upper = bytes;
  return true;
}

[[nodiscard]] bool
VulkanMapCheckSourceUpperBytes(const rund::kernel::LoweringArtifact &artifact,
                               std::uint64_t &upper) noexcept {
  std::uint64_t check_count = 0u;
  std::uint64_t limit_digits = 0u;
  for (std::size_t index = 0u; index < artifact.metadata.read_routes.size();
       ++index) {
    bool first = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (artifact.metadata.read_routes[prior].index ==
          artifact.metadata.read_routes[index].index) {
        first = false;
        break;
      }
    }
    if (!first) {
      continue;
    }
    if (!rund::kernel::checked::add(
            limit_digits,
            VulkanDecimalDigitCount(artifact.metadata.read_routes[index].count),
            limit_digits) ||
        !rund::kernel::checked::add(check_count, 1u, check_count)) {
      return false;
    }
  }
  std::uint64_t offset_digits = 0u;
  std::uint64_t stride_digits = 0u;
  return rund::kernel::checked::mul(check_count, 20u, offset_digits) &&
         rund::kernel::checked::mul(check_count, 20u, stride_digits) &&
         VulkanMapCheckSourceUpperBytes(check_count, offset_digits,
                                        stride_digits, limit_digits, upper);
}

} // namespace rund::node::accel::detail
