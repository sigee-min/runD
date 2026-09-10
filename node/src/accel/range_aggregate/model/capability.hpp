#pragma once

#include "traits.hpp"

#include <cstdint>
#include <limits>
#include <optional>

namespace rund::node::accel::detail {

class RangeCaps final {
public:
  RangeCaps() = delete;

  [[nodiscard]] static constexpr RangeCaps unavailable() noexcept {
    return RangeCaps{RangeCapsKind::Unavailable,
                     RangeSource::Unavailable,
                     0u,
                     0u,
                     0u,
                     0u,
                     0u,
                     0u,
                     0u,
                     0u};
  }

  [[nodiscard]] static constexpr RangeCaps cpu() noexcept {
    return RangeCaps{RangeCapsKind::Cpu,
                     RangeSource::Cpu,
                     0u,
                     0u,
                     0u,
                     0u,
                     0u,
                     std::numeric_limits<rund::kernel::u64>::max(),
                     std::numeric_limits<rund::kernel::u64>::max(),
                     RangeSupportBit(RangeSupport::Direct) |
                         RangeSupportBit(RangeSupport::PrefixDifference) |
                         RangeSupportBit(RangeSupport::BlockPrefixSuffix)};
  }

  [[nodiscard]] static constexpr RangeCaps cpu_reference() noexcept {
    return RangeCaps{RangeCapsKind::Cpu,
                     RangeSource::Cpu,
                     0u,
                     0u,
                     0u,
                     0u,
                     0u,
                     std::numeric_limits<rund::kernel::u64>::max(),
                     std::numeric_limits<rund::kernel::u64>::max(),
                     RangeSupportBit(RangeSupport::Direct)};
  }

  [[nodiscard]] static constexpr std::optional<RangeCaps>
  gpu(const RangeSource source_variant, const std::uint8_t legal_width_mask,
      const rund::kernel::u32 maximum_threads_per_workgroup,
      const rund::kernel::u32 shared_memory_occupancy_budget,
      const rund::kernel::u64 shared_memory_limit,
      const rund::kernel::u64 maximum_group_count,
      const rund::kernel::u64 maximum_storage_element_count,
      const std::uint8_t supported_candidates) noexcept {
    return gpu(source_variant, legal_width_mask, maximum_threads_per_workgroup,
               shared_memory_occupancy_budget, shared_memory_limit,
               maximum_group_count, maximum_storage_element_count,
               std::numeric_limits<rund::kernel::u64>::max(),
               supported_candidates);
  }

  [[nodiscard]] static constexpr std::optional<RangeCaps>
  gpu(const RangeSource source_variant, const std::uint8_t legal_width_mask,
      const rund::kernel::u32 maximum_threads_per_workgroup,
      const rund::kernel::u32 shared_memory_occupancy_budget,
      const rund::kernel::u64 shared_memory_limit,
      const rund::kernel::u64 maximum_group_count,
      const rund::kernel::u64 maximum_storage_element_count,
      const rund::kernel::u64 maximum_storage_binding_bytes,
      const std::uint8_t supported_candidates) noexcept {
    const RangeCaps capabilities{RangeCapsKind::Gpu,
                                 source_variant,
                                 legal_width_mask,
                                 maximum_threads_per_workgroup,
                                 shared_memory_occupancy_budget,
                                 shared_memory_limit,
                                 maximum_group_count,
                                 maximum_storage_element_count,
                                 maximum_storage_binding_bytes,
                                 supported_candidates};
    return capabilities.valid() ? std::optional<RangeCaps>{capabilities}
                                : std::nullopt;
  }

  [[nodiscard]] constexpr RangeSource source_variant() const noexcept {
    return source_variant_;
  }

  [[nodiscard]] constexpr RangeCapsKind disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr bool available() const noexcept {
    return disposition_ != RangeCapsKind::Unavailable;
  }

  [[nodiscard]] constexpr bool
  supports(const RangeSupport support) const noexcept {
    return (supported_candidates_ & RangeSupportBit(support)) != 0u;
  }

  [[nodiscard]] constexpr bool
  supports_width(const rund::kernel::u32 width) const noexcept {
    const std::uint8_t bit = width == 64u    ? kRangeWidth64Bit
                             : width == 128u ? kRangeWidth128Bit
                             : width == 256u ? kRangeWidth256Bit
                                             : 0u;
    return bit != 0u && width <= maximum_threads_per_workgroup_ &&
           (legal_width_mask_ & bit) != 0u;
  }

  [[nodiscard]] constexpr bool cpu_only() const noexcept {
    return disposition_ == RangeCapsKind::Cpu &&
           source_variant_ == RangeSource::Cpu && legal_width_mask_ == 0u &&
           maximum_threads_per_workgroup_ == 0u && maximum_group_count_ == 0u &&
           maximum_storage_element_count_ ==
               std::numeric_limits<rund::kernel::u64>::max() &&
           maximum_storage_binding_bytes_ ==
               std::numeric_limits<rund::kernel::u64>::max() &&
           (supported_candidates_ == RangeSupportBit(RangeSupport::Direct) ||
            supported_candidates_ ==
                (RangeSupportBit(RangeSupport::Direct) |
                 RangeSupportBit(RangeSupport::PrefixDifference) |
                 RangeSupportBit(RangeSupport::BlockPrefixSuffix)));
  }

  [[nodiscard]] constexpr std::uint8_t legal_width_mask() const noexcept {
    return legal_width_mask_;
  }

  [[nodiscard]] constexpr rund::kernel::u32
  maximum_threads_per_workgroup() const noexcept {
    return maximum_threads_per_workgroup_;
  }

  [[nodiscard]] constexpr rund::kernel::u32
  shared_memory_occupancy_budget() const noexcept {
    return shared_memory_occupancy_budget_;
  }

  [[nodiscard]] constexpr rund::kernel::u64
  shared_memory_limit() const noexcept {
    return shared_memory_limit_;
  }

  [[nodiscard]] constexpr rund::kernel::u64
  maximum_group_count() const noexcept {
    return maximum_group_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64
  maximum_storage_element_count() const noexcept {
    return maximum_storage_element_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64
  maximum_storage_binding_bytes() const noexcept {
    return maximum_storage_binding_bytes_;
  }

  [[nodiscard]] constexpr std::uint8_t supported_candidates() const noexcept {
    return supported_candidates_;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    if (disposition_ == RangeCapsKind::Unavailable) {
      return false;
    }
    if (cpu_only()) {
      return true;
    }
    const bool gpu_variant = source_variant_ == RangeSource::Metal ||
                             source_variant_ == RangeSource::Vulkan;
    const bool source_index_limit =
        source_variant_ != RangeSource::Vulkan ||
        maximum_storage_element_count_ <=
            std::numeric_limits<rund::kernel::u32>::max();
    return disposition_ == RangeCapsKind::Gpu && gpu_variant &&
           source_index_limit && legal_width_mask_ != 0u &&
           (legal_width_mask_ & ~kRangeKnownWidthMask) == 0u &&
           maximum_threads_per_workgroup_ >= 64u &&
           maximum_group_count_ != 0u && maximum_storage_element_count_ != 0u &&
           maximum_storage_binding_bytes_ != 0u &&
           (supported_candidates_ & RangeSupportBit(RangeSupport::Direct)) !=
               0u &&
           (supported_candidates_ & ~kRangeKnownSupportMask) == 0u;
  }

private:
  constexpr RangeCaps(const RangeCapsKind disposition,
                      const RangeSource source_variant,
                      const std::uint8_t legal_width_mask,
                      const rund::kernel::u32 maximum_threads_per_workgroup,
                      const rund::kernel::u32 shared_memory_occupancy_budget,
                      const rund::kernel::u64 shared_memory_limit,
                      const rund::kernel::u64 maximum_group_count,
                      const rund::kernel::u64 maximum_storage_element_count,
                      const rund::kernel::u64 maximum_storage_binding_bytes,
                      const std::uint8_t supported_candidates) noexcept
      : disposition_(disposition), source_variant_(source_variant),
        legal_width_mask_(legal_width_mask),
        maximum_threads_per_workgroup_(maximum_threads_per_workgroup),
        shared_memory_occupancy_budget_(shared_memory_occupancy_budget),
        shared_memory_limit_(shared_memory_limit),
        maximum_group_count_(maximum_group_count),
        maximum_storage_element_count_(maximum_storage_element_count),
        maximum_storage_binding_bytes_(maximum_storage_binding_bytes),
        supported_candidates_(supported_candidates) {}

  RangeCapsKind disposition_;
  RangeSource source_variant_;
  std::uint8_t legal_width_mask_;
  rund::kernel::u32 maximum_threads_per_workgroup_;
  rund::kernel::u32 shared_memory_occupancy_budget_;
  rund::kernel::u64 shared_memory_limit_;
  rund::kernel::u64 maximum_group_count_;
  rund::kernel::u64 maximum_storage_element_count_;
  rund::kernel::u64 maximum_storage_binding_bytes_;
  std::uint8_t supported_candidates_;
};

} // namespace rund::node::accel::detail
