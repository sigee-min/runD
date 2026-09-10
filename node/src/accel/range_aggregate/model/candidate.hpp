#pragma once

#include "traits.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace rund::node::accel::detail {

class RangeCandidate final {
public:
  RangeCandidate() = delete;

  [[nodiscard]] static constexpr RangeCandidate direct_cpu() noexcept {
    return RangeCandidate{RangePath::Direct, 0u, 0u};
  }

  [[nodiscard]] static constexpr std::optional<RangeCandidate>
  direct_gpu(const rund::kernel::u32 width) noexcept {
    return Make(RangePath::Direct, width, 0u);
  }

  [[nodiscard]] static constexpr std::optional<RangeCandidate>
  shared_halo(const rund::kernel::u32 width,
              const rund::kernel::u32 radius_capacity) noexcept {
    return SupportedWidth(width) && radius_capacity != 0u &&
                   radius_capacity <= width
               ? std::optional<RangeCandidate>{RangeCandidate{
                     RangePath::SharedHalo, width, radius_capacity}}
               : std::nullopt;
  }

  [[nodiscard]] static constexpr std::optional<RangeCandidate>
  prefix_difference(const rund::kernel::u32 width) noexcept {
    return Make(RangePath::PrefixDifference, width, 0u);
  }

  [[nodiscard]] static constexpr std::optional<RangeCandidate>
  block_prefix_suffix(const rund::kernel::u32 width) noexcept {
    return Make(RangePath::BlockPrefixSuffix, width, 0u);
  }

  [[nodiscard]] static constexpr std::optional<RangeCandidate>
  tiled_difference(const rund::kernel::u32 width) noexcept {
    return Make(RangePath::TiledDifference, width, 0u);
  }

  [[nodiscard]] constexpr RangePath disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 width() const noexcept {
    return width_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 radius_capacity() const noexcept {
    return radius_capacity_;
  }

  [[nodiscard]] constexpr bool uses_shared_halo() const noexcept {
    return disposition_ == RangePath::SharedHalo;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const RangeCandidate &, const RangeCandidate &) = default;

private:
  [[nodiscard]] static constexpr bool
  SupportedWidth(const rund::kernel::u32 width) noexcept {
    return width == 64u || width == 128u || width == 256u;
  }

  [[nodiscard]] static constexpr std::optional<RangeCandidate>
  Make(const RangePath disposition, const rund::kernel::u32 width,
       const rund::kernel::u32 radius_capacity) noexcept {
    return SupportedWidth(width) ? std::optional<RangeCandidate>{RangeCandidate{
                                       disposition, width, radius_capacity}}
                                 : std::nullopt;
  }

  constexpr RangeCandidate(const RangePath disposition,
                           const rund::kernel::u32 width,
                           const rund::kernel::u32 radius_capacity) noexcept
      : disposition_(disposition), width_(width),
        radius_capacity_(radius_capacity) {}

  RangePath disposition_;
  rund::kernel::u32 width_;
  rund::kernel::u32 radius_capacity_;
};

struct RangeCost final {
  rund::kernel::u128 global_read_bytes{};
  rund::kernel::u128 global_write_bytes{};
  rund::kernel::u128 combine_ops{};
  rund::kernel::u128 inverse_ops{};
  rund::kernel::u128 scale_ops{};
  rund::kernel::u64 shared_bytes{};
  rund::kernel::u64 scratch_bytes{};
  rund::kernel::u64 dispatch_count{};
  rund::kernel::u64 workgroup_count{};
  rund::kernel::u128 launched_lanes{};

  [[nodiscard]] friend constexpr bool operator==(const RangeCost &,
                                                 const RangeCost &) = default;
};

struct RangeTempReq final {
  RangeTempRole role{};
  std::uint8_t ordinal{};
  rund::kernel::u64 bytes{};
  rund::kernel::u64 alignment{};
  std::uint8_t first_stage{};
  std::uint8_t last_stage{};

  [[nodiscard]] friend constexpr bool
  operator==(const RangeTempReq &, const RangeTempReq &) = default;
};

struct RangeStagePlan final {
  RangeStageKind disposition{};
  std::uint8_t level{};
  rund::kernel::u64 element_count{};
  rund::kernel::u64 groups{};
  rund::kernel::u32 width{};

  [[nodiscard]] friend constexpr bool
  operator==(const RangeStagePlan &, const RangeStagePlan &) = default;
};

// Width 64 needs at most eleven hierarchy levels for any admitted u64-sized
// payload. Prefix has one up stage per level, one fewer down stage, and one
// output stage. Fixed storage preserves allocation-free planning.
inline constexpr std::size_t kRangeStageCap = 24u;
inline constexpr std::size_t kRangeTempCap = 12u;
inline constexpr std::size_t kRangeCandidateCap = 15u;

} // namespace rund::node::accel::detail
