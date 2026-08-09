#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/core/model.hpp>
#include <kernel/program/compute/model.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace rund::node::accel::detail {

// Range is primitive-neutral. Adapters project their semantic descriptors into
// this sole algebra-and-shape planning authority.
enum class RangeOp : std::uint8_t {
  Sum,
  Minimum,
  Maximum,
};

enum class RangeBoundary : std::uint8_t {
  Clamp,
  Clip,
};

enum class RangeCount : std::uint8_t {
  Descriptor,
  U32,
  U64,
};

enum class RangeLaw : std::uint8_t {
  ModuloWidth,
  Saturating,
  OrderOnly,
};

enum class RangeSource : std::uint8_t {
  Unavailable,
  Cpu,
  Metal,
  Vulkan,
};

enum class RangeCapsKind : std::uint8_t {
  Unavailable,
  Cpu,
  Gpu,
};

enum class RangePath : std::uint8_t {
  Direct,
  SharedHalo,
  PrefixDifference,
  BlockPrefixSuffix,
};

enum class RangePlanKind : std::uint8_t {
  Rejected,
  Selected,
};

enum class RangeStageKind : std::uint8_t {
  Direct,
  SharedHalo,
  PrefixSequential,
  PrefixBlock,
  PrefixSummary,
  PrefixFixup,
  PrefixWindow,
  BlockPrefixSuffix,
  BlockWindow,
};

enum class RangeTempRole : std::uint8_t {
  PrefixValues,
  BlockSummaries,
  ForwardValues,
  BackwardValues,
};

enum class RangeSupport : std::uint8_t {
  Direct = 1u << 0u,
  SharedHalo = 1u << 1u,
  PrefixDifference = 1u << 2u,
  BlockPrefixSuffix = 1u << 3u,
};

[[nodiscard]] constexpr std::uint8_t
RangeSupportBit(const RangeSupport support) noexcept {
  return static_cast<std::uint8_t>(support);
}

inline constexpr std::uint8_t kRangeKnownSupportMask =
    RangeSupportBit(RangeSupport::Direct) |
    RangeSupportBit(RangeSupport::SharedHalo) |
    RangeSupportBit(RangeSupport::PrefixDifference) |
    RangeSupportBit(RangeSupport::BlockPrefixSuffix);

inline constexpr std::uint8_t kRangeWidth64Bit = 1u << 0u;
inline constexpr std::uint8_t kRangeWidth128Bit = 1u << 1u;
inline constexpr std::uint8_t kRangeWidth256Bit = 1u << 2u;
inline constexpr std::uint8_t kRangeKnownWidthMask =
    kRangeWidth64Bit | kRangeWidth128Bit | kRangeWidth256Bit;
inline constexpr std::array<rund::kernel::u32, 3u> kRangeWidths{64u, 128u,
                                                                256u};
inline constexpr rund::kernel::u32 kRangeCpuBlockWidth = 64u;
// This is an integer shared-memory reserve policy, not a claim about physical
// resident workgroups.  Backends feed the actual per-workgroup limit into the
// capability value and the planner requires q * declared shared bytes to fit.
inline constexpr rund::kernel::u32 kRangeSharedReserve = 4u;

struct RangeIdentity final {
  std::uint64_t hi{};
  std::uint64_t lo{};

  [[nodiscard]] friend constexpr bool
  operator==(const RangeIdentity &, const RangeIdentity &) = default;
};

class RangeTraits final {
public:
  RangeTraits() = delete;

  [[nodiscard]] static constexpr std::optional<RangeTraits>
  make(const RangeOp operation, const rund::kernel::ComputeDomain domain,
       const RangeLaw arithmetic_law) noexcept {
    return KnownOperation(operation) && KnownDomain(domain) &&
                   CompatibleArithmeticLaw(operation, domain, arithmetic_law)
               ? std::optional<RangeTraits>{RangeTraits{operation, domain,
                                                        arithmetic_law}}
               : std::nullopt;
  }

  [[nodiscard]] static constexpr std::optional<RangeTraits>
  sum_modulo(const rund::kernel::ComputeDomain domain) noexcept {
    return make(RangeOp::Sum, domain, RangeLaw::ModuloWidth);
  }

  [[nodiscard]] static constexpr std::optional<RangeTraits>
  sum_saturating(const rund::kernel::ComputeDomain domain) noexcept {
    return make(RangeOp::Sum, domain, RangeLaw::Saturating);
  }

  [[nodiscard]] static constexpr std::optional<RangeTraits>
  minimum(const rund::kernel::ComputeDomain domain) noexcept {
    return make(RangeOp::Minimum, domain, RangeLaw::OrderOnly);
  }

  [[nodiscard]] static constexpr std::optional<RangeTraits>
  maximum(const rund::kernel::ComputeDomain domain) noexcept {
    return make(RangeOp::Maximum, domain, RangeLaw::OrderOnly);
  }

  [[nodiscard]] constexpr RangeOp operation() const noexcept {
    return operation_;
  }

  [[nodiscard]] constexpr rund::kernel::ComputeDomain domain() const noexcept {
    return domain_;
  }

  [[nodiscard]] constexpr RangeLaw arithmetic_law() const noexcept {
    return arithmetic_law_;
  }

  [[nodiscard]] constexpr bool associative() const noexcept {
    return operation_ != RangeOp::Sum ||
           arithmetic_law_ == RangeLaw::ModuloWidth;
  }
  [[nodiscard]] constexpr bool has_identity() const noexcept { return true; }
  [[nodiscard]] constexpr bool commutative() const noexcept { return true; }

  [[nodiscard]] constexpr bool invertible() const noexcept {
    return operation_ == RangeOp::Sum && associative();
  }

  [[nodiscard]] constexpr bool idempotent() const noexcept {
    return operation_ == RangeOp::Minimum || operation_ == RangeOp::Maximum;
  }

  [[nodiscard]] constexpr bool signed_domain() const noexcept {
    return domain_ == rund::kernel::ComputeDomain::I32 ||
           domain_ == rund::kernel::ComputeDomain::I64 ||
           domain_ == rund::kernel::ComputeDomain::Fixed;
  }

  [[nodiscard]] constexpr bool unsigned_domain() const noexcept {
    return domain_ == rund::kernel::ComputeDomain::U32 ||
           domain_ == rund::kernel::ComputeDomain::U64;
  }

  [[nodiscard]] constexpr bool fixed_domain() const noexcept {
    return domain_ == rund::kernel::ComputeDomain::Fixed;
  }

  [[nodiscard]] constexpr bool ordered() const noexcept {
    return operation_ == RangeOp::Minimum || operation_ == RangeOp::Maximum;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return KnownOperation(operation_) && KnownDomain(domain_) &&
           CompatibleArithmeticLaw(operation_, domain_, arithmetic_law_);
  }

private:
  [[nodiscard]] static constexpr bool
  KnownOperation(const RangeOp operation) noexcept {
    return operation == RangeOp::Sum || operation == RangeOp::Minimum ||
           operation == RangeOp::Maximum;
  }

  [[nodiscard]] static constexpr bool
  KnownDomain(const rund::kernel::ComputeDomain domain) noexcept {
    return domain == rund::kernel::ComputeDomain::I32 ||
           domain == rund::kernel::ComputeDomain::U32 ||
           domain == rund::kernel::ComputeDomain::I64 ||
           domain == rund::kernel::ComputeDomain::U64 ||
           domain == rund::kernel::ComputeDomain::Fixed;
  }

  [[nodiscard]] static constexpr bool
  CompatibleArithmeticLaw(const RangeOp operation,
                          const rund::kernel::ComputeDomain domain,
                          const RangeLaw arithmetic_law) noexcept {
    if (operation == RangeOp::Sum) {
      return arithmetic_law == RangeLaw::ModuloWidth ||
             (arithmetic_law == RangeLaw::Saturating &&
              (domain == rund::kernel::ComputeDomain::I32 ||
               domain == rund::kernel::ComputeDomain::I64 ||
               domain == rund::kernel::ComputeDomain::Fixed));
    }
    return (operation == RangeOp::Minimum || operation == RangeOp::Maximum) &&
           arithmetic_law == RangeLaw::OrderOnly;
  }

  constexpr RangeTraits(const RangeOp operation,
                        const rund::kernel::ComputeDomain domain,
                        const RangeLaw arithmetic_law) noexcept
      : operation_(operation), domain_(domain),
        arithmetic_law_(arithmetic_law) {}

  RangeOp operation_;
  rund::kernel::ComputeDomain domain_;
  RangeLaw arithmetic_law_;
};

class RangeShape final {
public:
  RangeShape() = delete;

  // Output q aggregates K logical positions beginning at q*S-P. Clamp
  // repeats an endpoint outside [0,N); Clip excludes those positions. P<K
  // makes the first window intersect the input; the checked last-start test
  // below makes every intervening window intersect as well.
  [[nodiscard]] static constexpr std::optional<RangeShape>
  affine(const RangeTraits traits, const RangeBoundary boundary,
         const rund::kernel::u64 input_count,
         const rund::kernel::u64 output_count,
         const rund::kernel::u64 window_size, const rund::kernel::u64 stride,
         const rund::kernel::u64 padding, const rund::kernel::u32 element_bytes,
         const RangeCount count = RangeCount::Descriptor) noexcept {
    const rund::kernel::u128 last_anchor =
        static_cast<rund::kernel::u128>(
            output_count == 0u ? 0u : output_count - 1u) *
        stride;
    if (!traits.valid() || !KnownBoundary(boundary) || !KnownCount(count) ||
        input_count == 0u || output_count == 0u || window_size == 0u ||
        stride == 0u || padding >= window_size ||
        (count != RangeCount::Descriptor &&
         (input_count != output_count || stride != 1u ||
          window_size - padding - 1u != padding)) ||
        last_anchor > std::numeric_limits<rund::kernel::u64>::max() ||
        last_anchor >= static_cast<rund::kernel::u128>(input_count) + padding ||
        (element_bytes != 4u && element_bytes != 8u) ||
        !DomainWidthMatches(traits.domain(), element_bytes) ||
        input_count >
            std::numeric_limits<rund::kernel::u64>::max() / element_bytes ||
        output_count >
            std::numeric_limits<rund::kernel::u64>::max() / element_bytes) {
      return std::nullopt;
    }
    return RangeShape{traits,       boundary,      input_count,
                      output_count, window_size,   stride,
                      padding,      element_bytes, count};
  }

  [[nodiscard]] constexpr const RangeTraits &traits() const noexcept {
    return traits_;
  }

  [[nodiscard]] constexpr RangeBoundary boundary() const noexcept {
    return boundary_;
  }

  [[nodiscard]] constexpr RangeCount count() const noexcept {
    return static_cast<RangeCount>((element_and_count_ >> 8u) & 0xffu);
  }

  [[nodiscard]] constexpr bool resident_counted() const noexcept {
    return count() != RangeCount::Descriptor;
  }

  [[nodiscard]] constexpr rund::kernel::u64 input_count() const noexcept {
    return input_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 output_count() const noexcept {
    return output_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 window_size() const noexcept {
    return window_size_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 stride() const noexcept {
    return stride_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 padding() const noexcept {
    return padding_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 right_extent() const noexcept {
    return window_size_ - padding_ - 1u;
  }

  [[nodiscard]] constexpr std::optional<rund::kernel::u64>
  affine_span() const noexcept {
    rund::kernel::u64 last_anchor = 0u;
    rund::kernel::u64 span = 0u;
    return rund::kernel::checked::mul(output_count_ - 1u, stride_,
                                      last_anchor) &&
                   rund::kernel::checked::add(last_anchor, window_size_, span)
               ? std::optional<rund::kernel::u64>{span}
               : std::nullopt;
  }

  [[nodiscard]] constexpr bool centered_clamp() const noexcept {
    return boundary_ == RangeBoundary::Clamp && input_count_ == output_count_ &&
           stride_ == 1u && right_extent() == padding_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 element_bytes() const noexcept {
    return element_and_count_ & 0xffu;
  }

  [[nodiscard]] constexpr rund::kernel::u64 payload_bytes() const noexcept {
    return input_count_ * element_bytes();
  }

  [[nodiscard]] constexpr rund::kernel::u64 output_bytes() const noexcept {
    return output_count_ * element_bytes();
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return affine(traits_, boundary_, input_count_, output_count_, window_size_,
                  stride_, padding_, element_bytes(), count())
        .has_value();
  }

private:
  [[nodiscard]] static constexpr bool
  KnownBoundary(const RangeBoundary boundary) noexcept {
    return boundary == RangeBoundary::Clamp || boundary == RangeBoundary::Clip;
  }

  [[nodiscard]] static constexpr bool
  KnownCount(const RangeCount count) noexcept {
    return count == RangeCount::Descriptor || count == RangeCount::U32 ||
           count == RangeCount::U64;
  }

  [[nodiscard]] static constexpr bool
  DomainWidthMatches(const rund::kernel::ComputeDomain domain,
                     const rund::kernel::u32 element_bytes) noexcept {
    if (domain == rund::kernel::ComputeDomain::I32 ||
        domain == rund::kernel::ComputeDomain::U32) {
      return element_bytes == 4u;
    }
    if (domain == rund::kernel::ComputeDomain::I64 ||
        domain == rund::kernel::ComputeDomain::U64) {
      return element_bytes == 8u;
    }
    return domain == rund::kernel::ComputeDomain::Fixed;
  }

  constexpr RangeShape(const RangeTraits traits, const RangeBoundary boundary,
                       const rund::kernel::u64 input_count,
                       const rund::kernel::u64 output_count,
                       const rund::kernel::u64 window_size,
                       const rund::kernel::u64 stride,
                       const rund::kernel::u64 padding,
                       const rund::kernel::u32 element_bytes,
                       const RangeCount count) noexcept
      : traits_(traits), boundary_(boundary),
        element_and_count_(element_bytes |
                           (static_cast<rund::kernel::u32>(count) << 8u)),
        input_count_(input_count), output_count_(output_count),
        window_size_(window_size), stride_(stride), padding_(padding) {}

  RangeTraits traits_;
  RangeBoundary boundary_;
  rund::kernel::u32 element_and_count_;
  rund::kernel::u64 input_count_;
  rund::kernel::u64 output_count_;
  rund::kernel::u64 window_size_;
  rund::kernel::u64 stride_;
  rund::kernel::u64 padding_;
};

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
inline constexpr std::size_t kRangeCandidateCap = 12u;

// This derives physical prefix stages and temporary lifetimes. It does not
// choose an aggregate algorithm: Range selects PrefixDifference, and
// native Scan projects its own observable-prefix semantics into the flat form.
enum class RangePrefixKind : std::uint8_t {
  Rejected,
  Hierarchical,
  FlatBlockTotals,
};

namespace range_prefix_detail {
class Builder;
}

class RangePrefixExec final {
public:
  RangePrefixExec() = delete;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return disposition_ != RangePrefixKind::Rejected;
  }

  [[nodiscard]] constexpr RangePrefixKind disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr const char *reason() const noexcept {
    return reason_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 width() const noexcept {
    return selection().width;
  }

  [[nodiscard]] constexpr std::size_t stage_count() const noexcept {
    return selection().stage_count;
  }

  [[nodiscard]] constexpr RangeStagePlan
  stage(const std::size_t index) const noexcept {
    assert(index < stage_count());
    return selection().stages[index];
  }

  [[nodiscard]] constexpr std::size_t temporary_count() const noexcept {
    return selection().temporary_count;
  }

  [[nodiscard]] constexpr RangeTempReq
  temporary(const std::size_t index) const noexcept {
    assert(index < temporary_count());
    return selection().temporaries[index];
  }

private:
  struct Selection final {
    rund::kernel::u32 width{};
    std::array<RangeStagePlan, kRangeStageCap> stages{};
    std::size_t stage_count{};
    std::array<RangeTempReq, kRangeTempCap> temporaries{};
    std::size_t temporary_count{};
  };

  friend class range_prefix_detail::Builder;

  [[nodiscard]] static constexpr RangePrefixExec
  rejected(const char *const reason) noexcept {
    return RangePrefixExec{RangePrefixKind::Rejected, reason};
  }

  [[nodiscard]] static constexpr RangePrefixExec
  selected(const RangePrefixKind disposition,
           const Selection selection) noexcept {
    return RangePrefixExec{disposition, selection};
  }

  [[nodiscard]] constexpr const Selection &selection() const noexcept {
    assert(ok() && selection_.has_value());
    return *selection_;
  }

  constexpr RangePrefixExec(const RangePrefixKind disposition,
                            const char *const reason) noexcept
      : disposition_(disposition), reason_(reason) {}

  constexpr RangePrefixExec(const RangePrefixKind disposition,
                            const Selection selection) noexcept
      : disposition_(disposition), selection_(selection), reason_("ok") {}

  RangePrefixKind disposition_;
  std::optional<Selection> selection_{};
  const char *reason_;
};

namespace range_prefix_detail {

class Builder final {
public:
  constexpr explicit Builder(const rund::kernel::u32 width) noexcept
      : selection_{.width = width} {}

  [[nodiscard]] static constexpr RangePrefixExec rejected() noexcept {
    return RangePrefixExec::rejected("compute_range_aggregate_prefix_invalid");
  }

  [[nodiscard]] constexpr bool
  append_stage(const RangeStageKind disposition, const std::uint8_t level,
               const rund::kernel::u64 element_count,
               const rund::kernel::u64 groups,
               const rund::kernel::u32 width) noexcept {
    if (selection_.stage_count == selection_.stages.size() || groups == 0u) {
      return false;
    }
    selection_.stages[selection_.stage_count++] = RangeStagePlan{
        .disposition = disposition,
        .level = level,
        .element_count = element_count,
        .groups = groups,
        .width = width,
    };
    return true;
  }

  [[nodiscard]] constexpr bool append_temporary(
      const RangeTempRole role, const std::uint8_t ordinal,
      const rund::kernel::u64 bytes, const rund::kernel::u32 alignment,
      const std::uint8_t first_stage, const std::uint8_t last_stage) noexcept {
    if (selection_.temporary_count == selection_.temporaries.size() ||
        bytes == 0u || alignment == 0u || first_stage > last_stage) {
      return false;
    }
    selection_.temporaries[selection_.temporary_count++] =
        RangeTempReq{.role = role,
                     .ordinal = ordinal,
                     .bytes = bytes,
                     .alignment = alignment,
                     .first_stage = first_stage,
                     .last_stage = last_stage};
    return true;
  }

  [[nodiscard]] constexpr std::size_t temporary_count() const noexcept {
    return selection_.temporary_count;
  }

  [[nodiscard]] constexpr RangeStagePlan
  stage(const std::size_t index) const noexcept {
    assert(index < selection_.stage_count);
    return selection_.stages[index];
  }

  constexpr void set_last_stage(const std::size_t index,
                                const std::uint8_t last_stage) noexcept {
    assert(index < selection_.temporary_count);
    selection_.temporaries[index].last_stage = last_stage;
  }

  [[nodiscard]] constexpr RangePrefixExec
  finish(const RangePrefixKind disposition) const noexcept {
    return RangePrefixExec::selected(disposition, selection_);
  }

private:
  RangePrefixExec::Selection selection_;
};

[[nodiscard]] constexpr bool
valid_width(const rund::kernel::u32 width) noexcept {
  return width == 64u || width == 128u || width == 256u;
}

[[nodiscard]] constexpr rund::kernel::u64
groups(const rund::kernel::u64 count, const rund::kernel::u32 width) noexcept {
  return width == 0u ? 0u
                     : count / width +
                           static_cast<rund::kernel::u64>(count % width != 0u);
}

} // namespace range_prefix_detail

// Derives the work-efficient recursive prefix hierarchy used by the
// PrefixDifference candidate. Every hierarchy stage must fit the physical
// single-stage dispatch limit supplied by the backend capability projection.
[[nodiscard]] constexpr RangePrefixExec
PlanRangePrefixTree(const rund::kernel::u64 element_count,
                    const rund::kernel::u32 width,
                    const rund::kernel::u32 element_bytes,
                    const rund::kernel::u64 maximum_group_count) noexcept {
  using namespace range_prefix_detail;
  if (element_count == 0u || !valid_width(width) ||
      (element_bytes != 4u && element_bytes != 8u) ||
      maximum_group_count == 0u) {
    return Builder::rejected();
  }

  Builder builder{width};
  std::array<std::size_t, kRangeTempCap> summary_index{};
  std::size_t level_count = 0u;
  rund::kernel::u64 values = element_count;
  while (true) {
    if (level_count == summary_index.size()) {
      return Builder::rejected();
    }
    const rund::kernel::u64 group_count = groups(values, width);
    if (group_count == 0u || group_count > maximum_group_count ||
        !builder.append_stage(level_count == 0u ? RangeStageKind::PrefixBlock
                                                : RangeStageKind::PrefixSummary,
                              static_cast<std::uint8_t>(level_count), values,
                              group_count, width)) {
      return Builder::rejected();
    }
    ++level_count;
    if (group_count == 1u) {
      break;
    }
    rund::kernel::u64 bytes = 0u;
    if (!rund::kernel::checked::mul(group_count, element_bytes, bytes) ||
        !builder.append_temporary(
            RangeTempRole::BlockSummaries,
            static_cast<std::uint8_t>(level_count - 1u), bytes, element_bytes,
            static_cast<std::uint8_t>(level_count - 1u),
            static_cast<std::uint8_t>(level_count - 1u))) {
      return Builder::rejected();
    }
    summary_index[level_count - 1u] = builder.temporary_count() - 1u;
    values = group_count;
  }

  for (std::size_t level = level_count - 1u; level != 0u; --level) {
    const std::size_t child = level - 1u;
    const RangeStagePlan child_stage = builder.stage(child);
    if (!builder.append_stage(
            RangeStageKind::PrefixFixup, static_cast<std::uint8_t>(child),
            child_stage.element_count, child_stage.groups, width)) {
      return Builder::rejected();
    }
    builder.set_last_stage(
        summary_index[child],
        static_cast<std::uint8_t>(2u * level_count - 2u - child));
  }
  return builder.finish(RangePrefixKind::Hierarchical);
}

// Derives the native Scan physical graph: block-local prefixes, one flat
// block-total prefix, then final offset/materialization. The caller supplies
// the semantic block count; device dispatch chunking remains a backend command
// concern, so this records full logical block count rather than an
// adapter-specific chunk count.
[[nodiscard]] constexpr RangePrefixExec
PlanRangeFlatPrefix(const rund::kernel::u64 element_count,
                    const rund::kernel::u64 block_count,
                    const rund::kernel::u32 width,
                    const rund::kernel::u32 element_bytes) noexcept {
  using namespace range_prefix_detail;
  if (element_count == 0u || block_count == 0u || !valid_width(width) ||
      (element_bytes != 4u && element_bytes != 8u)) {
    return Builder::rejected();
  }
  rund::kernel::u64 summary_bytes = 0u;
  if (block_count == 0u ||
      !rund::kernel::checked::mul(block_count, element_bytes, summary_bytes)) {
    return Builder::rejected();
  }

  Builder builder{width};
  if (!builder.append_stage(RangeStageKind::PrefixBlock, 0u, element_count,
                            block_count, width) ||
      !builder.append_temporary(
          RangeTempRole::BlockSummaries, 0u, summary_bytes, element_bytes, 0u,
          static_cast<std::uint8_t>(block_count == 1u ? 0u : 2u))) {
    return Builder::rejected();
  }
  if (block_count == 1u) {
    return builder.finish(RangePrefixKind::FlatBlockTotals);
  }
  if (!builder.append_stage(RangeStageKind::PrefixSummary, 0u, block_count, 1u,
                            width) ||
      !builder.append_stage(RangeStageKind::PrefixFixup, 0u, element_count,
                            block_count, width)) {
    return Builder::rejected();
  }
  return builder.finish(RangePrefixKind::FlatBlockTotals);
}

class RangePlan final {
public:
  RangePlan() = delete;

  [[nodiscard]] static constexpr RangePlan
  rejected(const char *const reason) noexcept {
    return RangePlan{RangePlanKind::Rejected, reason};
  }

  [[nodiscard]] constexpr RangePlanKind disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr bool ok() const noexcept {
    return disposition_ == RangePlanKind::Selected;
  }

  [[nodiscard]] constexpr const char *reason() const noexcept {
    return reason_;
  }

  [[nodiscard]] constexpr const RangeShape &shape() const noexcept {
    return selection().shape;
  }

  [[nodiscard]] constexpr const RangeCandidate &candidate() const noexcept {
    return selection().candidate;
  }

  [[nodiscard]] constexpr RangeSource source_variant() const noexcept {
    return selection().source_variant;
  }

  [[nodiscard]] constexpr const RangeCost &cost() const noexcept {
    return selection().cost;
  }

  [[nodiscard]] constexpr std::size_t stage_count() const noexcept {
    return selection().stage_count;
  }

  [[nodiscard]] constexpr RangeStagePlan
  stage(const std::size_t index) const noexcept {
    assert(index < stage_count());
    const Selection &selected = selection();
    const RangePath disposition = selected.candidate.disposition();
    if (disposition == RangePath::Direct) {
      return RangeStagePlan{.disposition = RangeStageKind::Direct,
                            .level = 0u,
                            .element_count = selected.shape.output_count(),
                            .groups =
                                selected.candidate.width() == 0u
                                    ? 1u
                                    : Groups(selected.shape.output_count(),
                                             selected.candidate.width()),
                            .width = selected.candidate.width()};
    }
    if (disposition == RangePath::SharedHalo) {
      return RangeStagePlan{.disposition = RangeStageKind::SharedHalo,
                            .level = 0u,
                            .element_count = selected.shape.input_count(),
                            .groups = Groups(selected.shape.input_count(),
                                             selected.candidate.width()),
                            .width = selected.candidate.width()};
    }
    if (disposition == RangePath::BlockPrefixSuffix) {
      if (selected.source_variant == RangeSource::Cpu) {
        const std::optional<rund::kernel::u64> span =
            selected.shape.affine_span();
        assert(span.has_value());
        return index == 0u
                   ? RangeStagePlan{.disposition =
                                        RangeStageKind::BlockPrefixSuffix,
                                    .level = 0u,
                                    .element_count = *span,
                                    .groups = 1u,
                                    .width = 0u}
                   : RangeStagePlan{.disposition = RangeStageKind::BlockWindow,
                                    .level = 0u,
                                    .element_count =
                                        selected.shape.output_count(),
                                    .groups = 1u,
                                    .width = 0u};
      }
      if (index == 0u) {
        const std::optional<rund::kernel::u64> span =
            selected.shape.affine_span();
        assert(span.has_value());
        const rund::kernel::u64 padded = *span;
        const rund::kernel::u64 window = selected.shape.window_size();
        const rund::kernel::u64 blocks = Groups(padded, window);
        return RangeStagePlan{.disposition = RangeStageKind::BlockPrefixSuffix,
                              .level = 0u,
                              .element_count = padded,
                              .groups =
                                  Groups(blocks, selected.candidate.width()),
                              .width = selected.candidate.width()};
      }
      return RangeStagePlan{.disposition = RangeStageKind::BlockWindow,
                            .level = 0u,
                            .element_count = selected.shape.output_count(),
                            .groups = Groups(selected.shape.output_count(),
                                             selected.candidate.width()),
                            .width = selected.candidate.width()};
    }

    if (selected.source_variant == RangeSource::Cpu) {
      return index == 0u
                 ? RangeStagePlan{.disposition =
                                      RangeStageKind::PrefixSequential,
                                  .level = 0u,
                                  .element_count = selected.shape.input_count(),
                                  .groups = 1u,
                                  .width = 0u}
                 : RangeStagePlan{.disposition = RangeStageKind::PrefixWindow,
                                  .level = 0u,
                                  .element_count =
                                      selected.shape.output_count(),
                                  .groups = 1u,
                                  .width = 0u};
    }
    const RangePrefixExec prefix = PlanRangePrefixTree(
        selected.shape.input_count(), selected.candidate.width(),
        selected.shape.element_bytes(),
        std::numeric_limits<rund::kernel::u64>::max());
    assert(prefix.ok() && selected.stage_count == prefix.stage_count() + 1u);
    if (index < prefix.stage_count()) {
      return prefix.stage(index);
    }
    return RangeStagePlan{.disposition = RangeStageKind::PrefixWindow,
                          .level = 0u,
                          .element_count = selected.shape.output_count(),
                          .groups = Groups(selected.shape.output_count(),
                                           selected.candidate.width()),
                          .width = selected.candidate.width()};
  }

  [[nodiscard]] constexpr std::size_t temporary_count() const noexcept {
    return selection().temporary_count;
  }

  [[nodiscard]] constexpr RangeTempReq
  temporary(const std::size_t index) const noexcept {
    assert(index < temporary_count());
    const Selection &selected = selection();
    if (selected.candidate.disposition() == RangePath::PrefixDifference) {
      if (index == 0u) {
        return RangeTempReq{
            .role = RangeTempRole::PrefixValues,
            .ordinal = 0u,
            .bytes = selected.shape.payload_bytes(),
            .alignment = selected.shape.element_bytes(),
            .first_stage = 0u,
            .last_stage = static_cast<std::uint8_t>(selected.stage_count - 1u)};
      }
      assert(selected.source_variant != RangeSource::Cpu);
      const RangePrefixExec prefix = PlanRangePrefixTree(
          selected.shape.input_count(), selected.candidate.width(),
          selected.shape.element_bytes(),
          std::numeric_limits<rund::kernel::u64>::max());
      assert(prefix.ok() && index - 1u < prefix.temporary_count() &&
             selected.temporary_count == prefix.temporary_count() + 1u);
      return prefix.temporary(index - 1u);
    }
    const std::optional<rund::kernel::u64> span = selected.shape.affine_span();
    assert(span.has_value());
    const rund::kernel::u64 bytes = *span * selected.shape.element_bytes();
    return RangeTempReq{.role = index == 0u ? RangeTempRole::ForwardValues
                                            : RangeTempRole::BackwardValues,
                        .ordinal = 0u,
                        .bytes = bytes,
                        .alignment = selected.shape.element_bytes(),
                        .first_stage = 0u,
                        .last_stage = 1u};
  }

  [[nodiscard]] constexpr std::uint8_t legal_candidate_count() const noexcept {
    return selection().legal_candidate_count;
  }

  [[nodiscard]] constexpr std::uint8_t pareto_candidate_count() const noexcept {
    return selection().pareto_candidate_count;
  }

  [[nodiscard]] constexpr RangeIdentity source_identity() const noexcept {
    return selection().source_identity;
  }

  [[nodiscard]] constexpr RangeIdentity execution_identity() const noexcept {
    return selection().execution_identity;
  }

private:
  friend constexpr RangePlan PlanRange(const RangeShape &,
                                       const RangeCaps &) noexcept;

  [[nodiscard]] static constexpr RangePlan
  selected(const RangeShape shape, const RangeCandidate candidate,
           const RangeSource source_variant, const RangeCost cost,
           const std::size_t stage_count, const std::size_t temporary_count,
           const std::uint8_t legal_candidate_count,
           const std::uint8_t pareto_candidate_count,
           const RangeIdentity source_identity,
           const RangeIdentity execution_identity) noexcept {
    return RangePlan{shape,
                     candidate,
                     source_variant,
                     cost,
                     stage_count,
                     temporary_count,
                     legal_candidate_count,
                     pareto_candidate_count,
                     source_identity,
                     execution_identity};
  }

  struct Selection final {
    RangeShape shape;
    RangeCandidate candidate;
    RangeSource source_variant{};
    RangeCost cost{};
    std::size_t stage_count{};
    std::size_t temporary_count{};
    std::uint8_t legal_candidate_count{};
    std::uint8_t pareto_candidate_count{};
    RangeIdentity source_identity{};
    RangeIdentity execution_identity{};
  };

  [[nodiscard]] constexpr const Selection &selection() const noexcept {
    assert(disposition_ == RangePlanKind::Selected && selection_.has_value());
    return *selection_;
  }

  [[nodiscard]] static constexpr rund::kernel::u64
  Groups(const rund::kernel::u64 count,
         const rund::kernel::u64 width) noexcept {
    return width == 0u
               ? 0u
               : count / width +
                     static_cast<rund::kernel::u64>(count % width != 0u);
  }

  constexpr RangePlan(const RangePlanKind disposition,
                      const char *const reason) noexcept
      : disposition_(disposition), reason_(reason) {}

  constexpr RangePlan(const RangeShape shape, const RangeCandidate candidate,
                      const RangeSource source_variant, const RangeCost cost,
                      const std::size_t stage_count,
                      const std::size_t temporary_count,
                      const std::uint8_t legal_candidate_count,
                      const std::uint8_t pareto_candidate_count,
                      const RangeIdentity source_identity,
                      const RangeIdentity execution_identity) noexcept
      : disposition_(RangePlanKind::Selected),
        selection_(Selection{shape, candidate, source_variant, cost,
                             stage_count, temporary_count,
                             legal_candidate_count, pareto_candidate_count,
                             source_identity, execution_identity}),
        reason_("ok") {}

  RangePlanKind disposition_;
  std::optional<Selection> selection_{};
  const char *reason_{};
};

static_assert(std::is_trivially_copyable_v<RangeIdentity>);
static_assert(std::is_trivially_copyable_v<RangeCost>);
static_assert(std::is_trivially_copyable_v<RangeTempReq>);
static_assert(std::is_trivially_copyable_v<RangeStagePlan>);
static_assert(std::is_trivially_copyable_v<RangePrefixExec>);

} // namespace rund::node::accel::detail
