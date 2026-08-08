#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/core/model.hpp>
#include <kernel/program/compute/model.hpp>
#include <kernel/program/compute/stencil/model.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace rund::node::accel::detail {

// RangeAggregate is primitive-neutral. Stencil is the first projection;
// Scan, pooling, and rolling aggregates can project the same algebra and
// window evidence without acquiring a second algorithm-selection authority.
enum class RangeAggregateOperation : std::uint8_t {
  Sum,
  Minimum,
  Maximum,
};

enum class RangeAggregateBoundary : std::uint8_t {
  Clamp,
};

enum class RangeAggregateArithmeticLaw : std::uint8_t {
  ModuloWidth,
  Saturating,
  OrderOnly,
};

enum class RangeAggregateSourceVariant : std::uint8_t {
  Unavailable,
  Cpu,
  Metal,
  Vulkan,
};

enum class RangeAggregateCapabilityDisposition : std::uint8_t {
  Unavailable,
  Cpu,
  Gpu,
};

enum class RangeAggregateCandidateDisposition : std::uint8_t {
  Direct,
  SharedHalo,
  PrefixDifference,
  BlockPrefixSuffix,
};

enum class RangeAggregatePlanDisposition : std::uint8_t {
  Rejected,
  Selected,
};

enum class RangeAggregateStageDisposition : std::uint8_t {
  Direct,
  SharedHalo,
  PrefixBlock,
  PrefixSummary,
  PrefixFixup,
  PrefixWindow,
  BlockPrefixSuffix,
  BlockWindow,
};

enum class RangeTemporaryRole : std::uint8_t {
  PrefixValues,
  BlockSummaries,
  ForwardValues,
  BackwardValues,
};

enum class RangeAggregateSupport : std::uint8_t {
  Direct = 1u << 0u,
  SharedHalo = 1u << 1u,
  PrefixDifference = 1u << 2u,
  BlockPrefixSuffix = 1u << 3u,
};

[[nodiscard]] constexpr std::uint8_t
RangeAggregateSupportBit(const RangeAggregateSupport support) noexcept {
  return static_cast<std::uint8_t>(support);
}

inline constexpr std::uint8_t kRangeAggregateKnownSupportMask =
    RangeAggregateSupportBit(RangeAggregateSupport::Direct) |
    RangeAggregateSupportBit(RangeAggregateSupport::SharedHalo) |
    RangeAggregateSupportBit(RangeAggregateSupport::PrefixDifference) |
    RangeAggregateSupportBit(RangeAggregateSupport::BlockPrefixSuffix);

inline constexpr std::uint8_t kRangeAggregateWidth64Bit = 1u << 0u;
inline constexpr std::uint8_t kRangeAggregateWidth128Bit = 1u << 1u;
inline constexpr std::uint8_t kRangeAggregateWidth256Bit = 1u << 2u;
inline constexpr std::uint8_t kRangeAggregateKnownWidthMask =
    kRangeAggregateWidth64Bit | kRangeAggregateWidth128Bit |
    kRangeAggregateWidth256Bit;
inline constexpr std::array<rund::kernel::u32, 3u>
    kRangeAggregateWorkgroupWidths{64u, 128u, 256u};
// This is an integer shared-memory reserve policy, not a claim about physical
// resident workgroups.  Backends feed the actual per-workgroup limit into the
// capability value and the planner requires q * declared shared bytes to fit.
inline constexpr rund::kernel::u32 kRangeAggregateSharedMemoryReserve = 4u;

struct RangeAggregateIdentity final {
  std::uint64_t hi{};
  std::uint64_t lo{};

  [[nodiscard]] friend constexpr bool
  operator==(const RangeAggregateIdentity &,
             const RangeAggregateIdentity &) = default;
};

class RangeAggregateTraits final {
public:
  RangeAggregateTraits() = delete;

  [[nodiscard]] static constexpr std::optional<RangeAggregateTraits>
  make(const RangeAggregateOperation operation,
       const rund::kernel::ComputeDomain domain,
       const RangeAggregateArithmeticLaw arithmetic_law) noexcept {
    return KnownOperation(operation) && KnownDomain(domain) &&
                   CompatibleArithmeticLaw(operation, arithmetic_law)
               ? std::optional<RangeAggregateTraits>{RangeAggregateTraits{
                     operation, domain, arithmetic_law}}
               : std::nullopt;
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateTraits>
  sum_modulo(const rund::kernel::ComputeDomain domain) noexcept {
    return make(RangeAggregateOperation::Sum, domain,
                RangeAggregateArithmeticLaw::ModuloWidth);
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateTraits>
  sum_saturating(const rund::kernel::ComputeDomain domain) noexcept {
    return make(RangeAggregateOperation::Sum, domain,
                RangeAggregateArithmeticLaw::Saturating);
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateTraits>
  minimum(const rund::kernel::ComputeDomain domain) noexcept {
    return make(RangeAggregateOperation::Minimum, domain,
                RangeAggregateArithmeticLaw::OrderOnly);
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateTraits>
  maximum(const rund::kernel::ComputeDomain domain) noexcept {
    return make(RangeAggregateOperation::Maximum, domain,
                RangeAggregateArithmeticLaw::OrderOnly);
  }

  [[nodiscard]] constexpr RangeAggregateOperation operation() const noexcept {
    return operation_;
  }

  [[nodiscard]] constexpr rund::kernel::ComputeDomain domain() const noexcept {
    return domain_;
  }

  [[nodiscard]] constexpr RangeAggregateArithmeticLaw
  arithmetic_law() const noexcept {
    return arithmetic_law_;
  }

  [[nodiscard]] constexpr bool associative() const noexcept {
    return operation_ != RangeAggregateOperation::Sum ||
           arithmetic_law_ == RangeAggregateArithmeticLaw::ModuloWidth;
  }
  [[nodiscard]] constexpr bool has_identity() const noexcept { return true; }
  [[nodiscard]] constexpr bool commutative() const noexcept { return true; }

  [[nodiscard]] constexpr bool invertible() const noexcept {
    return operation_ == RangeAggregateOperation::Sum && associative();
  }

  [[nodiscard]] constexpr bool idempotent() const noexcept {
    return operation_ == RangeAggregateOperation::Minimum ||
           operation_ == RangeAggregateOperation::Maximum;
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
    return operation_ == RangeAggregateOperation::Minimum ||
           operation_ == RangeAggregateOperation::Maximum;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return KnownOperation(operation_) && KnownDomain(domain_) &&
           CompatibleArithmeticLaw(operation_, arithmetic_law_);
  }

private:
  [[nodiscard]] static constexpr bool
  KnownOperation(const RangeAggregateOperation operation) noexcept {
    return operation == RangeAggregateOperation::Sum ||
           operation == RangeAggregateOperation::Minimum ||
           operation == RangeAggregateOperation::Maximum;
  }

  [[nodiscard]] static constexpr bool
  KnownDomain(const rund::kernel::ComputeDomain domain) noexcept {
    return domain == rund::kernel::ComputeDomain::I32 ||
           domain == rund::kernel::ComputeDomain::U32 ||
           domain == rund::kernel::ComputeDomain::I64 ||
           domain == rund::kernel::ComputeDomain::U64 ||
           domain == rund::kernel::ComputeDomain::Fixed;
  }

  [[nodiscard]] static constexpr bool CompatibleArithmeticLaw(
      const RangeAggregateOperation operation,
      const RangeAggregateArithmeticLaw arithmetic_law) noexcept {
    if (operation == RangeAggregateOperation::Sum) {
      return arithmetic_law == RangeAggregateArithmeticLaw::ModuloWidth ||
             arithmetic_law == RangeAggregateArithmeticLaw::Saturating;
    }
    return (operation == RangeAggregateOperation::Minimum ||
            operation == RangeAggregateOperation::Maximum) &&
           arithmetic_law == RangeAggregateArithmeticLaw::OrderOnly;
  }

  constexpr RangeAggregateTraits(
      const RangeAggregateOperation operation,
      const rund::kernel::ComputeDomain domain,
      const RangeAggregateArithmeticLaw arithmetic_law) noexcept
      : operation_(operation), domain_(domain),
        arithmetic_law_(arithmetic_law) {}

  RangeAggregateOperation operation_;
  rund::kernel::ComputeDomain domain_;
  RangeAggregateArithmeticLaw arithmetic_law_;
};

class RangeAggregateShape final {
public:
  RangeAggregateShape() = delete;

  [[nodiscard]] static constexpr std::optional<RangeAggregateShape>
  window(const RangeAggregateTraits traits,
         const RangeAggregateBoundary boundary,
         const rund::kernel::u64 element_count, const rund::kernel::u64 radius,
         const rund::kernel::u32 element_bytes) noexcept {
    if (!traits.valid() || boundary != RangeAggregateBoundary::Clamp ||
        element_count == 0u || radius == 0u || radius > element_count ||
        (element_bytes != 4u && element_bytes != 8u) ||
        !DomainWidthMatches(traits.domain(), element_bytes) ||
        element_count >
            std::numeric_limits<rund::kernel::u64>::max() / element_bytes) {
      return std::nullopt;
    }
    return RangeAggregateShape{traits, boundary, element_count, radius,
                               element_bytes};
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateShape>
  from_stencil(const rund::kernel::StencilPlan &plan,
               const rund::kernel::ComputeDomain domain) noexcept {
    const std::optional<RangeAggregateOperation> operation =
        OperationFor(plan.op);
    if (!plan.ok || !operation.has_value() ||
        plan.boundary != rund::kernel::StencilBoundary::Clamp) {
      return std::nullopt;
    }
    const std::optional<RangeAggregateTraits> traits =
        *operation == RangeAggregateOperation::Sum
            ? RangeAggregateTraits::sum_modulo(domain)
            : RangeAggregateTraits::make(
                  *operation, domain, RangeAggregateArithmeticLaw::OrderOnly);
    if (!traits.has_value()) {
      return std::nullopt;
    }
    return window(*traits, RangeAggregateBoundary::Clamp, plan.element_count,
                  plan.radius,
                  static_cast<rund::kernel::u32>(plan.element_bytes));
  }

  [[nodiscard]] constexpr const RangeAggregateTraits &traits() const noexcept {
    return traits_;
  }

  [[nodiscard]] constexpr RangeAggregateBoundary boundary() const noexcept {
    return boundary_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 element_count() const noexcept {
    return element_count_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 radius() const noexcept {
    return radius_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 element_bytes() const noexcept {
    return element_bytes_;
  }

  [[nodiscard]] constexpr rund::kernel::u64 payload_bytes() const noexcept {
    return element_count_ * element_bytes_;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return traits_.valid() && boundary_ == RangeAggregateBoundary::Clamp &&
           element_count_ != 0u && radius_ != 0u && radius_ <= element_count_ &&
           (element_bytes_ == 4u || element_bytes_ == 8u) &&
           DomainWidthMatches(traits_.domain(), element_bytes_) &&
           element_count_ <=
               std::numeric_limits<rund::kernel::u64>::max() / element_bytes_;
  }

private:
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

  [[nodiscard]] static constexpr std::optional<RangeAggregateOperation>
  OperationFor(const rund::kernel::StencilOp operation) noexcept {
    switch (operation) {
    case rund::kernel::StencilOp::Sum:
      return RangeAggregateOperation::Sum;
    case rund::kernel::StencilOp::Min:
      return RangeAggregateOperation::Minimum;
    case rund::kernel::StencilOp::Max:
      return RangeAggregateOperation::Maximum;
    }
    return std::nullopt;
  }

  constexpr RangeAggregateShape(const RangeAggregateTraits traits,
                                const RangeAggregateBoundary boundary,
                                const rund::kernel::u64 element_count,
                                const rund::kernel::u64 radius,
                                const rund::kernel::u32 element_bytes) noexcept
      : traits_(traits), boundary_(boundary), element_count_(element_count),
        radius_(radius), element_bytes_(element_bytes) {}

  RangeAggregateTraits traits_;
  RangeAggregateBoundary boundary_;
  rund::kernel::u64 element_count_;
  rund::kernel::u64 radius_;
  rund::kernel::u32 element_bytes_;
};

class RangeAggregateCapabilities final {
public:
  RangeAggregateCapabilities() = delete;

  [[nodiscard]] static constexpr RangeAggregateCapabilities
  unavailable() noexcept {
    return RangeAggregateCapabilities{
        RangeAggregateCapabilityDisposition::Unavailable,
        RangeAggregateSourceVariant::Unavailable,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u};
  }

  [[nodiscard]] static constexpr RangeAggregateCapabilities cpu() noexcept {
    return RangeAggregateCapabilities{
        RangeAggregateCapabilityDisposition::Cpu,
        RangeAggregateSourceVariant::Cpu,
        0u,
        0u,
        0u,
        0u,
        0u,
        RangeAggregateSupportBit(RangeAggregateSupport::Direct)};
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateCapabilities>
  gpu(const RangeAggregateSourceVariant source_variant,
      const std::uint8_t legal_width_mask,
      const rund::kernel::u32 maximum_threads_per_workgroup,
      const rund::kernel::u32 shared_memory_occupancy_budget,
      const rund::kernel::u64 shared_memory_limit,
      const rund::kernel::u64 maximum_group_count,
      const std::uint8_t supported_candidates) noexcept {
    const RangeAggregateCapabilities capabilities{
        RangeAggregateCapabilityDisposition::Gpu,
        source_variant,
        legal_width_mask,
        maximum_threads_per_workgroup,
        shared_memory_occupancy_budget,
        shared_memory_limit,
        maximum_group_count,
        supported_candidates};
    return capabilities.valid()
               ? std::optional<RangeAggregateCapabilities>{capabilities}
               : std::nullopt;
  }

  [[nodiscard]] constexpr RangeAggregateSourceVariant
  source_variant() const noexcept {
    return source_variant_;
  }

  [[nodiscard]] constexpr RangeAggregateCapabilityDisposition
  disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr bool available() const noexcept {
    return disposition_ != RangeAggregateCapabilityDisposition::Unavailable;
  }

  [[nodiscard]] constexpr bool
  supports(const RangeAggregateSupport support) const noexcept {
    return (supported_candidates_ & RangeAggregateSupportBit(support)) != 0u;
  }

  [[nodiscard]] constexpr bool
  supports_width(const rund::kernel::u32 width) const noexcept {
    const std::uint8_t bit = width == 64u    ? kRangeAggregateWidth64Bit
                             : width == 128u ? kRangeAggregateWidth128Bit
                             : width == 256u ? kRangeAggregateWidth256Bit
                                             : 0u;
    return bit != 0u && width <= maximum_threads_per_workgroup_ &&
           (legal_width_mask_ & bit) != 0u;
  }

  [[nodiscard]] constexpr bool cpu_only() const noexcept {
    return disposition_ == RangeAggregateCapabilityDisposition::Cpu &&
           source_variant_ == RangeAggregateSourceVariant::Cpu &&
           legal_width_mask_ == 0u && maximum_threads_per_workgroup_ == 0u &&
           maximum_group_count_ == 0u &&
           supported_candidates_ ==
               RangeAggregateSupportBit(RangeAggregateSupport::Direct);
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

  [[nodiscard]] constexpr std::uint8_t supported_candidates() const noexcept {
    return supported_candidates_;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    if (disposition_ == RangeAggregateCapabilityDisposition::Unavailable) {
      return false;
    }
    if (cpu_only()) {
      return true;
    }
    const bool gpu_variant =
        source_variant_ == RangeAggregateSourceVariant::Metal ||
        source_variant_ == RangeAggregateSourceVariant::Vulkan;
    return disposition_ == RangeAggregateCapabilityDisposition::Gpu &&
           gpu_variant && legal_width_mask_ != 0u &&
           (legal_width_mask_ & ~kRangeAggregateKnownWidthMask) == 0u &&
           maximum_threads_per_workgroup_ >= 64u &&
           maximum_group_count_ != 0u &&
           (supported_candidates_ &
            RangeAggregateSupportBit(RangeAggregateSupport::Direct)) != 0u &&
           (supported_candidates_ & ~kRangeAggregateKnownSupportMask) == 0u;
  }

private:
  constexpr RangeAggregateCapabilities(
      const RangeAggregateCapabilityDisposition disposition,
      const RangeAggregateSourceVariant source_variant,
      const std::uint8_t legal_width_mask,
      const rund::kernel::u32 maximum_threads_per_workgroup,
      const rund::kernel::u32 shared_memory_occupancy_budget,
      const rund::kernel::u64 shared_memory_limit,
      const rund::kernel::u64 maximum_group_count,
      const std::uint8_t supported_candidates) noexcept
      : disposition_(disposition), source_variant_(source_variant),
        legal_width_mask_(legal_width_mask),
        maximum_threads_per_workgroup_(maximum_threads_per_workgroup),
        shared_memory_occupancy_budget_(shared_memory_occupancy_budget),
        shared_memory_limit_(shared_memory_limit),
        maximum_group_count_(maximum_group_count),
        supported_candidates_(supported_candidates) {}

  RangeAggregateCapabilityDisposition disposition_;
  RangeAggregateSourceVariant source_variant_;
  std::uint8_t legal_width_mask_;
  rund::kernel::u32 maximum_threads_per_workgroup_;
  rund::kernel::u32 shared_memory_occupancy_budget_;
  rund::kernel::u64 shared_memory_limit_;
  rund::kernel::u64 maximum_group_count_;
  std::uint8_t supported_candidates_;
};

class RangeAggregateCandidate final {
public:
  RangeAggregateCandidate() = delete;

  [[nodiscard]] static constexpr RangeAggregateCandidate direct_cpu() noexcept {
    return RangeAggregateCandidate{RangeAggregateCandidateDisposition::Direct,
                                   0u, 0u};
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateCandidate>
  direct_gpu(const rund::kernel::u32 width) noexcept {
    return Make(RangeAggregateCandidateDisposition::Direct, width, 0u);
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateCandidate>
  shared_halo(const rund::kernel::u32 width,
              const rund::kernel::u32 radius_capacity) noexcept {
    return SupportedWidth(width) && radius_capacity != 0u &&
                   radius_capacity <= width
               ? std::optional<RangeAggregateCandidate>{RangeAggregateCandidate{
                     RangeAggregateCandidateDisposition::SharedHalo, width,
                     radius_capacity}}
               : std::nullopt;
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateCandidate>
  prefix_difference(const rund::kernel::u32 width) noexcept {
    return Make(RangeAggregateCandidateDisposition::PrefixDifference, width,
                0u);
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateCandidate>
  block_prefix_suffix(const rund::kernel::u32 width) noexcept {
    return Make(RangeAggregateCandidateDisposition::BlockPrefixSuffix, width,
                0u);
  }

  [[nodiscard]] constexpr RangeAggregateCandidateDisposition
  disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 width() const noexcept {
    return width_;
  }

  [[nodiscard]] constexpr rund::kernel::u32 radius_capacity() const noexcept {
    return radius_capacity_;
  }

  [[nodiscard]] constexpr bool uses_shared_halo() const noexcept {
    return disposition_ == RangeAggregateCandidateDisposition::SharedHalo;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const RangeAggregateCandidate &,
             const RangeAggregateCandidate &) = default;

private:
  [[nodiscard]] static constexpr bool
  SupportedWidth(const rund::kernel::u32 width) noexcept {
    return width == 64u || width == 128u || width == 256u;
  }

  [[nodiscard]] static constexpr std::optional<RangeAggregateCandidate>
  Make(const RangeAggregateCandidateDisposition disposition,
       const rund::kernel::u32 width,
       const rund::kernel::u32 radius_capacity) noexcept {
    return SupportedWidth(width)
               ? std::optional<RangeAggregateCandidate>{RangeAggregateCandidate{
                     disposition, width, radius_capacity}}
               : std::nullopt;
  }

  constexpr RangeAggregateCandidate(
      const RangeAggregateCandidateDisposition disposition,
      const rund::kernel::u32 width,
      const rund::kernel::u32 radius_capacity) noexcept
      : disposition_(disposition), width_(width),
        radius_capacity_(radius_capacity) {}

  RangeAggregateCandidateDisposition disposition_;
  rund::kernel::u32 width_;
  rund::kernel::u32 radius_capacity_;
};

struct RangeAggregateCost final {
  rund::kernel::u128 global_read_bytes{};
  rund::kernel::u128 global_write_bytes{};
  rund::kernel::u128 combine_ops{};
  rund::kernel::u128 inverse_ops{};
  rund::kernel::u128 scale_ops{};
  rund::kernel::u64 shared_bytes{};
  rund::kernel::u64 scratch_bytes{};
  rund::kernel::u64 dispatch_count{};
  rund::kernel::u128 launched_lanes{};

  [[nodiscard]] friend constexpr bool
  operator==(const RangeAggregateCost &, const RangeAggregateCost &) = default;
};

struct RangeTemporaryRequirement final {
  RangeTemporaryRole role{};
  std::uint8_t ordinal{};
  rund::kernel::u64 bytes{};
  rund::kernel::u64 alignment{};
  std::uint8_t first_stage{};
  std::uint8_t last_stage{};

  [[nodiscard]] friend constexpr bool
  operator==(const RangeTemporaryRequirement &,
             const RangeTemporaryRequirement &) = default;
};

struct RangeAggregateStagePlan final {
  RangeAggregateStageDisposition disposition{};
  std::uint8_t level{};
  rund::kernel::u64 element_count{};
  rund::kernel::u64 groups{};
  rund::kernel::u32 width{};

  [[nodiscard]] friend constexpr bool
  operator==(const RangeAggregateStagePlan &,
             const RangeAggregateStagePlan &) = default;
};

// Width 64 needs at most eleven hierarchy levels for any admitted u64-sized
// payload. Prefix has one up stage per level, one fewer down stage, and one
// output stage. Fixed storage preserves allocation-free planning.
inline constexpr std::size_t kRangeAggregateStageCapacity = 24u;
inline constexpr std::size_t kRangeTemporaryCapacity = 12u;
inline constexpr std::size_t kRangeAggregateCandidateCapacity = 12u;

// This derives physical prefix stages and temporary lifetimes. It does not
// choose an aggregate algorithm: RangeAggregate selects PrefixDifference, and
// native Scan projects its own observable-prefix semantics into the flat form.
enum class RangeAggregatePrefixDisposition : std::uint8_t {
  Rejected,
  Hierarchical,
  FlatBlockTotals,
};

namespace range_aggregate_prefix_detail {
class Builder;
}

class RangeAggregatePrefixExecution final {
public:
  RangeAggregatePrefixExecution() = delete;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return disposition_ != RangeAggregatePrefixDisposition::Rejected;
  }

  [[nodiscard]] constexpr RangeAggregatePrefixDisposition
  disposition() const noexcept {
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

  [[nodiscard]] constexpr RangeAggregateStagePlan
  stage(const std::size_t index) const noexcept {
    assert(index < stage_count());
    return selection().stages[index];
  }

  [[nodiscard]] constexpr std::size_t temporary_count() const noexcept {
    return selection().temporary_count;
  }

  [[nodiscard]] constexpr RangeTemporaryRequirement
  temporary(const std::size_t index) const noexcept {
    assert(index < temporary_count());
    return selection().temporaries[index];
  }

private:
  struct Selection final {
    rund::kernel::u32 width{};
    std::array<RangeAggregateStagePlan, kRangeAggregateStageCapacity> stages{};
    std::size_t stage_count{};
    std::array<RangeTemporaryRequirement, kRangeTemporaryCapacity>
        temporaries{};
    std::size_t temporary_count{};
  };

  friend class range_aggregate_prefix_detail::Builder;

  [[nodiscard]] static constexpr RangeAggregatePrefixExecution
  rejected(const char *const reason) noexcept {
    return RangeAggregatePrefixExecution{
        RangeAggregatePrefixDisposition::Rejected, reason};
  }

  [[nodiscard]] static constexpr RangeAggregatePrefixExecution
  selected(const RangeAggregatePrefixDisposition disposition,
           const Selection selection) noexcept {
    return RangeAggregatePrefixExecution{disposition, selection};
  }

  [[nodiscard]] constexpr const Selection &selection() const noexcept {
    assert(ok() && selection_.has_value());
    return *selection_;
  }

  constexpr RangeAggregatePrefixExecution(
      const RangeAggregatePrefixDisposition disposition,
      const char *const reason) noexcept
      : disposition_(disposition), reason_(reason) {}

  constexpr RangeAggregatePrefixExecution(
      const RangeAggregatePrefixDisposition disposition,
      const Selection selection) noexcept
      : disposition_(disposition), selection_(selection), reason_("ok") {}

  RangeAggregatePrefixDisposition disposition_;
  std::optional<Selection> selection_{};
  const char *reason_;
};

namespace range_aggregate_prefix_detail {

class Builder final {
public:
  constexpr explicit Builder(const rund::kernel::u32 width) noexcept
      : selection_{.width = width} {}

  [[nodiscard]] static constexpr RangeAggregatePrefixExecution
  rejected() noexcept {
    return RangeAggregatePrefixExecution::rejected(
        "compute_range_aggregate_prefix_invalid");
  }

  [[nodiscard]] constexpr bool
  append_stage(const RangeAggregateStageDisposition disposition,
               const std::uint8_t level, const rund::kernel::u64 element_count,
               const rund::kernel::u64 groups,
               const rund::kernel::u32 width) noexcept {
    if (selection_.stage_count == selection_.stages.size() || groups == 0u) {
      return false;
    }
    selection_.stages[selection_.stage_count++] = RangeAggregateStagePlan{
        .disposition = disposition,
        .level = level,
        .element_count = element_count,
        .groups = groups,
        .width = width,
    };
    return true;
  }

  [[nodiscard]] constexpr bool append_temporary(
      const RangeTemporaryRole role, const std::uint8_t ordinal,
      const rund::kernel::u64 bytes, const rund::kernel::u32 alignment,
      const std::uint8_t first_stage, const std::uint8_t last_stage) noexcept {
    if (selection_.temporary_count == selection_.temporaries.size() ||
        bytes == 0u || alignment == 0u || first_stage > last_stage) {
      return false;
    }
    selection_.temporaries[selection_.temporary_count++] =
        RangeTemporaryRequirement{.role = role,
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

  [[nodiscard]] constexpr RangeAggregateStagePlan
  stage(const std::size_t index) const noexcept {
    assert(index < selection_.stage_count);
    return selection_.stages[index];
  }

  constexpr void set_last_stage(const std::size_t index,
                                const std::uint8_t last_stage) noexcept {
    assert(index < selection_.temporary_count);
    selection_.temporaries[index].last_stage = last_stage;
  }

  [[nodiscard]] constexpr RangeAggregatePrefixExecution
  finish(const RangeAggregatePrefixDisposition disposition) const noexcept {
    return RangeAggregatePrefixExecution::selected(disposition, selection_);
  }

private:
  RangeAggregatePrefixExecution::Selection selection_;
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

} // namespace range_aggregate_prefix_detail

// Derives the work-efficient recursive prefix hierarchy used by the
// PrefixDifference candidate. Every hierarchy stage must fit the physical
// single-stage dispatch limit supplied by the backend capability projection.
[[nodiscard]] constexpr RangeAggregatePrefixExecution
PlanRangeAggregatePrefixHierarchy(
    const rund::kernel::u64 element_count, const rund::kernel::u32 width,
    const rund::kernel::u32 element_bytes,
    const rund::kernel::u64 maximum_group_count) noexcept {
  using namespace range_aggregate_prefix_detail;
  if (element_count == 0u || !valid_width(width) ||
      (element_bytes != 4u && element_bytes != 8u) ||
      maximum_group_count == 0u) {
    return Builder::rejected();
  }

  Builder builder{width};
  std::array<std::size_t, kRangeTemporaryCapacity> summary_index{};
  std::size_t level_count = 0u;
  rund::kernel::u64 values = element_count;
  while (true) {
    if (level_count == summary_index.size()) {
      return Builder::rejected();
    }
    const rund::kernel::u64 group_count = groups(values, width);
    if (group_count == 0u || group_count > maximum_group_count ||
        !builder.append_stage(
            level_count == 0u ? RangeAggregateStageDisposition::PrefixBlock
                              : RangeAggregateStageDisposition::PrefixSummary,
            static_cast<std::uint8_t>(level_count), values, group_count,
            width)) {
      return Builder::rejected();
    }
    ++level_count;
    if (group_count == 1u) {
      break;
    }
    rund::kernel::u64 bytes = 0u;
    if (!rund::kernel::checked::mul(group_count, element_bytes, bytes) ||
        !builder.append_temporary(
            RangeTemporaryRole::BlockSummaries,
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
    const RangeAggregateStagePlan child_stage = builder.stage(child);
    if (!builder.append_stage(RangeAggregateStageDisposition::PrefixFixup,
                              static_cast<std::uint8_t>(child),
                              child_stage.element_count, child_stage.groups,
                              width)) {
      return Builder::rejected();
    }
    builder.set_last_stage(
        summary_index[child],
        static_cast<std::uint8_t>(2u * level_count - 2u - child));
  }
  return builder.finish(RangeAggregatePrefixDisposition::Hierarchical);
}

// Derives the native Scan physical graph: block-local prefixes, one flat
// block-total prefix, then final offset/materialization. The caller supplies
// the semantic block count; device dispatch chunking remains a backend command
// concern, so this records full logical block count rather than an
// adapter-specific chunk count.
[[nodiscard]] constexpr RangeAggregatePrefixExecution
PlanRangeAggregateFlatPrefix(const rund::kernel::u64 element_count,
                             const rund::kernel::u64 block_count,
                             const rund::kernel::u32 width,
                             const rund::kernel::u32 element_bytes) noexcept {
  using namespace range_aggregate_prefix_detail;
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
  if (!builder.append_stage(RangeAggregateStageDisposition::PrefixBlock, 0u,
                            element_count, block_count, width) ||
      !builder.append_temporary(
          RangeTemporaryRole::BlockSummaries, 0u, summary_bytes, element_bytes,
          0u, static_cast<std::uint8_t>(block_count == 1u ? 0u : 2u))) {
    return Builder::rejected();
  }
  if (block_count == 1u) {
    return builder.finish(RangeAggregatePrefixDisposition::FlatBlockTotals);
  }
  if (!builder.append_stage(RangeAggregateStageDisposition::PrefixSummary, 0u,
                            block_count, 1u, width) ||
      !builder.append_stage(RangeAggregateStageDisposition::PrefixFixup, 0u,
                            element_count, block_count, width)) {
    return Builder::rejected();
  }
  return builder.finish(RangeAggregatePrefixDisposition::FlatBlockTotals);
}

class RangeAggregatePlan final {
public:
  RangeAggregatePlan() = delete;

  [[nodiscard]] static constexpr RangeAggregatePlan
  rejected(const char *const reason) noexcept {
    return RangeAggregatePlan{RangeAggregatePlanDisposition::Rejected, reason};
  }

  [[nodiscard]] constexpr RangeAggregatePlanDisposition
  disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr bool ok() const noexcept {
    return disposition_ == RangeAggregatePlanDisposition::Selected;
  }

  [[nodiscard]] constexpr const char *reason() const noexcept {
    return reason_;
  }

  [[nodiscard]] constexpr const RangeAggregateShape &shape() const noexcept {
    return selection().shape;
  }

  [[nodiscard]] constexpr const RangeAggregateCandidate &
  candidate() const noexcept {
    return selection().candidate;
  }

  [[nodiscard]] constexpr const RangeAggregateCost &cost() const noexcept {
    return selection().cost;
  }

  [[nodiscard]] constexpr std::size_t stage_count() const noexcept {
    return selection().stage_count;
  }

  [[nodiscard]] constexpr RangeAggregateStagePlan
  stage(const std::size_t index) const noexcept {
    assert(index < stage_count());
    const Selection &selected = selection();
    const RangeAggregateCandidateDisposition disposition =
        selected.candidate.disposition();
    if (disposition == RangeAggregateCandidateDisposition::Direct) {
      return RangeAggregateStagePlan{
          .disposition = RangeAggregateStageDisposition::Direct,
          .level = 0u,
          .element_count = selected.shape.element_count(),
          .groups = selected.candidate.width() == 0u
                        ? 1u
                        : Groups(selected.shape.element_count(),
                                 selected.candidate.width()),
          .width = selected.candidate.width()};
    }
    if (disposition == RangeAggregateCandidateDisposition::SharedHalo) {
      return RangeAggregateStagePlan{
          .disposition = RangeAggregateStageDisposition::SharedHalo,
          .level = 0u,
          .element_count = selected.shape.element_count(),
          .groups = Groups(selected.shape.element_count(),
                           selected.candidate.width()),
          .width = selected.candidate.width()};
    }
    if (disposition == RangeAggregateCandidateDisposition::BlockPrefixSuffix) {
      if (index == 0u) {
        const rund::kernel::u64 padded =
            selected.shape.element_count() + 2u * selected.shape.radius();
        const rund::kernel::u64 window = 2u * selected.shape.radius() + 1u;
        const rund::kernel::u64 blocks = Groups(padded, window);
        return RangeAggregateStagePlan{
            .disposition = RangeAggregateStageDisposition::BlockPrefixSuffix,
            .level = 0u,
            .element_count = padded,
            .groups = Groups(blocks, selected.candidate.width()),
            .width = selected.candidate.width()};
      }
      return RangeAggregateStagePlan{
          .disposition = RangeAggregateStageDisposition::BlockWindow,
          .level = 0u,
          .element_count = selected.shape.element_count(),
          .groups = Groups(selected.shape.element_count(),
                           selected.candidate.width()),
          .width = selected.candidate.width()};
    }

    const RangeAggregatePrefixExecution prefix =
        PlanRangeAggregatePrefixHierarchy(
            selected.shape.element_count(), selected.candidate.width(),
            selected.shape.element_bytes(),
            std::numeric_limits<rund::kernel::u64>::max());
    assert(prefix.ok() && selected.stage_count == prefix.stage_count() + 1u);
    if (index < prefix.stage_count()) {
      return prefix.stage(index);
    }
    return RangeAggregateStagePlan{
        .disposition = RangeAggregateStageDisposition::PrefixWindow,
        .level = 0u,
        .element_count = selected.shape.element_count(),
        .groups =
            Groups(selected.shape.element_count(), selected.candidate.width()),
        .width = selected.candidate.width()};
  }

  [[nodiscard]] constexpr std::size_t temporary_count() const noexcept {
    return selection().temporary_count;
  }

  [[nodiscard]] constexpr RangeTemporaryRequirement
  temporary(const std::size_t index) const noexcept {
    assert(index < temporary_count());
    const Selection &selected = selection();
    if (selected.candidate.disposition() ==
        RangeAggregateCandidateDisposition::PrefixDifference) {
      if (index == 0u) {
        return RangeTemporaryRequirement{
            .role = RangeTemporaryRole::PrefixValues,
            .ordinal = 0u,
            .bytes = selected.shape.payload_bytes(),
            .alignment = selected.shape.element_bytes(),
            .first_stage = 0u,
            .last_stage = static_cast<std::uint8_t>(selected.stage_count - 1u)};
      }
      const RangeAggregatePrefixExecution prefix =
          PlanRangeAggregatePrefixHierarchy(
              selected.shape.element_count(), selected.candidate.width(),
              selected.shape.element_bytes(),
              std::numeric_limits<rund::kernel::u64>::max());
      assert(prefix.ok() && index - 1u < prefix.temporary_count() &&
             selected.temporary_count == prefix.temporary_count() + 1u);
      return prefix.temporary(index - 1u);
    }
    const rund::kernel::u64 bytes =
        (selected.shape.element_count() + 2u * selected.shape.radius()) *
        selected.shape.element_bytes();
    return RangeTemporaryRequirement{
        .role = index == 0u ? RangeTemporaryRole::ForwardValues
                            : RangeTemporaryRole::BackwardValues,
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

  [[nodiscard]] constexpr RangeAggregateIdentity
  source_identity() const noexcept {
    return selection().source_identity;
  }

  [[nodiscard]] constexpr RangeAggregateIdentity
  execution_identity() const noexcept {
    return selection().execution_identity;
  }

private:
  friend constexpr RangeAggregatePlan
  PlanRangeAggregate(const RangeAggregateShape &,
                     const RangeAggregateCapabilities &) noexcept;

  [[nodiscard]] static constexpr RangeAggregatePlan
  selected(const RangeAggregateShape shape,
           const RangeAggregateCandidate candidate,
           const RangeAggregateCost cost, const std::size_t stage_count,
           const std::size_t temporary_count,
           const std::uint8_t legal_candidate_count,
           const std::uint8_t pareto_candidate_count,
           const RangeAggregateIdentity source_identity,
           const RangeAggregateIdentity execution_identity) noexcept {
    return RangeAggregatePlan{shape,
                              candidate,
                              cost,
                              stage_count,
                              temporary_count,
                              legal_candidate_count,
                              pareto_candidate_count,
                              source_identity,
                              execution_identity};
  }

  struct Selection final {
    RangeAggregateShape shape;
    RangeAggregateCandidate candidate;
    RangeAggregateCost cost{};
    std::size_t stage_count{};
    std::size_t temporary_count{};
    std::uint8_t legal_candidate_count{};
    std::uint8_t pareto_candidate_count{};
    RangeAggregateIdentity source_identity{};
    RangeAggregateIdentity execution_identity{};
  };

  [[nodiscard]] constexpr const Selection &selection() const noexcept {
    assert(disposition_ == RangeAggregatePlanDisposition::Selected &&
           selection_.has_value());
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

  constexpr RangeAggregatePlan(const RangeAggregatePlanDisposition disposition,
                               const char *const reason) noexcept
      : disposition_(disposition), reason_(reason) {}

  constexpr RangeAggregatePlan(
      const RangeAggregateShape shape, const RangeAggregateCandidate candidate,
      const RangeAggregateCost cost, const std::size_t stage_count,
      const std::size_t temporary_count,
      const std::uint8_t legal_candidate_count,
      const std::uint8_t pareto_candidate_count,
      const RangeAggregateIdentity source_identity,
      const RangeAggregateIdentity execution_identity) noexcept
      : disposition_(RangeAggregatePlanDisposition::Selected),
        selection_(Selection{shape, candidate, cost, stage_count,
                             temporary_count, legal_candidate_count,
                             pareto_candidate_count, source_identity,
                             execution_identity}),
        reason_("ok") {}

  RangeAggregatePlanDisposition disposition_;
  std::optional<Selection> selection_{};
  const char *reason_{};
};

static_assert(std::is_trivially_copyable_v<RangeAggregateIdentity>);
static_assert(std::is_trivially_copyable_v<RangeAggregateCost>);
static_assert(std::is_trivially_copyable_v<RangeTemporaryRequirement>);
static_assert(std::is_trivially_copyable_v<RangeAggregateStagePlan>);
static_assert(std::is_trivially_copyable_v<RangeAggregatePrefixExecution>);

} // namespace rund::node::accel::detail
