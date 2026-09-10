#pragma once

#include <kernel/core/model.hpp>
#include <kernel/program/compute/model.hpp>

#include <array>
#include <cstdint>
#include <optional>

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
  TiledDifference,
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
  TiledDifference,
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
  TiledDifference = 1u << 4u,
};

[[nodiscard]] constexpr std::uint8_t
RangeSupportBit(const RangeSupport support) noexcept {
  return static_cast<std::uint8_t>(support);
}

inline constexpr std::uint8_t kRangeKnownSupportMask =
    RangeSupportBit(RangeSupport::Direct) |
    RangeSupportBit(RangeSupport::SharedHalo) |
    RangeSupportBit(RangeSupport::PrefixDifference) |
    RangeSupportBit(RangeSupport::BlockPrefixSuffix) |
    RangeSupportBit(RangeSupport::TiledDifference);

inline constexpr std::uint8_t kRangeWidth64Bit = 1u << 0u;
inline constexpr std::uint8_t kRangeWidth128Bit = 1u << 1u;
inline constexpr std::uint8_t kRangeWidth256Bit = 1u << 2u;
inline constexpr std::uint8_t kRangeKnownWidthMask =
    kRangeWidth64Bit | kRangeWidth128Bit | kRangeWidth256Bit;
inline constexpr std::array<rund::kernel::u32, 3u> kRangeWidths{64u, 128u,
                                                                256u};
inline constexpr rund::kernel::u32 kRangeCpuBlockWidth = 64u;
// Fixed source geometry: each lane owns this many consecutive outputs.
inline constexpr rund::kernel::u32 kRangeTileOutputsPerLane = 16u;
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

} // namespace rund::node::accel::detail
