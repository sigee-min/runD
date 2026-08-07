#pragma once

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

enum class PreparedKernelPublicationKind : std::uint8_t {
  Terminal = 0u,
  Window = 1u,
};

static_assert(
    static_cast<std::uint8_t>(PreparedKernelPublicationKind::Terminal) == 0u);
static_assert(
    static_cast<std::uint8_t>(PreparedKernelPublicationKind::Window) == 1u);

[[nodiscard]] inline constexpr bool ValidPreparedKernelPublicationKind(
    const PreparedKernelPublicationKind kind) noexcept {
  return kind == PreparedKernelPublicationKind::Terminal ||
         kind == PreparedKernelPublicationKind::Window;
}

// Sole arithmetic authority for converting one semantic publication into its
// physical backend command contribution.
[[nodiscard]] inline bool PreparedKernelPublicationCommandContribution(
    const PreparedKernelPublicationKind kind, const std::uint64_t outer_bound,
    std::uint64_t &commands) noexcept {
  commands = 0u;
  if (!ValidPreparedKernelPublicationKind(kind)) {
    return false;
  }
  if (kind == PreparedKernelPublicationKind::Terminal) {
    if (outer_bound != 0u) {
      return false;
    }
    commands = 2u;
    return true;
  }
  commands = outer_bound;
  return commands != 0u;
}

struct PreparedKernelPublicationViewIdentity final {
  // Filled only when a backend materializes the canonical resource ordinal.
  // It is excluded from the plan/template fingerprint because transactional
  // parity may select a different physical owner without changing semantics.
  std::uint64_t resident_id{};
  std::uint64_t backing_bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t count{};
  std::uint64_t stride_bytes{};
  std::uint64_t element_bytes{};
  std::uint32_t resource_ordinal{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t usage{};
};

[[nodiscard]] inline constexpr bool SamePreparedKernelPublicationViewIdentity(
    const PreparedKernelPublicationViewIdentity &left,
    const PreparedKernelPublicationViewIdentity &right) noexcept {
  return left.resident_id == right.resident_id &&
         left.backing_bytes == right.backing_bytes &&
         left.offset_bytes == right.offset_bytes && left.count == right.count &&
         left.stride_bytes == right.stride_bytes &&
         left.element_bytes == right.element_bytes &&
         left.resource_ordinal == right.resource_ordinal &&
         left.usage == right.usage;
}

struct PreparedKernelPublicationIdentity final {
  PreparedKernelPublicationViewIdentity sources[3]{};
  PreparedKernelPublicationViewIdentity count{};
  PreparedKernelPublicationViewIdentity target{};
  std::uint32_t state{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t final{};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  // Projected from NestedTemplateShape. maximum/tile remain the native copy
  // geometry; they are never used to reconstruct this command cardinality.
  std::uint32_t outer_bound{};
  PreparedKernelPublicationKind kind{PreparedKernelPublicationKind::Terminal};
};

inline void
SeedPreparedKernelPublicationFingerprint(std::uint64_t &hi,
                                         std::uint64_t &lo) noexcept {
  hi = 0x72756e442e707562ull;
  lo = 0x6c69636174696f6eull;
}

inline void MixPreparedKernelPublicationFingerprint(
    std::uint64_t &hi, std::uint64_t &lo,
    const PreparedKernelPublicationIdentity &identity) noexcept {
  const auto mix = [](std::uint64_t &hash, const std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  const auto view = [&](const PreparedKernelPublicationViewIdentity &value) {
    mix(hi, value.resource_ordinal);
    mix(lo, value.backing_bytes);
    mix(hi, value.offset_bytes);
    mix(lo, value.count);
    mix(hi, value.stride_bytes);
    mix(lo, value.element_bytes);
    mix(hi, value.usage);
  };
  for (const PreparedKernelPublicationViewIdentity &source : identity.sources) {
    view(source);
  }
  view(identity.count);
  view(identity.target);
  mix(lo, identity.state);
  mix(hi, identity.final);
  mix(lo, identity.maximum);
  mix(hi, identity.tile);
  mix(lo, identity.outer_bound);
  mix(lo, static_cast<std::uint8_t>(identity.kind));
}

} // namespace rund::node::accel::detail
