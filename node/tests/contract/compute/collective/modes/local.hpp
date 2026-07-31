#pragma once

#include <rund/compute.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund_node_collective_modes {

inline constexpr std::size_t kTail = 255u;

struct Hash final {
  std::uint64_t graph{};
  std::uint64_t output{};
};

struct DomainEvidence final {
  Hash modes{};
  Hash clip{};
  Hash extrema{};
  Hash bounded{};
  Hash bounded_window{};
  Hash exact_empty{};
  std::array<Hash, 2u> exact_empty_window{};
  Hash segmented_empty{};
  Hash bounded_empty{};
  Hash bounded_empty_window{};
  Hash cross_block_cancellation{};
  Hash reduce_cancellation{};
  Hash segmented_reduce_cancellation{};
  Hash reduce_scale{};
  Hash segmented_reduce_scale{};
  rund::compute::graph::Fingerprint exclusive_overflow{};
  rund::compute::graph::Fingerprint segmented_exclusive_overflow{};
  rund::compute::graph::Fingerprint segmented_carry_inclusive_overflow{};
  rund::compute::graph::Fingerprint segmented_carry_exclusive_overflow{};
  rund::compute::graph::Fingerprint segmented_exclusive_priority{};
  rund::compute::graph::Fingerprint segmented_reduce_priority{};
  rund::compute::graph::Fingerprint reduce_overflow{};
  rund::compute::graph::Fingerprint segmented_reduce_overflow{};
  rund::compute::graph::Fingerprint bounded_empty_min{};
  rund::compute::graph::Fingerprint bounded_empty_max{};
};

struct Evidence final {
  std::array<DomainEvidence, 6u> domain{};
  std::array<DomainEvidence, 2u> lane_overflow{};
  std::array<rund::compute::graph::Fingerprint, 6u> inclusive_overflow{};
  std::array<rund::compute::graph::Fingerprint, 6u>
      segmented_inclusive_overflow{};
  std::array<rund::compute::graph::Fingerprint, 6u> segmented_scan_priority{};
  std::array<rund::compute::graph::Fingerprint, 2u> segmented_reset_overflow{};
  rund::compute::graph::Fingerprint segmented_head_zero{};
  rund::compute::graph::Fingerprint segmented_head_range{};
};

enum class Domain : std::uint8_t {
  I32,
  U32,
  I64,
  U64,
  Fixed16x16,
  Fixed20x44,
  Lane32,
  Lane64,
};

[[nodiscard]] bool CheckCore(rund::compute::Backend, DomainEvidence &, Domain);
[[nodiscard]] bool CheckBounded(rund::compute::Backend, DomainEvidence &,
                                Domain);
[[nodiscard]] bool CheckEmpty(rund::compute::Backend, DomainEvidence &, Domain);
[[nodiscard]] bool CheckCrossBlock(rund::compute::Backend, DomainEvidence &,
                                   Domain);
[[nodiscard]] bool CheckReductionCancellation(rund::compute::Backend,
                                              DomainEvidence &, Domain);
[[nodiscard]] bool CheckScale(rund::compute::Backend, DomainEvidence &, Domain);
[[nodiscard]] bool CheckReductionOverflow(rund::compute::Backend,
                                          DomainEvidence &, Domain);
[[nodiscard]] bool CheckExclusive(rund::compute::Backend, DomainEvidence &,
                                  Domain);
[[nodiscard]] bool CheckInclusive(rund::compute::Backend, Evidence &, Domain);
[[nodiscard]] bool CheckReset(rund::compute::Backend, Evidence &, Domain);
[[nodiscard]] bool CheckHeads(rund::compute::Backend, Evidence &);
[[nodiscard]] bool CheckBackend(rund::compute::Backend, Evidence &);

} // namespace rund_node_collective_modes
