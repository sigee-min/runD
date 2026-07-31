#include "model.hpp"

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace rund_node_collective_modes {

template <class T>
[[nodiscard]] bool CheckExclusiveOverflow(const rund::compute::Backend backend,
                                          DomainEvidence &evidence) {
  using namespace rund::compute;
  // The final total is part of the exclusive-scan result contract even though
  // it is not stored in the output array.
  std::array<T, 4u> input{Maximum<T>(), Small<T>(), Zero<T>(), Zero<T>()};
  std::array<std::uint32_t, 4u> heads{1u, 0u, 0u, 0u};
  auto mixed_heads = heads;
  mixed_heads[3u] = 2u;
  auto scan_target = flow_on(backend, Target::cpu(2u));
  auto scan =
      std::move(scan_target)
          .template input<T>(input.size())
          .branch([](auto values) { return values.scan(Scan::ExclusiveSum); })
          .compile();
  auto segmented_target = flow_on(backend, Target::cpu(2u));
  auto segmented =
      std::move(segmented_target)
          .template input<T>(input.size())
          .template zip_input<std::uint32_t>(heads.size())
          .branch([](auto values, auto segments) {
            return values.segmented_scan(segments, Scan::ExclusiveSum);
          })
          .compile();
  return scan && segmented &&
         SameFailure(*scan, backend, "exclusive-overflow",
                     "compute_scan_sum_overflow", evidence.exclusive_overflow,
                     input) &&
         SameFailure(*segmented, backend, "segmented-exclusive-overflow",
                     "compute_segmented_scan_sum_overflow",
                     evidence.segmented_exclusive_overflow, input, heads) &&
         SameFailure(*segmented, backend, "segmented-exclusive-error-priority",
                     "compute_segmented_scan_segment_invalid",
                     evidence.segmented_exclusive_priority, input, mixed_heads);
}

template <class T>
[[nodiscard]] bool
CheckInclusiveOverflow(const rund::compute::Backend backend,
                       rund::compute::graph::Fingerprint &scan_reference,
                       rund::compute::graph::Fingerprint &segmented_reference,
                       rund::compute::graph::Fingerprint &priority_reference) {
  using namespace rund::compute;
  // Keep both 256-element blocks locally valid and force overflow only while
  // the
  // second block receives the first block's prefix. This covers offset merge
  // and segmented offset merge with the existing all-backend reason oracle.
  std::array<T, 512u> input{};
  std::array<std::uint32_t, 512u> heads{};
  input[255u] = Maximum<T>();
  input[256u] = Small<T>();
  heads[0u] = 1u;
  auto mixed_heads = heads;
  mixed_heads[257u] = 2u;
  auto scan_target = flow_on(backend, Target::cpu(2u));
  auto scan =
      std::move(scan_target)
          .template input<T>(input.size())
          .branch([](auto values) { return values.scan(Scan::InclusiveSum); })
          .compile();
  auto segmented_target = flow_on(backend, Target::cpu(2u));
  auto segmented =
      std::move(segmented_target)
          .template input<T>(input.size())
          .template zip_input<std::uint32_t>(heads.size())
          .branch([](auto values, auto segments) {
            return values.segmented_scan(segments, Scan::InclusiveSum);
          })
          .compile();
  return scan && segmented &&
         SameFailure(*scan, backend, "inclusive-overflow",
                     "compute_scan_sum_overflow", scan_reference, input) &&
         SameFailure(*segmented, backend, "segmented-inclusive-overflow",
                     "compute_segmented_scan_sum_overflow", segmented_reference,
                     input, heads) &&
         SameFailure(*segmented, backend, "segmented-inclusive-error-priority",
                     "compute_segmented_scan_segment_invalid",
                     priority_reference, input, mixed_heads);
}

template <class T>
[[nodiscard]] bool
CheckSegmentedCarryOverflow(const rund::compute::Backend backend,
                            DomainEvidence &evidence) {
  if constexpr (!(std::is_signed_v<T> ||
                  rund::compute::detail::FixedValue<T>)) {
    return true;
  } else {
    using namespace rund::compute;
    std::array<T, 512u> input{};
    std::array<std::uint32_t, 512u> heads{};
    input[255u] = Maximum<T>();
    input[256u] = Small<T>();
    input[257u] = NegativeSmall<T>();
    heads[0u] = 1u;

    auto inclusive_target = flow_on(backend, Target::cpu(2u));
    auto inclusive =
        std::move(inclusive_target)
            .template input<T>(input.size())
            .template zip_input<std::uint32_t>(heads.size())
            .branch([](auto values, auto segments) {
              return values.segmented_scan(segments, Scan::InclusiveSum);
            })
            .compile();
    auto exclusive_target = flow_on(backend, Target::cpu(2u));
    auto exclusive =
        std::move(exclusive_target)
            .template input<T>(input.size())
            .template zip_input<std::uint32_t>(heads.size())
            .branch([](auto values, auto segments) {
              return values.segmented_scan(segments, Scan::ExclusiveSum);
            })
            .compile();
    return inclusive && exclusive &&
           SameFailure(*inclusive, backend, "segmented-carry-inclusive",
                       "compute_segmented_scan_sum_overflow",
                       evidence.segmented_carry_inclusive_overflow, input,
                       heads) &&
           SameFailure(*exclusive, backend, "segmented-carry-exclusive",
                       "compute_segmented_scan_sum_overflow",
                       evidence.segmented_carry_exclusive_overflow, input,
                       heads);
  }
}

template <class T>
[[nodiscard]] bool
CheckSegmentedResetOverflow(const rund::compute::Backend backend,
                            rund::compute::graph::Fingerprint &reference) {
  static_assert(std::is_unsigned_v<T>);
  using namespace rund::compute;
  std::array<T, 64u> input{};
  std::array<std::uint32_t, 64u> heads{};
  input[31u] = Maximum<T>();
  input[32u] = Small<T>();
  heads[0u] = 1u;
  heads[63u] = 1u;
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template input<T>(input.size())
          .template zip_input<std::uint32_t>(heads.size())
          .branch([](auto values, auto segments) {
            return values.segmented_scan(segments, Scan::InclusiveSum);
          })
          .compile();
  return program && SameFailure(*program, backend, "segmented-reset-overflow",
                                "compute_segmented_scan_sum_overflow",
                                reference, input, heads);
}

[[nodiscard]] bool
CheckSegmentHeads(const rund::compute::Backend backend,
                  rund::compute::graph::Fingerprint &zero_reference,
                  rund::compute::graph::Fingerprint &range_reference) {
  using namespace rund::compute;
  std::array<std::int32_t, 2u> input{1, 2};
  std::array<std::uint32_t, 2u> zero{0u, 0u};
  std::array<std::uint32_t, 2u> range{1u, 2u};
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .input<std::int32_t>(input.size())
          .zip_input<std::uint32_t>(zero.size())
          .branch([](auto values, auto segments) {
            return values.segmented_scan(segments, Scan::InclusiveSum);
          })
          .compile();
  return program &&
         SameFailure(*program, backend, "segmented-head-zero",
                     "compute_segmented_scan_segment_invalid", zero_reference,
                     input, zero) &&
         SameFailure(*program, backend, "segmented-head-range",
                     "compute_segmented_scan_segment_invalid", range_reference,
                     input, range);
}

template <class T>
[[nodiscard]] bool CheckExclusiveDomain(const rund::compute::Backend backend,
                                        DomainEvidence &evidence) {
  return CheckExclusiveOverflow<T>(backend, evidence) &&
         CheckSegmentedCarryOverflow<T>(backend, evidence);
}

[[nodiscard]] bool CheckExclusive(const rund::compute::Backend backend,
                                  DomainEvidence &evidence,
                                  const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckExclusiveDomain<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckExclusiveDomain<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckExclusiveDomain<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckExclusiveDomain<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckExclusiveDomain<rund::compute::Fixed<16, 16>>(backend,
                                                              evidence);
  case Domain::Fixed20x44:
    return CheckExclusiveDomain<rund::compute::Fixed<20, 44>>(backend,
                                                              evidence);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

[[nodiscard]] bool CheckInclusive(const rund::compute::Backend backend,
                                  Evidence &evidence, const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckInclusiveOverflow<std::int32_t>(
        backend, evidence.inclusive_overflow[0u],
        evidence.segmented_inclusive_overflow[0u],
        evidence.segmented_scan_priority[0u]);
  case Domain::U32:
    return CheckInclusiveOverflow<std::uint32_t>(
        backend, evidence.inclusive_overflow[1u],
        evidence.segmented_inclusive_overflow[1u],
        evidence.segmented_scan_priority[1u]);
  case Domain::I64:
    return CheckInclusiveOverflow<std::int64_t>(
        backend, evidence.inclusive_overflow[2u],
        evidence.segmented_inclusive_overflow[2u],
        evidence.segmented_scan_priority[2u]);
  case Domain::U64:
    return CheckInclusiveOverflow<std::uint64_t>(
        backend, evidence.inclusive_overflow[3u],
        evidence.segmented_inclusive_overflow[3u],
        evidence.segmented_scan_priority[3u]);
  case Domain::Lane32:
    return CheckInclusiveOverflow<rund::compute::Fixed<1, 31>>(
        backend, evidence.inclusive_overflow[4u],
        evidence.segmented_inclusive_overflow[4u],
        evidence.segmented_scan_priority[4u]);
  case Domain::Lane64:
    return CheckInclusiveOverflow<rund::compute::Fixed<1, 63>>(
        backend, evidence.inclusive_overflow[5u],
        evidence.segmented_inclusive_overflow[5u],
        evidence.segmented_scan_priority[5u]);
  case Domain::Fixed16x16:
  case Domain::Fixed20x44:
    return false;
  }
  return false;
}

[[nodiscard]] bool CheckReset(const rund::compute::Backend backend,
                              Evidence &evidence, const Domain domain) {
  if (domain == Domain::U32) {
    return CheckSegmentedResetOverflow<std::uint32_t>(
        backend, evidence.segmented_reset_overflow[0u]);
  }
  if (domain == Domain::U64) {
    return CheckSegmentedResetOverflow<std::uint64_t>(
        backend, evidence.segmented_reset_overflow[1u]);
  }
  return false;
}

[[nodiscard]] bool CheckHeads(const rund::compute::Backend backend,
                              Evidence &evidence) {
  return CheckSegmentHeads(backend, evidence.segmented_head_zero,
                           evidence.segmented_head_range);
}

} // namespace rund_node_collective_modes
