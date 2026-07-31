#include "model.hpp"

#include <array>
#include <cstdint>
#include <utility>

namespace rund_node_collective_modes {

template <class T>
[[nodiscard]] bool CheckReductionOverflow(const rund::compute::Backend backend,
                                          DomainEvidence &evidence) {
  using namespace rund::compute;
  constexpr std::size_t kCount = 512u;
  constexpr std::size_t kShort = 128u;
  std::array<T, kCount> short_values{};
  std::array<std::uint32_t, kCount> short_heads{};
  short_values[0u] = Maximum<T>();
  short_values[1u] = Small<T>();
  for (std::size_t index = 0u; index < kCount; index += kShort) {
    short_heads[index] = 1u;
  }
  auto short_mixed = short_heads;
  short_mixed[2u] = 2u;

  // One 512-element segment crosses eight values per fixed 64-lane
  // subsequence. Its two 256-element halves are independently representable;
  // only their deterministic tree merge overflows. The same compiled Program
  // serves the 128-element rows through that one tree authority.
  std::array<T, kCount> long_values{};
  std::array<std::uint32_t, kCount> long_heads{};
  long_values[255u] = Maximum<T>();
  long_values[256u] = Small<T>();
  long_heads[0u] = 1u;
  auto long_mixed = long_heads;
  long_mixed[511u] = 2u;
  auto reduce_target = flow_on(backend, Target::cpu(2u));
  auto reduce =
      std::move(reduce_target)
          .template input<T>(long_values.size())
          .branch([](auto values) { return values.reduce(Reduce::Sum); })
          .compile();
  auto segmented_target = flow_on(backend, Target::cpu(2u));
  auto segmented = std::move(segmented_target)
                       .template input<T>(long_values.size())
                       .template zip_input<std::uint32_t>(long_heads.size())
                       .branch([](auto values, auto segments) {
                         return values.segmented_reduce(segments, Reduce::Sum);
                       })
                       .compile();
  return reduce && segmented &&
         SameFailure(*reduce, backend, "reduce-overflow-short",
                     "compute_reduce_sum_overflow", evidence.reduce_overflow,
                     short_values) &&
         SameFailure(*segmented, backend, "segmented-reduce-overflow-short",
                     "compute_segmented_reduce_sum_overflow",
                     evidence.segmented_reduce_overflow, short_values,
                     short_heads) &&
         SameFailure(*segmented, backend, "segmented-reduce-priority-short",
                     "compute_segmented_reduce_segment_invalid",
                     evidence.segmented_reduce_priority, short_values,
                     short_mixed) &&
         SameFailure(*reduce, backend, "reduce-overflow-long",
                     "compute_reduce_sum_overflow", evidence.reduce_overflow,
                     long_values) &&
         SameFailure(*segmented, backend, "segmented-reduce-overflow-long",
                     "compute_segmented_reduce_sum_overflow",
                     evidence.segmented_reduce_overflow, long_values,
                     long_heads) &&
         SameFailure(*segmented, backend, "segmented-reduce-priority-long",
                     "compute_segmented_reduce_segment_invalid",
                     evidence.segmented_reduce_priority, long_values,
                     long_mixed);
}

[[nodiscard]] bool CheckReductionOverflow(const rund::compute::Backend backend,
                                          DomainEvidence &evidence,
                                          const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckReductionOverflow<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckReductionOverflow<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckReductionOverflow<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckReductionOverflow<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckReductionOverflow<rund::compute::Fixed<16, 16>>(backend,
                                                                evidence);
  case Domain::Fixed20x44:
    return CheckReductionOverflow<rund::compute::Fixed<20, 44>>(backend,
                                                                evidence);
  case Domain::Lane32:
    return CheckReductionOverflow<rund::compute::Fixed<1, 31>>(backend,
                                                               evidence);
  case Domain::Lane64:
    return CheckReductionOverflow<rund::compute::Fixed<1, 63>>(backend,
                                                               evidence);
  }
  return false;
}

} // namespace rund_node_collective_modes
