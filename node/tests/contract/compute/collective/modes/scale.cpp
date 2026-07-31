#include "model.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace rund_node_collective_modes {

template <class T>
[[nodiscard]] bool CheckReductionScale(const rund::compute::Backend backend,
                                       DomainEvidence &evidence) {
  using namespace rund::compute;
  constexpr std::size_t kScaleCount = 65536u;
  std::vector<T> input(kScaleCount, Small<T>());
  std::vector<std::uint32_t> heads(kScaleCount, 0u);
  heads[0u] = 1u;
  auto reduce_target = flow_on(backend, Target::cpu(2u));
  auto reduce =
      std::move(reduce_target)
          .template input<T>(input.size())
          .branch([](auto values) { return values.reduce(Reduce::Sum); })
          .compile();
  auto segmented_target = flow_on(backend, Target::cpu(2u));
  auto segmented = std::move(segmented_target)
                       .template input<T>(input.size())
                       .template zip_input<std::uint32_t>(heads.size())
                       .branch([](auto values, auto segments) {
                         return values.segmented_reduce(segments, Reduce::Sum);
                       })
                       .compile();
  if (!reduce || !segmented) {
    std::fprintf(stderr,
                 "compute modes reduction scale compile backend=%u width=%zu "
                 "reduce=%.*s segmented=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(reduce.error().size()), reduce.error().data(),
                 static_cast<int>(segmented.error().size()),
                 segmented.error().data());
    return false;
  }
  auto reduce_job = reduce->resident(input);
  auto segmented_job = segmented->resident(input, heads);
  if (!reduce_job ||
      !SameSuccess(*reduce_job, backend, "reduce-scale",
                   evidence.reduce_scale) ||
      !segmented_job ||
      !SameSuccess(*segmented_job, backend, "segmented-reduce-scale",
                   evidence.segmented_reduce_scale)) {
    return false;
  }
  auto reduced = reduce_job->read();
  auto segmented_output = segmented_job->read();
  T expected_total{};
  if constexpr (detail::FixedValue<T>) {
    using Raw = typename T::Raw;
    expected_total = T::from_raw(static_cast<Raw>(kScaleCount));
  } else {
    expected_total = static_cast<T>(kScaleCount);
  }
  const std::vector<T> reduced_expected{expected_total};
  std::vector<T> segmented_expected(kScaleCount, Zero<T>());
  segmented_expected.front() = expected_total;
  const bool same = reduced && *reduced == reduced_expected &&
                    segmented_output && *segmented_output == segmented_expected;
  if (!same) {
    std::fprintf(stderr,
                 "compute modes reduction scale mismatch backend=%u "
                 "width=%zu count=%zu\n",
                 static_cast<unsigned>(backend), sizeof(T), kScaleCount);
  }
  return same;
}

[[nodiscard]] bool CheckScale(const rund::compute::Backend backend,
                              DomainEvidence &evidence, const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckReductionScale<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckReductionScale<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckReductionScale<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckReductionScale<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckReductionScale<rund::compute::Fixed<16, 16>>(backend, evidence);
  case Domain::Fixed20x44:
    return CheckReductionScale<rund::compute::Fixed<20, 44>>(backend, evidence);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

} // namespace rund_node_collective_modes
