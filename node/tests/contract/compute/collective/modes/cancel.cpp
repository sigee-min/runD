#include "model.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund_node_collective_modes {

template <class T>
[[nodiscard]] bool
CheckCrossBlockCancellation(const rund::compute::Backend backend,
                            DomainEvidence &evidence) {
  if constexpr (!(std::is_signed_v<T> ||
                  rund::compute::detail::FixedValue<T>)) {
    return true;
  } else {
    using namespace rund::compute;
    auto fixture = CrossBlockCancellation<T>();
    auto target = flow_on(backend, Target::cpu(2u));
    auto program =
        std::move(target)
            .template input<T>(fixture.input.size())
            .template zip_input<std::uint32_t>(fixture.heads.size())
            .branch([](auto values, auto segments) {
              return outputs(
                  values.scan(Scan::InclusiveSum),
                  values.scan(Scan::ExclusiveSum),
                  values.segmented_scan(segments, Scan::InclusiveSum),
                  values.segmented_scan(segments, Scan::ExclusiveSum));
            })
            .compile();
    if (!program) {
      std::fprintf(stderr,
                   "compute modes cross-block compile backend=%u width=%zu "
                   "reason=%.*s\n",
                   static_cast<unsigned>(backend), sizeof(T),
                   static_cast<int>(program.error().size()),
                   program.error().data());
      return false;
    }
    auto job = program->resident(fixture.input, fixture.heads);
    if (!job) {
      std::fprintf(stderr,
                   "compute modes cross-block resident backend=%u width=%zu "
                   "reason=%.*s\n",
                   static_cast<unsigned>(backend), sizeof(T),
                   static_cast<int>(job.error().size()), job.error().data());
      return false;
    }
    if (!SameSuccess(*job, backend, "cross-block-cancellation",
                     evidence.cross_block_cancellation)) {
      const auto observed = job->read_all();
      const auto raw = [](const T value) -> std::int64_t {
        if constexpr (detail::FixedValue<T>) {
          return static_cast<std::int64_t>(value.raw());
        } else {
          return static_cast<std::int64_t>(value);
        }
      };
      if (observed) {
        const auto dump = [&](const char *const name, const auto &values) {
          std::fprintf(
              stderr,
              "compute modes cross-block observed backend=%u width=%zu "
              "output=%s values=%lld,%lld,%lld,%lld\n",
              static_cast<unsigned>(backend), sizeof(T), name,
              static_cast<long long>(raw(values[255u])),
              static_cast<long long>(raw(values[256u])),
              static_cast<long long>(raw(values[257u])),
              static_cast<long long>(raw(values[511u])));
        };
        dump("inclusive", std::get<0>(*observed));
        dump("exclusive", std::get<1>(*observed));
        dump("segmented-inclusive", std::get<2>(*observed));
        dump("segmented-exclusive", std::get<3>(*observed));
      }
      return false;
    }
    auto output = job->read_all();
    const bool same = output && std::get<0>(*output) == fixture.inclusive &&
                      std::get<1>(*output) == fixture.exclusive &&
                      std::get<2>(*output) == fixture.inclusive &&
                      std::get<3>(*output) == fixture.exclusive;
    if (!same) {
      std::fprintf(stderr,
                   "compute modes cross-block golden mismatch backend=%u "
                   "width=%zu\n",
                   static_cast<unsigned>(backend), sizeof(T));
    }
    if constexpr (detail::FixedValue<T> && sizeof(T) == 8u) {
      if (same && backend == Backend::Vulkan) {
        // Descriptor/resource churn used to expose a Vulkan-only stale
        // fixed64 hierarchy after earlier collective families had completed.
        // Recreate the prepared owner while retaining one Program/pipeline
        // cache authority and verify every published bit pattern.
        for (std::size_t attempt = 0u; attempt < 32u; ++attempt) {
          auto witness = program->resident(fixture.input, fixture.heads);
          if (!witness || !witness->run()) {
            return false;
          }
          const auto repeated = witness->read_all();
          if (!repeated ||
              std::get<0>(*repeated) != fixture.inclusive ||
              std::get<1>(*repeated) != fixture.exclusive ||
              std::get<2>(*repeated) != fixture.inclusive ||
              std::get<3>(*repeated) != fixture.exclusive) {
            std::fprintf(stderr,
                         "compute modes cross-block churn backend=%u "
                         "width=%zu attempt=%zu\n",
                         static_cast<unsigned>(backend), sizeof(T), attempt);
            if (repeated) {
              const auto dump = [&](const char *const name,
                                    const auto &values) {
                std::fprintf(
                    stderr,
                    "compute modes cross-block churn output=%s "
                    "values=%lld,%lld,%lld,%lld\n",
                    name,
                    static_cast<long long>(values[255u].raw()),
                    static_cast<long long>(values[256u].raw()),
                    static_cast<long long>(values[257u].raw()),
                    static_cast<long long>(values[511u].raw()));
              };
              dump("inclusive", std::get<0>(*repeated));
              dump("exclusive", std::get<1>(*repeated));
              dump("segmented-inclusive", std::get<2>(*repeated));
              dump("segmented-exclusive", std::get<3>(*repeated));
            }
            return false;
          }
        }
      }
    }
    return same;
  }
}

template <class T>
[[nodiscard]] bool
CheckReductionCancellation(const rund::compute::Backend backend,
                           DomainEvidence &evidence) {
  using namespace rund::compute;
  constexpr std::size_t kReduceCount = 512u;
  std::vector<T> input(kReduceCount, Zero<T>());
  std::vector<T> segmented_input(kReduceCount, Zero<T>());
  if constexpr (std::is_signed_v<T> || detail::FixedValue<T>) {
    T half{};
    T negative_half{};
    if constexpr (detail::FixedValue<T>) {
      using Raw = typename T::Raw;
      half = T::from_raw(static_cast<Raw>(T::max().raw() / Raw{2}));
      negative_half = T::from_raw(static_cast<Raw>(-half.raw()));
    } else {
      half = static_cast<T>(std::numeric_limits<T>::max() / T{2});
      negative_half = static_cast<T>(-half);
    }
    // A later block's local sum exceeds T, while the earlier negative block
    // restores the one observable global reduction result to max.
    input[0u] = negative_half;
    input[256u] = Maximum<T>();
    input[257u] = half;
    // Put all three nonzero terms in one lane's 64-stride subsequence. The
    // lane-local sum exceeds T before the last term cancels it, so a native-
    // width lane accumulator would fail even though the exact segment result
    // is representable.
    segmented_input[0u] = Maximum<T>();
    segmented_input[64u] = half;
    segmented_input[128u] = negative_half;
  } else {
    // Unsigned sums are monotone and therefore have no cancellation
    // counterexample; still cover their exact boundary value in this matrix.
    input[0u] = Maximum<T>();
    segmented_input[0u] = Maximum<T>();
  }
  std::vector<std::uint32_t> heads(kReduceCount, 0u);
  heads[0u] = 1u;
  auto reduce_target = flow_on(backend, Target::cpu(2u));
  auto reduce =
      std::move(reduce_target)
          .template input<T>(input.size())
          .branch([](auto values) { return values.reduce(Reduce::Sum); })
          .compile();
  auto segmented_target = flow_on(backend, Target::cpu(2u));
  auto segmented = std::move(segmented_target)
                       .template input<T>(segmented_input.size())
                       .template zip_input<std::uint32_t>(heads.size())
                       .branch([](auto values, auto segments) {
                         return values.segmented_reduce(segments, Reduce::Sum);
                       })
                       .compile();
  if (!reduce || !segmented) {
    std::fprintf(stderr,
                 "compute modes reduction cancellation compile backend=%u "
                 "width=%zu reduce=%.*s segmented=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(reduce.error().size()), reduce.error().data(),
                 static_cast<int>(segmented.error().size()),
                 segmented.error().data());
    return false;
  }
  auto reduce_job = reduce->resident(input);
  auto segmented_job = segmented->resident(segmented_input, heads);
  if (!reduce_job ||
      !SameSuccess(*reduce_job, backend, "reduce-cancellation",
                   evidence.reduce_cancellation) ||
      !segmented_job ||
      !SameSuccess(*segmented_job, backend, "segmented-reduce-cancellation",
                   evidence.segmented_reduce_cancellation)) {
    return false;
  }
  auto reduced = reduce_job->read();
  auto segmented_output = segmented_job->read();
  const std::vector<T> reduced_expected{Maximum<T>()};
  std::vector<T> expected(segmented_input.size(), Zero<T>());
  expected.front() = Maximum<T>();
  const bool same = reduced && *reduced == reduced_expected &&
                    segmented_output && *segmented_output == expected;
  if (!same) {
    const auto raw = [](const T value) -> std::int64_t {
      if constexpr (detail::FixedValue<T>) {
        return static_cast<std::int64_t>(value.raw());
      } else {
        return static_cast<std::int64_t>(value);
      }
    };
    const std::int64_t reduced_value =
        reduced && !reduced->empty() ? raw(reduced->front()) : std::int64_t{};
    const auto &segmented_values =
        segmented_output ? *segmented_output : expected;
    const std::int64_t segment = segmented_values.empty()
                                     ? std::int64_t{}
                                     : raw(segmented_values.front());
    std::fprintf(stderr,
                 "compute modes reduction cancellation golden mismatch "
                 "backend=%u width=%zu reduce=%lld segment=%lld count=%zu "
                 "expected=%lld\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<long long>(reduced_value),
                 static_cast<long long>(segment), segmented_values.size(),
                 static_cast<long long>(raw(Maximum<T>())));
  }
  return same;
}

[[nodiscard]] bool CheckCrossBlock(const rund::compute::Backend backend,
                                   DomainEvidence &evidence,
                                   const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckCrossBlockCancellation<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckCrossBlockCancellation<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckCrossBlockCancellation<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckCrossBlockCancellation<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckCrossBlockCancellation<rund::compute::Fixed<16, 16>>(backend,
                                                                     evidence);
  case Domain::Fixed20x44:
    return CheckCrossBlockCancellation<rund::compute::Fixed<20, 44>>(backend,
                                                                     evidence);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

[[nodiscard]] bool
CheckReductionCancellation(const rund::compute::Backend backend,
                           DomainEvidence &evidence, const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckReductionCancellation<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckReductionCancellation<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckReductionCancellation<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckReductionCancellation<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckReductionCancellation<rund::compute::Fixed<16, 16>>(backend,
                                                                    evidence);
  case Domain::Fixed20x44:
    return CheckReductionCancellation<rund::compute::Fixed<20, 44>>(backend,
                                                                    evidence);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

} // namespace rund_node_collective_modes
