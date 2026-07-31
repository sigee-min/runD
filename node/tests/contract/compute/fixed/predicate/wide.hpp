#pragma once

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include "../../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace rund::node::test_contract {
namespace {

template <class Executor>
[[nodiscard]] int
CheckNonFixedPolicyIsAbsent(Executor &executor,
                            const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  const std::vector<std::int32_t> expected{2, 3, 4, 5};
  auto program = (on(rund::node::test_contract::target_for(backend, 2u)))
                     .map<std::int32_t>("nonfixed-policy-absent", input.size(),
                                        [](auto value) { return value + 1; })
                     .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    return 2;
  }
  const auto status = executor(*job);
  if (!status) {
    return 3;
  }
  auto output = job->read();
  return output && *output == expected ? 0 : 4;
}

template <class T, class Executor, class Function>
[[nodiscard]] int CheckFixedWidePredicate(Executor &executor,
                                          const rund::compute::Backend backend,
                                          const char *const label,
                                          const Function function) {
  using namespace rund::compute;
  constexpr std::size_t count = 4u;
  const std::array<T, count> input{T::min(), T::zero(), T::from_raw(1),
                                   T::max()};
  const std::vector<T> expected{T::from_raw(1), T::zero(), T::from_raw(1),
                                T::from_raw(1)};
  auto program = (on(rund::node::test_contract::target_for(backend, 2u)))
                     .template map<T>(label, count, function)
                     .compile();
  if (!program) {
    std::fprintf(stderr,
                 "fixed wide predicate compile failed label=%s backend=%u "
                 "reason=%.*s\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    std::fprintf(stderr,
                 "fixed wide predicate resident failed label=%s backend=%u "
                 "reason=%.*s\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<int>(job.error().size()), job.error().data());
    return 2;
  }
  const auto status = executor(*job);
  if (!status) {
    std::fprintf(stderr,
                 "fixed wide predicate run failed label=%s backend=%u "
                 "reason=%.*s\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<int>(status.error().size()),
                 status.error().data());
    return 3;
  }
  auto output = job->read();
  if (!output || *output != expected) {
    std::fprintf(stderr,
                 "fixed wide predicate output mismatch label=%s backend=%u\n",
                 label, static_cast<unsigned>(backend));
    return 4;
  }
  return 0;
}

template <class T, class Executor>
[[nodiscard]] int
CheckFixedWidePredicateType(Executor &executor,
                            const rund::compute::Backend backend) {
  using namespace rund::compute;
  const int direct = CheckFixedWidePredicate<T>(
      executor, backend, "fixed-wide-direct-select", [](auto value) {
        const auto wide = (value + value) != T::zero();
        return quantize<T>(select(wide, T::from_raw(1), T::zero()));
      });
  if (direct != 0) {
    return direct;
  }
  return CheckFixedWidePredicate<T>(
      executor, backend, "fixed-wide-predicate-connectives", [](auto value) {
        const auto wide = (value + value) != T::zero();
        const auto zero = (value - value) != T::zero();
        const auto predicate =
            predicate_and(predicate_or(wide, zero), predicate_not(zero));
        return quantize<T>(select(predicate, T::from_raw(1), T::zero()));
      });
}

} // namespace

template <class Executor>
[[nodiscard]] int RunFixedWidePredicateInventory(Executor &&executor) {
  using rund::compute::Backend;
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (const int result = CheckNonFixedPolicyIsAbsent(executor, backend);
        result != 0) {
      return 1000 * static_cast<int>(backend) + result;
    }
    if (const int result =
            CheckFixedWidePredicateType<rund::compute::Fixed<16u, 16u>>(
                executor, backend);
        result != 0) {
      return 1000 * static_cast<int>(backend) + 100 + result;
    }
    if (const int result =
            CheckFixedWidePredicateType<rund::compute::Fixed<20u, 44u>>(
                executor, backend);
        result != 0) {
      return 1000 * static_cast<int>(backend) + 200 + result;
    }
  }
  return 0;
}

} // namespace rund::node::test_contract
