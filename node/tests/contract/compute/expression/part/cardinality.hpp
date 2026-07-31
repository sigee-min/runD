#pragma once

#include "../support/values.hpp"

#include "../../../target/selection.hpp"

#include <rund/compute/expr/functions/core.hpp>
#include <rund/compute/flow/builder.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <vector>

namespace rund::node::test_contract::expression {
namespace {

template <class Executor, class T, std::size_t N>
[[nodiscard]] int check_parity(Executor &executor,
                               const rund::compute::Backend backend,
                               const std::array<T, N> &input) {
  using namespace rund::compute;
  const auto make = [&](const Backend selected) {
    return (on(rund::node::test_contract::target_for(selected, 2u)))
        .template map<T>("expression-shape-input", input.size(),
                         [](auto value) {
                           using V = typename decltype(value)::Value;
                           if constexpr (is_fixed<V>) {
                             return quantize<V>(value);
                           } else {
                             return value;
                           }
                         })
        .branch([](auto values) {
          const auto exact =
              values.map("expression-shape-exact", [](auto value) {
                using V = typename decltype(value)::Value;
                if constexpr (is_fixed<V>) {
                  return quantize<V>(add_sat(value, fixed_zero(value)));
                } else {
                  return bit_not(bit_not(value));
                }
              });
          const auto bounded =
              values
                  .filter([](auto value) {
                    using V = typename decltype(value)::Value;
                    if constexpr (is_fixed<V>) {
                      return value != fixed_zero(value);
                    } else {
                      return value != V{0};
                    }
                  })
                  .map("expression-shape-bounded", [](auto value) {
                    using V = typename decltype(value)::Value;
                    if constexpr (is_fixed<V>) {
                      return quantize<V>(abs(value));
                    } else {
                      return bit_not(bit_not(value));
                    }
                  });
          const auto scalar =
              values.reduce(Reduce::Sum)
                  .map("expression-shape-scalar", [](auto value) {
                    using V = typename decltype(value)::Value;
                    if constexpr (is_fixed<V>) {
                      return quantize<V>(add_sat(value, fixed_zero(value)));
                    } else {
                      return bit_not(bit_not(value));
                    }
                  });
          return outputs(exact, bounded, scalar);
        })
        .compile();
  };
  auto cpu_program = make(Backend::Cpu);
  auto target_program = make(backend);
  if (!cpu_program || !target_program) {
    const auto &failed = !cpu_program ? cpu_program : target_program;
    std::fprintf(
        stderr, "expression shape compile failed backend=%u reason=%.*s\n",
        static_cast<unsigned>(backend), static_cast<int>(failed.error().size()),
        failed.error().data());
    return 1;
  }
  auto cpu = cpu_program->resident(input);
  auto target = target_program->resident(input);
  if (!cpu || !target || !executor(*cpu) || !executor(*target)) {
    std::fprintf(stderr, "expression shape run failed backend=%u\n",
                 static_cast<unsigned>(backend));
    return 2;
  }
  auto cpu_output = cpu->read_all();
  auto target_output = target->read_all();
  const Stats cpu_stats = cpu->stats();
  const Stats target_stats = target->stats();
  return cpu_output && target_output && *cpu_output == *target_output &&
                 cpu_stats.graph_hash == target_stats.graph_hash &&
                 cpu_stats.output_hash == target_stats.output_hash
             ? 0
             : 3;
}

template <class T, class Executor>
[[nodiscard]] int check_cardinality(Executor &executor,
                                    const rund::compute::Backend backend) {
  if constexpr (is_fixed<T>) {
    const std::array<T, 5u> input{fixed_value<T>(-1, 2), T::zero(),
                                  fixed_value<T>(1, 4), fixed_value<T>(1, 2),
                                  fixed_value<T>(3, 4)};
    return check_parity(executor, backend, input);
  } else if constexpr (std::signed_integral<T>) {
    return check_parity(executor, backend,
                        std::array<T, 5u>{T{-3}, T{0}, T{1}, T{2}, T{4}});
  } else {
    return check_parity(executor, backend,
                        std::array<T, 5u>{T{0}, T{1}, T{2}, T{3}, T{4}});
  }
}

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_golden(Executor &executor,
                               const rund::compute::Backend backend) {
  using namespace rund::compute;
  const T one = fixed_value<T>(1, 1);
  const std::array<T, 1u> input{one};
  auto program =
      (on(rund::node::test_contract::target_for(backend, 2u)))
          .template map<T>(
              "public-fixed-nonlinear-golden", input.size(),
              [](auto value) {
                using V = typename decltype(value)::Value;
                const auto zero = fixed_zero(value);
                const auto half = fixed(FixedOp::Half, value);
                const auto store = [](auto expression) {
                  return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                                  Approximation::Deterministic>(expression);
                };
                return record(field<Field<0>>(store(recip(value))),
                              field<Field<1>>(store(sqrt(value))),
                              field<Field<2>>(store(rsqrt(value))),
                              field<Field<3>>(store(sin(zero))),
                              field<Field<4>>(store(tan(zero))),
                              field<Field<5>>(store(exp(zero))),
                              field<Field<6>>(store(log(value))),
                              field<Field<7>>(store(pow(value, half))),
                              field<Field<8>>(store(atan2(zero, value))));
              })
          .compile();
  if (!program) {
    std::fprintf(stderr,
                 "fixed nonlinear golden compile failed backend=%u "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    std::fprintf(stderr,
                 "fixed nonlinear golden resident failed backend=%u "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(job.error().size()), job.error().data());
    return 2;
  }
  const auto run = executor(*job);
  if (!run) {
    std::fprintf(stderr,
                 "fixed nonlinear golden run failed backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(run.error().size()), run.error().data());
    return 3;
  }
  auto result = job->read_all();
  if (!result) {
    return 4;
  }
  const Stats stats = job->stats();
  const std::vector<T> ones{one};
  const std::vector<T> zeros{T::zero()};
  return stats.graph_hash != 0u && stats.output_hash != 0u &&
                 std::get<0>(*result) == ones && std::get<1>(*result) == ones &&
                 std::get<2>(*result) == ones &&
                 std::get<3>(*result) == zeros &&
                 std::get<4>(*result) == zeros &&
                 std::get<5>(*result) == ones &&
                 std::get<6>(*result) == zeros &&
                 std::get<7>(*result) == ones && std::get<8>(*result) == zeros
             ? 0
             : 5;
}

} // namespace
} // namespace rund::node::test_contract::expression
