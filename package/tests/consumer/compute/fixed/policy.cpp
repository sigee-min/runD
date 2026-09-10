#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include <array>
#include <cstdio>
#include <vector>

namespace package_compute::fixed_contract {

namespace {

template <class T, class Side> int MixedPolicyFlow() {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  using SideRaw = typename Side::Raw;
  const Raw one_raw = static_cast<Raw>(Raw{1} << T::fraction_bits);
  const SideRaw side_one_raw =
      static_cast<SideRaw>(SideRaw{1} << Side::fraction_bits);
  const T zero = T::zero();
  const T one = T::from_raw(one_raw);
  const T two = T::from_raw(static_cast<Raw>(one_raw * Raw{2}));
  const Side side_zero = Side::zero();
  const Side side_one = Side::from_raw(side_one_raw);
  const Side side_two =
      Side::from_raw(static_cast<SideRaw>(side_one_raw * SideRaw{2}));
  const std::array<T, 4u> left{zero, one, two, T::from_raw(-one_raw)};
  const std::array<Side, 4u> right{side_one, side_zero,
                                   Side::from_raw(-side_one_raw), side_two};

  auto program =
      on(Target::cpu(2u))
          .template input<T>(left.size())
          .template zip_input<Side>(right.size())
          .branch([](auto first, auto second) {
            const auto normalized =
                zip(first, second)
                    .map("package-fixed-mixed-normalize",
                         [](auto left_value, auto right_value) {
                           const auto stored_left =
                               quantize<T, Rounding::Down, Overflow::Wrap>(
                                   left_value);
                           const auto stored_right =
                               quantize<T, Rounding::Down, Overflow::Wrap>(
                                   right_value);
                           return quantize<T, Rounding::Down, Overflow::Wrap>(
                               stored_left + stored_right);
                         });
            const auto mapped =
                normalized.map("package-fixed-policy-map", [](auto value) {
                  return quantize<T, Rounding::Down, Overflow::Wrap>(value);
                });
            return mapped.combine(
                "package-fixed-policy-combine", normalized,
                [](auto left_value, auto right_value) {
                  return quantize<T, Rounding::Down, Overflow::Wrap>(
                      left_value + right_value);
                });
          })
          .compile();
  if (!program) {
    std::fprintf(stderr, "package mixed Fixed compile: %.*s\n",
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return program.exit_code();
  }
  const auto graph = program->graph();
  if (graph.inputs.size() != 2u || graph.outputs.size() != 1u) {
    return 2;
  }
  const auto &side_input = graph.resources[graph.inputs[1u] - 1u];
  const auto &output = graph.resources[graph.outputs.front() - 1u];
  if (side_input.integer_bits != Side::integer_bits ||
      side_input.fraction_bits != Side::fraction_bits ||
      output.integer_bits != T::integer_bits ||
      output.fraction_bits != T::fraction_bits ||
      output.rounding != Rounding::Down || output.overflow != Overflow::Wrap) {
    return 2;
  }
  auto job = program->resident(left, right);
  if (!job) {
    return job.exit_code();
  }
  const auto executed = job->run();
  if (!executed) {
    return executed.exit_code();
  }
  auto result = job->read();
  if (!result) {
    return result.exit_code();
  }
  return *result == std::vector<T>{two, two, two, two} ? 0 : 2;
}

} // namespace

int CheckMixedPolicy() {
  using rund::compute::Fixed;
  if (const int result = MixedPolicyFlow<Fixed<16, 16>, Fixed<17, 15>>();
      result != 0) {
    return result;
  }
  return MixedPolicyFlow<Fixed<20, 44>, Fixed<21, 43>>();
}

} // namespace package_compute::fixed_contract
