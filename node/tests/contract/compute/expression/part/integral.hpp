#pragma once

#include "../support/record.hpp"
#include "../support/single.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <std::integral T, class Executor>
[[nodiscard]] int check_operators(Executor &executor,
                                  const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<T, 6u> input = [] {
    if constexpr (std::signed_integral<T>) {
      return std::array<T, 6u>{T{-9}, T{-3}, T{-1}, T{0}, T{4}, T{11}};
    } else {
      return std::array<T, 6u>{T{0}, T{1}, T{3}, T{7}, T{11}, T{19}};
    }
  }();

  const int arithmetic = check_record_parity(
      executor, backend, input, "public-integral-arithmetic", [](auto value) {
        using V = typename decltype(value)::Value;
        return record(
            field<Field<0>>(value + V{3}), field<Field<1>>(value - V{2}),
            field<Field<2>>(value * V{3}), field<Field<3>>(value / V{2}),
            field<Field<4>>(value & V{6}), field<Field<5>>(value | V{5}),
            field<Field<6>>(value ^ V{3}), field<Field<7>>(bit_not(value)),
            field<Field<8>>(min(value, V{2})),
            field<Field<9>>(max(value, V{2})),
            field<Field<10>>(clamp(value, V{1}, V{6})),
            field<Field<11>>(shl<1u>(value)),
            field<Field<12>>(shr_logical<1u>(value)),
            field<Field<13>>(mul_wrap(value, value)),
            field<Field<14>>(select(value < V{2}, value, V{2})),
            field<Field<15>>(select(value != V{0}, V{1}, V{0})));
      });
  if (arithmetic != 0) {
    return arithmetic;
  }

  if constexpr (std::signed_integral<T>) {
    const int signed_result = check_record_parity(
        executor, backend, input, "public-signed-functions", [](auto value) {
          using V = typename decltype(value)::Value;
          const auto low = value < V{0};
          const auto small = value <= V{4};
          return record(field<Field<0>>(-value), field<Field<1>>(abs(value)),
                        field<Field<2>>(abs_magnitude(value)),
                        field<Field<3>>(sign(value)),
                        field<Field<4>>(add_sat(value, value)),
                        field<Field<5>>(sub_sat(value, value)),
                        field<Field<6>>(shr_arithmetic<1u>(value)),
                        field<Field<7>>(select(value == V{0}, V{1}, V{0})),
                        field<Field<8>>(select(value != V{0}, V{1}, V{0})),
                        field<Field<9>>(select(value < V{0}, V{1}, V{0})),
                        field<Field<10>>(select(value <= V{0}, V{1}, V{0})),
                        field<Field<11>>(select(value > V{0}, V{1}, V{0})),
                        field<Field<12>>(select(value >= V{0}, V{1}, V{0})),
                        field<Field<13>>(select(!low, V{1}, V{0})),
                        field<Field<14>>(select(low && small, V{1}, V{0})),
                        field<Field<15>>(select(low || small, V{1}, V{0})));
        });
    if (signed_result != 0) {
      return 10 + signed_result;
    }
  } else {
    const int unsigned_result = check_record_parity(
        executor, backend, input, "public-unsigned-functions", [](auto value) {
          using V = typename decltype(value)::Value;
          const auto low = value < V{4};
          const auto nonzero = value != V{0};
          return record(field<Field<0>>(add_sat_unsigned(value, value)),
                        field<Field<1>>(select(value == V{0}, V{1}, V{0})),
                        field<Field<2>>(select(value != V{0}, V{1}, V{0})),
                        field<Field<3>>(select(value < V{4}, V{1}, V{0})),
                        field<Field<4>>(select(value <= V{4}, V{1}, V{0})),
                        field<Field<5>>(select(value > V{4}, V{1}, V{0})),
                        field<Field<6>>(select(value >= V{4}, V{1}, V{0})),
                        field<Field<7>>(select(!low, V{1}, V{0})),
                        field<Field<8>>(select(low && nonzero, V{1}, V{0})),
                        field<Field<9>>(select(low || nonzero, V{1}, V{0})),
                        field<Field<10>>(select(low, V{7}, V{9})),
                        field<Field<11>>(min(value, V{7})),
                        field<Field<12>>(max(value, V{7})),
                        field<Field<13>>(clamp(value, V{2}, V{11})),
                        field<Field<14>>(shl<0u>(value)),
                        field<Field<15>>(shr_logical<0u>(value)));
        });
    if (unsigned_result != 0) {
      return 20 + unsigned_result;
    }
  }
  const int hash_result = check_record_parity(
      executor, backend, input, "public-integral-hash-helpers", [](auto value) {
        using V = typename decltype(value)::Value;
        const auto seed = value ^ V{7};
        return record(field<Field<0>>(hash(value)),
                      field<Field<1>>(hash(value, seed)),
                      field<Field<2>>(hash(HashOp::Unit, value)),
                      field<Field<3>>(hash(HashOp::Unit, value, seed)));
      });
  if (hash_result != 0) {
    return 30 + hash_result;
  }
  const int mask_result = check_single_parity(
      executor, backend, input, "public-integral-mask", [](auto value) {
        using V = typename decltype(value)::Value;
        return mask(value != V{0});
      });
  return mask_result == 0 ? 0 : 40 + mask_result;
}

template <std::integral T, class Executor>
[[nodiscard]] int
check_unsigned_high_bit(Executor &executor,
                        const rund::compute::Backend backend) {
  static_assert(std::unsigned_integral<T>);
  using namespace rund::compute;
  constexpr T high = T{1} << (sizeof(T) * 8u - 1u);
  constexpr T maximum = std::numeric_limits<T>::max();
  const std::array<T, 6u> input{T{0},
                                T{1},
                                static_cast<T>(high - T{1}),
                                high,
                                static_cast<T>(maximum - T{1}),
                                maximum};
  const auto target = on(rund::node::test_contract::target_for(backend, 2u));
  auto program =
      target.template input<T>(input.size())
          .branch([](auto values) {
            const auto upper =
                values.map("unsigned-high-max",
                           capture([](auto value,
                                      auto bound) { return max(value, bound); },
                                   high));
            return outputs(
                values.map("unsigned-high-lt",
                           capture(
                               [](auto value, auto bound) {
                                 return select(value < bound, T{1}, T{0});
                               },
                               maximum)),
                values.map("unsigned-high-le",
                           capture(
                               [](auto value, auto bound) {
                                 return select(value <= bound, T{1}, T{0});
                               },
                               high)),
                values.map("unsigned-high-gt",
                           capture(
                               [](auto value, auto bound) {
                                 return select(value > bound, T{1}, T{0});
                               },
                               high)),
                values.map("unsigned-high-ge",
                           capture(
                               [](auto value, auto bound) {
                                 return select(value >= bound, T{1}, T{0});
                               },
                               high)),
                values.map("unsigned-high-min",
                           capture([](auto value,
                                      auto bound) { return min(value, bound); },
                                   high)),
                upper,
                values.map("unsigned-high-clamp",
                           capture(
                               [](auto value, auto bound) {
                                 return clamp(value, T{1}, bound);
                               },
                               static_cast<T>(maximum - T{1}))),
                values.map(
                    "unsigned-high-mask",
                    capture([](auto value,
                               auto bound) { return mask(value >= bound); },
                            high)));
          })
          .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(input);
  if (!job || !executor(*job)) {
    return 2;
  }
  auto output = job->read_all();
  if (!output ||
      std::get<0>(*output) !=
          std::vector<T>{T{1}, T{1}, T{1}, T{1}, T{1}, T{0}} ||
      std::get<1>(*output) !=
          std::vector<T>{T{1}, T{1}, T{1}, T{1}, T{0}, T{0}} ||
      std::get<2>(*output) !=
          std::vector<T>{T{0}, T{0}, T{0}, T{0}, T{1}, T{1}} ||
      std::get<3>(*output) !=
          std::vector<T>{T{0}, T{0}, T{0}, T{1}, T{1}, T{1}} ||
      std::get<4>(*output) != std::vector<T>{T{0}, T{1},
                                             static_cast<T>(high - T{1}), high,
                                             high, high} ||
      std::get<5>(*output) != std::vector<T>{high, high, high, high,
                                             static_cast<T>(maximum - T{1}),
                                             maximum} ||
      std::get<6>(*output) != std::vector<T>{T{1}, T{1},
                                             static_cast<T>(high - T{1}), high,
                                             static_cast<T>(maximum - T{1}),
                                             static_cast<T>(maximum - T{1})} ||
      std::get<7>(*output) !=
          std::vector<std::uint32_t>{0u, 0u, 0u, 1u, 1u, 1u}) {
    std::fprintf(stderr,
                 "expression unsigned high-bit mismatch backend=%u bytes=%zu\n",
                 static_cast<unsigned>(backend), sizeof(T));
    return 3;
  }
  const Stats stats = job->stats();
  return stats.graph_hash != 0u && stats.output_hash != 0u ? 0 : 4;
}

template <std::integral T, class Executor>
[[nodiscard]] int check_boundaries(Executor &executor,
                                   const rund::compute::Backend backend) {
  using namespace rund::compute;
  if constexpr (std::signed_integral<T>) {
    constexpr T minimum = std::numeric_limits<T>::min();
    constexpr T maximum = std::numeric_limits<T>::max();
    const std::array<T, 5u> input{minimum, maximum, T{0}, T{1}, T{-1}};
    if (const int result = check_record_parity(
            executor, backend, input, "public-signed-absolute-boundaries",
            [](auto value) {
              using V = typename decltype(value)::Value;
              const auto one = select(value == value, V{1}, V{1});
              return record(field<Field<0>>(add_sat(value, value)),
                            field<Field<1>>(sub_sat(value, one)),
                            field<Field<2>>(mul_wrap(value, value)),
                            field<Field<3>>(bit_not(value)),
                            field<Field<4>>(shl<1u>(value)),
                            field<Field<5>>(shr_logical<1u>(value)),
                            field<Field<6>>(shr_arithmetic<1u>(value)),
                            field<Field<7>>(abs(value)),
                            field<Field<8>>(abs_magnitude(value)),
                            field<Field<9>>(sign(value)));
            });
        result != 0) {
      return result;
    }
    const std::array<T, 5u> saturated{minimum, maximum, T{0}, T{2}, T{-2}};
    if (const int result = check_single_expected(
            executor, backend, input, saturated,
            "public-signed-add-saturating-golden",
            [](auto value) { return add_sat(value, value); });
        result != 0) {
      return 10 + result;
    }
    const std::array<T, 5u> magnitude{maximum, maximum, T{0}, T{1}, T{1}};
    if (const int result = check_single_expected(
            executor, backend, input, magnitude, "public-signed-abs-golden",
            [](auto value) { return abs(value); });
        result != 0) {
      return 20 + result;
    }
    const std::array<T, 5u> safe{static_cast<T>(minimum + T{1}), maximum, T{0},
                                 T{1}, T{-1}};
    const std::array<T, 5u> negated{maximum, static_cast<T>(minimum + T{1}),
                                    T{0}, T{-1}, T{1}};
    if (const int result = check_single_parity(
            executor, backend, safe, "public-signed-safe-negation",
            [](auto value) { return -value; });
        result != 0) {
      return 30 + result;
    }
    if (const int result =
            check_single_expected(executor, backend, safe, negated,
                                  "public-signed-safe-negation-golden",
                                  [](auto value) { return -value; });
        result != 0) {
      return 40 + result;
    }
  } else {
    constexpr T maximum = std::numeric_limits<T>::max();
    const std::array<T, 4u> input{T{0}, T{1}, maximum,
                                  static_cast<T>(maximum - T{1})};
    if (const int result = check_record_parity(
            executor, backend, input, "public-unsigned-absolute-boundaries",
            [](auto value) {
              return record(field<Field<0>>(add_sat_unsigned(value, value)),
                            field<Field<1>>(mul_wrap(value, value)),
                            field<Field<2>>(bit_not(value)),
                            field<Field<3>>(shl<1u>(value)),
                            field<Field<4>>(shr_logical<1u>(value)),
                            field<Field<5>>(min(value, value)),
                            field<Field<6>>(max(value, value)));
            });
        result != 0) {
      return 50 + result;
    }
    const std::array<T, 4u> saturated{T{0}, T{2}, maximum, maximum};
    if (const int result = check_single_expected(
            executor, backend, input, saturated,
            "public-unsigned-add-saturating-golden",
            [](auto value) { return add_sat_unsigned(value, value); });
        result != 0) {
      return 60 + result;
    }
    if (const int result = check_unsigned_high_bit<T>(executor, backend);
        result != 0) {
      return 70 + result;
    }
  }
  return 0;
}

} // namespace
} // namespace rund::node::test_contract::expression
