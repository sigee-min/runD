#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include "../../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace rund::node::test_contract::numeric {
namespace {

template <class Input, std::size_t N, class Function>
[[nodiscard]] int CheckElementParity(const std::array<Input, N> &input,
                                     const char *const label,
                                     Function function) {
  using namespace rund::compute;
  const auto make = [&](const Backend selected) {
    return on(rund::node::test_contract::target_for(selected))
        .template map<Input>(label, input.size(), function)
        .compile();
  };
  auto cpu_program = make(Backend::Cpu);
  if (!cpu_program) {
    return 1;
  }
  auto cpu = cpu_program->resident(input);
  if (!cpu || !cpu->run()) {
    return 2;
  }
  const Stats cpu_stats = cpu->stats();
  auto cpu_output = cpu->read();
  if (!cpu_output) {
    return 3;
  }
  for (const Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    auto target_program = make(backend);
    if (!target_program) {
      return 1;
    }
    auto target = target_program->resident(input);
    if (!target || !target->run()) {
      return 2;
    }
    const Stats target_stats = target->stats();
    auto target_output = target->read();
    if (!target_output || *cpu_output != *target_output ||
        cpu_stats.graph_hash != target_stats.graph_hash ||
        cpu_stats.output_hash != target_stats.output_hash) {
      return 3;
    }
  }
  return 0;
}

[[nodiscard]] int CheckArbitraryFormats() {
  using namespace rund::compute;
  using Storage32 = Fixed<16, 16>;
  using Storage64 = Fixed<20, 44>;
  const std::array storage32{Storage32::from_raw(98304),
                             Storage32::from_raw(-98304),
                             Storage32::from_raw(32768), Storage32::max()};
  if (const int result = CheckElementParity(
          storage32, "fixed-i16-f16-square",
          [](auto value) { return quantize<Storage32>(value * value); });
      result != 0) {
    return 10 + result;
  }
  const std::array fused32{Storage32::max(), Storage32::min(),
                           Storage32::from_raw(98304),
                           Storage32::from_raw(-98304)};
  if (const int result = CheckElementParity(
          fused32, "fixed-i16-f16-multiply-add-carry",
          [](auto value) {
            return quantize<Storage32>(mul_add_fixed(value, value, value));
          });
      result != 0) {
    return 12 + result;
  }
  if (const int result = CheckElementParity(
          storage32, "fixed-i16-f16-divide",
          capture(
              [](auto value, auto divisor) {
                return quantize<Storage32, Rounding::NearestEven,
                                Overflow::Saturate,
                                Approximation::Deterministic>(value / divisor);
              },
              Storage32::from_raw(32768)));
      result != 0) {
    return 14 + result;
  }
  if (const int result = CheckElementParity(
          storage32, "fixed-i16-f16-sqrt",
          [](auto value) {
            return quantize<Storage32, Rounding::NearestEven,
                            Overflow::Saturate, Approximation::Deterministic>(
                sqrt(value));
          });
      result != 0) {
    return 18 + result;
  }
  const std::array transcendental32{
      Storage32::from_raw(4096), Storage32::from_raw(8192),
      Storage32::from_raw(12288), Storage32::from_raw(32768)};
  if (const int result = CheckElementParity(
          transcendental32, "fixed-i16-f16-transcendental",
          capture(
              [](auto value, auto axis) {
                return quantize<Storage32, Rounding::NearestEven,
                                Overflow::Saturate,
                                Approximation::Deterministic>(
                    sin(value) + cos(value) + tan(value) + exp(value) +
                    log(value) + atan2(value, axis));
              },
              Storage32::from_raw(65536)));
      result != 0) {
    return 19 + result;
  }

  constexpr std::int64_t kOne64 = std::int64_t{1} << 44;
  const std::array storage64{Storage64::from_raw(kOne64 + (kOne64 >> 1)),
                             Storage64::from_raw(-(kOne64 + (kOne64 >> 1))),
                             Storage64::from_raw(kOne64 >> 1),
                             Storage64::max()};
  if (const int result = CheckElementParity(
          storage64, "fixed-i20-f44-square",
          [](auto value) { return quantize<Storage64>(value * value); });
      result != 0) {
    return 20 + result;
  }
  if (const int result = CheckElementParity(
          storage64, "fixed-i20-f44-recip",
          [](auto value) {
            return quantize<Storage64, Rounding::NearestEven,
                            Overflow::Saturate, Approximation::Deterministic>(
                recip(value));
          });
      result != 0) {
    return 24 + result;
  }
  if (const int result = CheckElementParity(
          storage64, "fixed-i20-f44-sqrt",
          [](auto value) {
            return quantize<Storage64, Rounding::NearestEven,
                            Overflow::Saturate, Approximation::Deterministic>(
                sqrt(value));
          });
      result != 0) {
    return 28 + result;
  }
  const std::array transcendental64{
      Storage64::from_raw(kOne64 >> 4), Storage64::from_raw(kOne64 >> 3),
      Storage64::from_raw((kOne64 * 3) >> 4), Storage64::from_raw(kOne64 >> 1)};
  if (const int result = CheckElementParity(
          transcendental64, "fixed-i20-f44-transcendental",
          capture(
              [](auto value, auto axis) {
                return quantize<Storage64, Rounding::NearestEven,
                                Overflow::Saturate,
                                Approximation::Deterministic>(
                    sin(value) + cos(value) + tan(value) + exp(value) +
                    log(value) + atan2(value, axis));
              },
              Storage64::from_raw(kOne64)));
      result != 0) {
    return 29 + result;
  }

  using HalfFraction = Fixed<17, 15>;
  const std::array rounding{Storage32::from_raw(1), Storage32::from_raw(3),
                            Storage32::from_raw(-1), Storage32::from_raw(-3)};
  if (const int result = CheckElementParity(
          rounding, "fixed-round-nearest-even",
          [](auto value) {
            return quantize<HalfFraction, Rounding::NearestEven>(value);
          });
      result != 0) {
    return 30 + result;
  }
  if (const int result = CheckElementParity(
          rounding, "fixed-round-down",
          [](auto value) {
            return quantize<HalfFraction, Rounding::Down>(value);
          });
      result != 0) {
    return 40 + result;
  }
  if (const int result = CheckElementParity(
          rounding, "fixed-round-up",
          [](auto value) {
            return quantize<HalfFraction, Rounding::Up>(value);
          });
      result != 0) {
    return 50 + result;
  }

  using HalfFraction64 = Fixed<21, 43>;
  const std::array rounding64{Storage64::from_raw(1), Storage64::from_raw(3),
                              Storage64::from_raw(-1), Storage64::from_raw(-3)};
  const auto nearest64 = [](auto value) {
    return quantize<HalfFraction64, Rounding::NearestEven>(value);
  };
  const auto down64 = [](auto value) {
    return quantize<HalfFraction64, Rounding::Down>(value);
  };
  const auto up64 = [](auto value) {
    return quantize<HalfFraction64, Rounding::Up>(value);
  };
  if (const int result = CheckElementParity(
          rounding64, "fixed-i21-f43-round-nearest-even", nearest64);
      result != 0) {
    return 80 + result;
  }
  if (const int result =
          CheckElementParity(rounding64, "fixed-i21-f43-round-down", down64);
      result != 0) {
    return 90 + result;
  }
  if (const int result =
          CheckElementParity(rounding64, "fixed-i21-f43-round-up", up64);
      result != 0) {
    return 100 + result;
  }
  for (const Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    const auto nearest64_output =
        on(rund::node::test_contract::target_for(backend), rounding64)
            .map("fixed-i21-f43-nearest-golden", nearest64)
            .collect();
    const auto down64_output =
        on(rund::node::test_contract::target_for(backend), rounding64)
            .map("fixed-i21-f43-down-golden", down64)
            .collect();
    const auto up64_output =
        on(rund::node::test_contract::target_for(backend), rounding64)
            .map("fixed-i21-f43-up-golden", up64)
            .collect();
    if (!nearest64_output || !down64_output || !up64_output ||
        *nearest64_output !=
            std::vector<HalfFraction64>{
                HalfFraction64::from_raw(0), HalfFraction64::from_raw(2),
                HalfFraction64::from_raw(0), HalfFraction64::from_raw(-2)} ||
        *down64_output !=
            std::vector<HalfFraction64>{
                HalfFraction64::from_raw(0), HalfFraction64::from_raw(1),
                HalfFraction64::from_raw(-1), HalfFraction64::from_raw(-2)} ||
        *up64_output !=
            std::vector<HalfFraction64>{
                HalfFraction64::from_raw(1), HalfFraction64::from_raw(2),
                HalfFraction64::from_raw(0), HalfFraction64::from_raw(-1)}) {
      return 104;
    }
  }

  const std::array overflow{Storage32::max(), Storage32::min(),
                            Storage32::from_raw(98304)};
  if (const int result = CheckElementParity(
          overflow, "fixed-overflow-saturate",
          [](auto value) {
            return quantize<Storage32, Rounding::NearestEven,
                            Overflow::Saturate>(value * value);
          });
      result != 0) {
    return 60 + result;
  }
  if (const int result = CheckElementParity(
          overflow, "fixed-overflow-wrap",
          [](auto value) {
            return quantize<Storage32, Rounding::NearestEven, Overflow::Wrap>(
                value * value);
          });
      result != 0) {
    return 70 + result;
  }
  const std::array overflow64{Storage64::from_raw(std::int64_t{1} << 62),
                              Storage64::min(),
                              Storage64::from_raw(kOne64 + (kOne64 >> 1))};
  const auto saturate64 = [](auto value) {
    return quantize<Storage64, Rounding::NearestEven, Overflow::Saturate>(
        value * value);
  };
  const auto wrap64 = [](auto value) {
    return quantize<Storage64, Rounding::NearestEven, Overflow::Wrap>(value *
                                                                      value);
  };
  if (const int result = CheckElementParity(
          overflow64, "fixed-i20-f44-overflow-saturate", saturate64);
      result != 0) {
    return 110 + result;
  }
  if (const int result =
          CheckElementParity(overflow64, "fixed-i20-f44-overflow-wrap", wrap64);
      result != 0) {
    return 120 + result;
  }
  for (const Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    const std::int64_t exact_square = std::int64_t{9} << 42;
    const auto saturate64_output =
        on(rund::node::test_contract::target_for(backend), overflow64)
            .map("fixed-i20-f44-saturate-golden", saturate64)
            .collect();
    const auto wrap64_output =
        on(rund::node::test_contract::target_for(backend), overflow64)
            .map("fixed-i20-f44-wrap-golden", wrap64)
            .collect();
    if (!saturate64_output || !wrap64_output ||
        *saturate64_output !=
            std::vector<Storage64>{Storage64::max(), Storage64::max(),
                                   Storage64::from_raw(exact_square)} ||
        *wrap64_output !=
            std::vector<Storage64>{Storage64::zero(), Storage64::zero(),
                                   Storage64::from_raw(exact_square)}) {
      return 124;
    }
  }
  return 0;
}
} // namespace

int CheckFormats() { return CheckArbitraryFormats(); }

} // namespace rund::node::test_contract::numeric
