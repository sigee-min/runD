#include "local.hpp"
#include "reference.hpp"

#include <rund/compute.hpp>

#include "contract/target/selection.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <tuple>
#include <type_traits>
#include <utility>

namespace rund::node::test_contract {
namespace declared_detail {

template <class T, std::size_t N, std::size_t FieldCount, class Function>
[[nodiscard]] int CheckDeclaredMultiplyRecord(
    const rund::compute::Backend backend, const std::array<T, N> &input,
    const std::array<std::array<T, N>, FieldCount> &expected,
    const char *const label, const Function function) {
  using namespace rund::compute;
  auto program = (on(rund::node::test_contract::target_for(backend, 2u)))
                     .template map<T>(label, input.size(), function)
                     .compile();
  if (!program) {
    std::fprintf(stderr,
                 "declared multiply compile failed label=%s backend=%u "
                 "reason=%.*s\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    std::fprintf(stderr,
                 "declared multiply resident failed label=%s backend=%u "
                 "reason=%.*s\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<int>(job.error().size()), job.error().data());
    return 2;
  }
  const auto status = job->run();
  if (!status) {
    std::fprintf(stderr,
                 "declared multiply run failed label=%s backend=%u "
                 "reason=%.*s\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<int>(status.error().size()),
                 status.error().data());
    return 3;
  }
  auto output = job->read_all();
  if (!output) {
    return 4;
  }
  constexpr std::size_t actual_fields =
      std::tuple_size_v<std::remove_cvref_t<decltype(*output)>>;
  static_assert(actual_fields == FieldCount);
  bool matches = true;
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (([&] {
       const auto &actual = std::get<I>(*output);
       if (actual.size() != N) {
         matches = false;
         std::fprintf(stderr,
                      "declared multiply field size mismatch label=%s "
                      "backend=%u field=%zu actual=%zu expected=%zu\n",
                      label, static_cast<unsigned>(backend), I, actual.size(),
                      N);
         return;
       }
       for (std::size_t index = 0u; index < N; ++index) {
         if (actual[index] == expected[I][index]) {
           continue;
         }
         matches = false;
         std::fprintf(stderr,
                      "declared multiply mismatch label=%s backend=%u "
                      "field=%zu index=%zu actual=%lld expected=%lld\n",
                      label, static_cast<unsigned>(backend), I, index,
                      static_cast<long long>(actual[index].raw()),
                      static_cast<long long>(expected[I][index].raw()));
         return;
       }
     }()),
     ...);
  }(std::make_index_sequence<FieldCount>{});
  if (!matches) {
    return 5;
  }
  const Stats stats = job->stats();
  if (stats.graph_hash == 0u || stats.output_hash == 0u) {
    std::fprintf(stderr,
                 "declared multiply evidence mismatch label=%s backend=%u "
                 "graph=%llu output=%llu\n",
                 label, static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(stats.graph_hash),
                 static_cast<unsigned long long>(stats.output_hash));
    return 6;
  }
  return 0;
}

template <class T>
[[nodiscard]] int
CheckDeclaredMultiplyType(const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  constexpr Raw one_raw = Raw{1} << T::fraction_bits;
  constexpr Raw half_raw = Raw{1} << (T::fraction_bits - 1u);
  constexpr Raw two_raw = Raw{2} << T::fraction_bits;
  constexpr std::size_t count = 5u;
  const std::array<T, count> input{T::from_raw(one_raw), T::from_raw(1),
                                   T::from_raw(-1), T::max(), T::min()};
  constexpr std::array<Rounding, 4u> roundings{Rounding::TowardZero,
                                               Rounding::Down, Rounding::Up,
                                               Rounding::NearestEven};

  std::array<std::array<T, count>, 13u> rounding_expected{};
  for (std::size_t index = 0u; index < count; ++index) {
    const T value = input[index];
    rounding_expected[0u][index] =
        ReferenceSignedMultiply(value, T::from_raw(-half_raw),
                                Rounding::NearestEven, Overflow::Saturate);
    for (std::size_t policy = 0u; policy < roundings.size(); ++policy) {
      rounding_expected[1u + policy][index] = ReferenceSignedMultiply(
          value, T::from_raw(half_raw), roundings[policy], Overflow::Saturate);
      rounding_expected[5u + policy][index] = ReferenceScaledMultiply(
          value, T::from_raw(half_raw), roundings[policy], Overflow::Saturate);
      rounding_expected[9u + policy][index] = ReferenceUnsignedMultiply(
          value, T::from_raw(half_raw), roundings[policy], Overflow::Saturate);
    }
  }
  if (rounding_expected[0u][0u].raw() != -half_raw) {
    return 7;
  }
  const int rounding_result = CheckDeclaredMultiplyRecord(
      backend, input, rounding_expected,
      T::storage_bits == 32u ? "fixed-16-16-declared-multiply-rounding"
                             : "fixed-20-44-declared-multiply-rounding",
      [](auto value) {
        using V = typename decltype(value)::Value;
        const auto toward =
            quantize<V, Rounding::TowardZero, Overflow::Saturate,
                     Approximation::Exact>(value);
        const auto down = quantize<V, Rounding::Down, Overflow::Saturate,
                                   Approximation::Exact>(value);
        const auto up =
            quantize<V, Rounding::Up, Overflow::Saturate, Approximation::Exact>(
                value);
        const auto nearest =
            quantize<V, Rounding::NearestEven, Overflow::Saturate,
                     Approximation::Exact>(value);
        const auto negative_half =
            neg_positive_fixed(fixed(FixedOp::Half, nearest));
        const auto half_toward = fixed(FixedOp::Half, toward);
        const auto half_down = fixed(FixedOp::Half, down);
        const auto half_up = fixed(FixedOp::Half, up);
        const auto half_nearest = fixed(FixedOp::Half, nearest);
        return record(
            field<DeclaredMultiplySlot<0>>(
                quantize<V, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_fixed(nearest, negative_half))),
            field<DeclaredMultiplySlot<1>>(
                quantize<V, Rounding::TowardZero, Overflow::Saturate,
                         Approximation::Exact>(mul_fixed(toward, half_toward))),
            field<DeclaredMultiplySlot<2>>(
                quantize<V, Rounding::Down, Overflow::Saturate,
                         Approximation::Exact>(mul_fixed(down, half_down))),
            field<DeclaredMultiplySlot<3>>(
                quantize<V, Rounding::Up, Overflow::Saturate,
                         Approximation::Exact>(mul_fixed(up, half_up))),
            field<DeclaredMultiplySlot<4>>(
                quantize<V, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_fixed(nearest, half_nearest))),
            field<DeclaredMultiplySlot<5>>(
                quantize<V, Rounding::TowardZero, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_fixed_scaled(toward, half_toward))),
            field<DeclaredMultiplySlot<6>>(
                quantize<V, Rounding::Down, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_fixed_scaled(down, half_down))),
            field<DeclaredMultiplySlot<7>>(
                quantize<V, Rounding::Up, Overflow::Saturate,
                         Approximation::Exact>(mul_fixed_scaled(up, half_up))),
            field<DeclaredMultiplySlot<8>>(
                quantize<V, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_fixed_scaled(nearest, half_nearest))),
            field<DeclaredMultiplySlot<9>>(
                quantize<V, Rounding::TowardZero, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_unsigned_fixed(toward, half_toward))),
            field<DeclaredMultiplySlot<10>>(
                quantize<V, Rounding::Down, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_unsigned_fixed(down, half_down))),
            field<DeclaredMultiplySlot<11>>(
                quantize<V, Rounding::Up, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_unsigned_fixed(up, half_up))),
            field<DeclaredMultiplySlot<12>>(
                quantize<V, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_unsigned_fixed(nearest, half_nearest))));
      });
  if (rounding_result != 0) {
    return 100 + rounding_result;
  }

  std::array<std::array<T, count>, 9u> boundary_expected{};
  const T two = T::from_raw(two_raw);
  const T one = T::from_raw(one_raw);
  const T negative_one = T::from_raw(-one_raw);
  for (std::size_t index = 0u; index < count; ++index) {
    const T value = input[index];
    boundary_expected[0u][index] = ReferenceSignedMultiply(
        value, two, Rounding::NearestEven, Overflow::Saturate);
    boundary_expected[1u][index] = ReferenceSignedMultiply(
        value, two, Rounding::NearestEven, Overflow::Wrap);
    boundary_expected[2u][index] = ReferenceScaledMultiply(
        value, two, Rounding::NearestEven, Overflow::Saturate);
    boundary_expected[3u][index] = ReferenceScaledMultiply(
        value, two, Rounding::NearestEven, Overflow::Wrap);
    boundary_expected[4u][index] = ReferenceUnsignedMultiply(
        value, two, Rounding::NearestEven, Overflow::Saturate);
    boundary_expected[5u][index] = ReferenceUnsignedMultiply(
        value, two, Rounding::NearestEven, Overflow::Wrap);
    boundary_expected[6u][index] = ReferenceMulAdd(
        value, negative_one, value, Rounding::NearestEven, Overflow::Saturate);
    boundary_expected[7u][index] = ReferenceMulAdd(
        value, one, value, Rounding::NearestEven, Overflow::Saturate);
    boundary_expected[8u][index] = ReferenceMulAdd(
        value, one, value, Rounding::NearestEven, Overflow::Wrap);
  }
  return CheckDeclaredMultiplyRecord(
      backend, input, boundary_expected,
      T::storage_bits == 32u ? "fixed-16-16-declared-multiply-boundary"
                             : "fixed-20-44-declared-multiply-boundary",
      [](auto value) {
        using V = typename decltype(value)::Value;
        const auto saturate =
            quantize<V, Rounding::NearestEven, Overflow::Saturate,
                     Approximation::Exact>(value);
        const auto wrap = quantize<V, Rounding::NearestEven, Overflow::Wrap,
                                   Approximation::Exact>(value);
        const auto one_saturate = fixed_one(saturate);
        const auto one_wrap = fixed_one(wrap);
        const auto two_saturate =
            quantize<V, Rounding::NearestEven, Overflow::Saturate,
                     Approximation::Exact>(one_saturate + one_saturate);
        const auto two_wrap =
            quantize<V, Rounding::NearestEven, Overflow::Wrap,
                     Approximation::Exact>(one_wrap + one_wrap);
        const auto negative_one = neg_positive_fixed(one_saturate);
        return record(
            field<DeclaredMultiplySlot<0>>(
                quantize<V, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_fixed(saturate, two_saturate))),
            field<DeclaredMultiplySlot<1>>(
                quantize<V, Rounding::NearestEven, Overflow::Wrap,
                         Approximation::Exact>(mul_fixed(wrap, two_wrap))),
            field<DeclaredMultiplySlot<2>>(
                quantize<V, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_fixed_scaled(saturate, two_saturate))),
            field<DeclaredMultiplySlot<3>>(
                quantize<V, Rounding::NearestEven, Overflow::Wrap,
                         Approximation::Exact>(
                    mul_fixed_scaled(wrap, two_wrap))),
            field<DeclaredMultiplySlot<4>>(
                quantize<V, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_unsigned_fixed(saturate, two_saturate))),
            field<DeclaredMultiplySlot<5>>(
                quantize<V, Rounding::NearestEven, Overflow::Wrap,
                         Approximation::Exact>(
                    mul_unsigned_fixed(wrap, two_wrap))),
            field<DeclaredMultiplySlot<6>>(
                quantize<V, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_add_fixed(saturate, negative_one, saturate))),
            field<DeclaredMultiplySlot<7>>(
                quantize<V, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(
                    mul_add_fixed(saturate, one_saturate, saturate))),
            field<DeclaredMultiplySlot<8>>(
                quantize<V, Rounding::NearestEven, Overflow::Wrap,
                         Approximation::Exact>(
                    mul_add_fixed(wrap, one_wrap, wrap))));
      });
}

} // namespace declared_detail

[[nodiscard]] int RunFixedDeclaredMultiplyInventory() {
  using rund::compute::Backend;
  for (const Backend backend : selected_compute_backends()) {
    if (const int result = declared_detail::CheckDeclaredMultiplyType<
            rund::compute::Fixed<16u, 16u>>(backend);
        result != 0) {
      return 1000 * static_cast<int>(backend) + 100 + result;
    }
    if (const int result = declared_detail::CheckDeclaredMultiplyType<
            rund::compute::Fixed<20u, 44u>>(backend);
        result != 0) {
      return 1000 * static_cast<int>(backend) + 200 + result;
    }
  }
  return 0;
}

} // namespace rund::node::test_contract
