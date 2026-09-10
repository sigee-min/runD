#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include <array>
#include <tuple>
#include <vector>

namespace package_compute::fixed_contract {

namespace {

struct FixedSignedProduct final {};
struct FixedScaledProduct final {};
struct FixedUnsignedProduct final {};
struct FixedFusedProduct final {};

template <class T> int DeclaredMultiply() {
  using Raw = typename T::Raw;
  constexpr Raw one = Raw{1} << T::fraction_bits;
  constexpr Raw half = Raw{1} << (T::fraction_bits - 1u);
  const std::array<T, 1u> input{T::from_raw(one)};
  auto output =
      rund::compute::on(rund::compute::Target::cpu(), input)
          .map(
              T::storage_bits == 32u ? "package-fixed-16-16-multiply"
                                     : "package-fixed-20-44-multiply",
              [](auto value) {
                const auto half_value =
                    rund::compute::fixed(rund::compute::FixedOp::Half, value);
                const auto negative_half =
                    rund::compute::neg_positive_fixed(half_value);
                const auto one_value = rund::compute::fixed_one(value);
                const auto negative_one =
                    rund::compute::neg_positive_fixed(one_value);
                return rund::compute::record(
                    rund::compute::field<FixedSignedProduct>(
                        rund::compute::quantize<T>(
                            rund::compute::mul_fixed(value, negative_half))),
                    rund::compute::field<FixedScaledProduct>(
                        rund::compute::quantize<T>(
                            rund::compute::mul_fixed_scaled(value,
                                                            half_value))),
                    rund::compute::field<FixedUnsignedProduct>(
                        rund::compute::quantize<T>(
                            rund::compute::mul_unsigned_fixed(value,
                                                              half_value))),
                    rund::compute::field<FixedFusedProduct>(
                        rund::compute::quantize<T>(rund::compute::mul_add_fixed(
                            value, negative_one, value))));
              })
          .collect();
  if (!output) {
    return output.exit_code();
  }
  if (std::get<0>(*output) != std::vector<T>{T::from_raw(-half)} ||
      std::get<1>(*output) != std::vector<T>{T::from_raw(half)} ||
      std::get<2>(*output) != std::vector<T>{T::from_raw(half)} ||
      std::get<3>(*output) != std::vector<T>{T::zero()}) {
    return 2;
  }
  return 0;
}

} // namespace

int CheckDeclaredMultiply() {
  using rund::compute::Fixed;
  if (const int result = DeclaredMultiply<Fixed<16, 16>>(); result != 0) {
    return result;
  }
  return DeclaredMultiply<Fixed<20, 44>>();
}

} // namespace package_compute::fixed_contract
