#pragma once

#include "../../support/record.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_approx(Executor &executor,
                               const rund::compute::Backend backend,
                               const std::array<T, 3u> &input) {
  using namespace rund::compute;
  if (const int result = check_record_parity(
          executor, backend, input,
          "public-fixed-approx-helper-activation-window",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto zero = fixed_zero(value);
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(activation(ActivationOp::Relu, value))),
                field<Field<1>>(
                    store(activation(ActivationOp::Relu, value, h))),
                field<Field<2>>(
                    store(activation(ActivationOp::LeakyRelu, value, q))),
                field<Field<3>>(
                    store(activation(ActivationOp::HardSigmoid, value))),
                field<Field<4>>(
                    store(activation(ActivationOp::HardSwish, value))),
                field<Field<5>>(
                    store(activation(ActivationOp::HardTanh, value))),
                field<Field<6>>(store(softsign(value))),
                field<Field<7>>(store(softsign(value, h))),
                field<Field<8>>(store(huber(value, q))),
                field<Field<9>>(store(smootherstep(zero, h, value))),
                field<Field<10>>(store(window(WindowOp::Parabolic, value))),
                field<Field<11>>(store(window(WindowOp::Triangular, value))),
                field<Field<12>>(store(window(WindowOp::Hann, value))),
                field<Field<13>>(store(window(WindowOp::Hamming, value))),
                field<Field<14>>(store(window(WindowOp::Blackman, value))),
                field<Field<15>>(store(window(WindowOp::Lanczos, value))));
          });
      result != 0) {
    return result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-approx-hash-noise",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto zero = fixed_zero(value);
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(
                    complex(ComplexOp::Conj, ComplexPart::Real, value, q))),
                field<Field<1>>(store(
                    complex(ComplexOp::Conj, ComplexPart::Imag, value, q))),
                field<Field<2>>(store(complex(ComplexOp::Abs2, value, q))),
                field<Field<3>>(store(complex(ComplexOp::Abs, value, q))),
                field<Field<4>>(store(complex(ComplexOp::Phase, value, q))),
                field<Field<5>>(store(complex(ComplexOp::Mul, ComplexPart::Real,
                                              value, q, h, q))),
                field<Field<6>>(store(complex(ComplexOp::Mul, ComplexPart::Imag,
                                              value, q, h, q))),
                field<Field<7>>(store(hash(value))),
                field<Field<8>>(store(hash(value, h))),
                field<Field<9>>(store(hash(HashOp::Unit, value))),
                field<Field<10>>(store(hash(HashOp::Unit, value, h))),
                field<Field<11>>(store(noise(value, q))),
                field<Field<12>>(store(noise(value, q, h))),
                field<Field<13>>(store(noise(value, h, q, h))),
                field<Field<14>>(store(noise(value, h, q, h, zero))));
          });
      result != 0) {
    return 20 + result;
  }
  return 0;
}

} // namespace
} // namespace rund::node::test_contract::expression
