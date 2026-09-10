#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include <cstdio>

namespace package_compute::fixed_contract {

int CheckRejections() {
  using rund::compute::Fixed;
  auto implicit_storage =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<16, 16>>("implicit-fixed-storage", 1u,
                              [](auto value) { return value + value; })
          .compile();
  if (implicit_storage ||
      implicit_storage.code() != rund::compute::Code::Binding ||
      implicit_storage.error() != "compute_fixed_quantize_required") {
    std::fprintf(stderr,
                 "package implicit Fixed storage: success=%d reason=%.*s\n",
                 implicit_storage ? 1 : 0,
                 static_cast<int>(implicit_storage.error().size()),
                 implicit_storage.error().data());
    return 2;
  }
  auto approximation_downgrade =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<16, 16>>("fixed-approximation-downgrade", 1u,
                              [](auto value) {
                                return rund::compute::quantize<Fixed<16, 16>>(
                                    rund::compute::sqrt(value));
                              })
          .compile();
  if (approximation_downgrade ||
      approximation_downgrade.code() != rund::compute::Code::Binding ||
      approximation_downgrade.error() !=
          "compute_fixed_approximation_downgrade") {
    return 2;
  }
  auto fixed_policy_mismatch =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<16, 16>>(
              "fixed-policy-mismatch-rejected", 1u,
              [](auto value) {
                const auto down_wrap =
                    rund::compute::quantize<Fixed<16, 16>,
                                            rund::compute::Rounding::Down,
                                            rund::compute::Overflow::Wrap>(
                        value);
                const auto up_saturate =
                    rund::compute::quantize<Fixed<16, 16>,
                                            rund::compute::Rounding::Up,
                                            rund::compute::Overflow::Saturate>(
                        value);
                return rund::compute::quantize<Fixed<16, 16>>(down_wrap +
                                                              up_saturate);
              })
          .compile();
  if (fixed_policy_mismatch ||
      fixed_policy_mismatch.code() != rund::compute::Code::Binding ||
      fixed_policy_mismatch.error() != "compute_fixed_format_mismatch") {
    return 2;
  }
  auto precision_capacity =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<32, 32>>("fixed-129-bit-add-rejected", 1u,
                              [](auto value) {
                                return rund::compute::quantize<Fixed<32, 32>>(
                                    value * value + value);
                              })
          .compile();
  if (precision_capacity ||
      precision_capacity.code() != rund::compute::Code::Capacity ||
      precision_capacity.error() != "compute_fixed_precision_capacity") {
    return 2;
  }
  auto select_precision_capacity =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<1, 63>>(
              "fixed-129-bit-select-rejected", 1u,
              [](auto value) {
                const auto high_fraction = value * value;
                const auto high_integer = (value + value) + value;
                return rund::compute::quantize<Fixed<1, 63>>(
                    rund::compute::select(value == value, high_fraction,
                                          high_integer));
              })
          .compile();
  if (select_precision_capacity ||
      select_precision_capacity.code() != rund::compute::Code::Capacity ||
      select_precision_capacity.error() != "compute_fixed_precision_capacity") {
    return 2;
  }
  auto clamp_precision_capacity =
      rund::compute::on(rund::compute::Target::cpu(1u))
          .map<Fixed<1, 63>>(
              "fixed-129-bit-clamp-rejected", 1u,
              [](auto value) {
                const auto high_fraction = value * value;
                const auto high_integer = (value + value) + value;
                return rund::compute::quantize<Fixed<1, 63>>(
                    rund::compute::clamp(high_integer, high_fraction,
                                         high_integer));
              })
          .compile();
  if (clamp_precision_capacity ||
      clamp_precision_capacity.code() != rund::compute::Code::Capacity ||
      clamp_precision_capacity.error() != "compute_fixed_precision_capacity") {
    return 2;
  }
  return 0;
}

} // namespace package_compute::fixed_contract
