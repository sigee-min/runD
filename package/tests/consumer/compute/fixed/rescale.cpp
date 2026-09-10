#include "local.hpp"

#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace package_compute::fixed_contract {

int CheckRescale() {
  using RescaleSource = rund::compute::Fixed<63, 1>;
  using RescaleTarget = rund::compute::Fixed<1, 63>;
  constexpr std::int64_t scaled_one = std::int64_t{1} << 61u;
  const std::array<RescaleSource, 4u> rescale_input{
      RescaleSource::min(), RescaleSource::max(), RescaleSource::from_raw(1),
      RescaleSource::from_raw(-1)};
  auto saturated =
      rund::compute::on(rund::compute::Target::cpu(), rescale_input)
          .map("fixed-left-rescale-saturate",
               [](auto value) {
                 return rund::compute::quantize<
                     RescaleTarget, rund::compute::Rounding::NearestEven,
                     rund::compute::Overflow::Saturate,
                     rund::compute::Approximation::Exact>(value * value);
               })
          .collect();
  const std::vector<RescaleTarget> saturated_expected{
      RescaleTarget::max(), RescaleTarget::max(),
      RescaleTarget::from_raw(scaled_one), RescaleTarget::from_raw(scaled_one)};
  if (!saturated) {
    return saturated.exit_code();
  }
  if (*saturated != saturated_expected) {
    return 2;
  }
  auto wrapped =
      rund::compute::on(rund::compute::Target::cpu(), rescale_input)
          .map("fixed-left-rescale-wrap",
               [](auto value) {
                 return rund::compute::quantize<
                     RescaleTarget, rund::compute::Rounding::NearestEven,
                     rund::compute::Overflow::Wrap,
                     rund::compute::Approximation::Exact>(value * value);
               })
          .collect();
  const std::vector<RescaleTarget> wrapped_expected{
      RescaleTarget::zero(), RescaleTarget::from_raw(scaled_one),
      RescaleTarget::from_raw(scaled_one), RescaleTarget::from_raw(scaled_one)};
  if (!wrapped) {
    return wrapped.exit_code();
  }
  return *wrapped == wrapped_expected ? 0 : 2;
}

} // namespace package_compute::fixed_contract
