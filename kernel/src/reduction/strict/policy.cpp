#include "float.hpp"

namespace rund::kernel {

StrictFloatReductionPolicy StrictFloat32ReductionPolicy() {
  return StrictFloatReductionPolicy{
      .valid = true,
      .value_domain = FoldValueDomain::Float32Strict,
      .rounding_mode = StrictFloatRoundingMode::NearestTiesToEven,
      .nan_policy = StrictFloatNanPolicy::CanonicalQuiet,
      .signed_zero_policy = StrictFloatSignedZeroPolicy::CanonicalPositive,
      .infinity_policy = StrictFloatInfinityPolicy::Preserve,
      .subnormal_policy = StrictFloatSubnormalPolicy::Preserve,
      .fma_policy = StrictFloatFmaPolicy::Forbidden,
  };
}

StrictFloatReductionPolicy StrictFloat64ReductionPolicy() {
  return StrictFloatReductionPolicy{
      .valid = true,
      .value_domain = FoldValueDomain::Float64Strict,
      .rounding_mode = StrictFloatRoundingMode::NearestTiesToEven,
      .nan_policy = StrictFloatNanPolicy::CanonicalQuiet,
      .signed_zero_policy = StrictFloatSignedZeroPolicy::CanonicalPositive,
      .infinity_policy = StrictFloatInfinityPolicy::Preserve,
      .subnormal_policy = StrictFloatSubnormalPolicy::Preserve,
      .fma_policy = StrictFloatFmaPolicy::Forbidden,
  };
}

} // namespace rund::kernel
