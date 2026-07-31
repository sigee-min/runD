#pragma once

#include <kernel/reduction/fold/operation.hpp>

namespace rund::kernel {

enum class StrictFloatRoundingMode : u8 {
  NearestTiesToEven = 0,
};

enum class StrictFloatNanPolicy : u8 {
  CanonicalQuiet = 0,
};

enum class StrictFloatSignedZeroPolicy : u8 {
  CanonicalPositive = 0,
  PreserveOnlyWhenBothNegative = 1,
};

enum class StrictFloatInfinityPolicy : u8 {
  Preserve = 0,
};

enum class StrictFloatSubnormalPolicy : u8 {
  Preserve = 0,
};

enum class StrictFloatFmaPolicy : u8 {
  Forbidden = 0,
};

struct StrictFloatReductionPolicy {
  bool valid = false;
  FoldValueDomain value_domain = FoldValueDomain::Unsupported;
  StrictFloatRoundingMode rounding_mode =
      StrictFloatRoundingMode::NearestTiesToEven;
  StrictFloatNanPolicy nan_policy = StrictFloatNanPolicy::CanonicalQuiet;
  StrictFloatSignedZeroPolicy signed_zero_policy =
      StrictFloatSignedZeroPolicy::CanonicalPositive;
  StrictFloatInfinityPolicy infinity_policy =
      StrictFloatInfinityPolicy::Preserve;
  StrictFloatSubnormalPolicy subnormal_policy =
      StrictFloatSubnormalPolicy::Preserve;
  StrictFloatFmaPolicy fma_policy = StrictFloatFmaPolicy::Forbidden;
};

StrictFloatReductionPolicy StrictFloat32ReductionPolicy();
StrictFloatReductionPolicy StrictFloat64ReductionPolicy();

} // namespace rund::kernel
