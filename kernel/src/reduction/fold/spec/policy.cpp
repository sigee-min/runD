#include "local.hpp"

namespace rund::kernel {
namespace {

bool IsFloatOperation(const FoldOperation operation) {
  return operation == FoldOperation::StrictFloat32Add ||
         operation == FoldOperation::StrictFloat64Add;
}

} // namespace

bool IsSupportedFoldOperation(const FoldOperation operation) {
  const FoldPrimitiveSpec spec = DescribeFoldOperation(operation);
  return spec.supported && spec.value_domain != FoldValueDomain::Unsupported;
}

bool FoldOperationAllowsFloatingPoint(const FoldOperation operation) {
  return DescribeFoldOperation(operation).admits_floating_point;
}

bool StrictFloatReductionValid(const FoldOperation operation, const StrictFloatReductionPolicy policy) {
  if (!IsFloatOperation(operation)) {
    return !policy.valid;
  }
  const FoldPrimitiveSpec primitive = DescribeFoldOperation(operation);
  return policy.valid &&
         policy.value_domain == primitive.value_domain &&
         policy.rounding_mode == StrictFloatRoundingMode::NearestTiesToEven &&
         policy.nan_policy == StrictFloatNanPolicy::CanonicalQuiet &&
         (policy.signed_zero_policy == StrictFloatSignedZeroPolicy::CanonicalPositive ||
          policy.signed_zero_policy == StrictFloatSignedZeroPolicy::PreserveOnlyWhenBothNegative) &&
         policy.infinity_policy == StrictFloatInfinityPolicy::Preserve &&
         policy.subnormal_policy == StrictFloatSubnormalPolicy::Preserve &&
         policy.fma_policy == StrictFloatFmaPolicy::Forbidden;
}

} // namespace rund::kernel
