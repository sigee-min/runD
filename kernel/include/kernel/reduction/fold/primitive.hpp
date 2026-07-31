#pragma once

#include <kernel/reduction/fold/operation.hpp>
#include <kernel/reduction/fold/strict.hpp>

namespace rund::kernel {

struct FoldPrimitiveSpec {
  FoldOperation operation = FoldOperation::FixedBinaryTreeHash;
  FoldValueDomain value_domain = FoldValueDomain::Unsupported;
  bool supported = false;
  bool fixed_topology = false;
  bool admits_floating_point = false;
  bool requires_strict_floating_point = false;
  u64 identity_value = 0u;
  u64 padding_value = 0u;
  FoldPaddingLaw padding_law = FoldPaddingLaw::None;
  FoldOverflowLaw overflow_law = FoldOverflowLaw::None;
  StrictFloatReductionPolicy strict_float_reduction{};
};

FoldPrimitiveSpec DescribeFoldOperation(FoldOperation operation);
bool IsSupportedFoldOperation(FoldOperation operation);
bool FoldOperationAllowsFloatingPoint(FoldOperation operation);
bool StrictFloatReductionValid(FoldOperation operation,
                               StrictFloatReductionPolicy policy);

} // namespace rund::kernel
