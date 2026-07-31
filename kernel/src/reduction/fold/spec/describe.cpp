#include "local.hpp"

namespace rund::kernel {
namespace {

FoldPrimitiveSpec IntegerSpec(const FoldOperation operation,
                              const FoldValueDomain value_domain) {
  return FoldPrimitiveSpec{
      .operation = operation,
      .value_domain = value_domain,
      .supported = true,
      .fixed_topology = true,
      .identity_value = reduction::fold::FoldIdentityValue(operation),
      .padding_value = reduction::fold::FoldPaddingValue(operation),
      .padding_law = reduction::fold::spec_detail::PaddingLaw(operation),
      .overflow_law = reduction::fold::spec_detail::OverflowLaw(operation),
  };
}

FoldPrimitiveSpec StrictFloatSpec(const FoldOperation operation,
                                  const FoldValueDomain value_domain,
                                  const StrictFloatReductionPolicy policy) {
  FoldPrimitiveSpec spec = IntegerSpec(operation, value_domain);
  spec.admits_floating_point = true;
  spec.requires_strict_floating_point = true;
  spec.strict_float_reduction = policy;
  return spec;
}

} // namespace

FoldPrimitiveSpec DescribeFoldOperation(const FoldOperation operation) {
  switch (operation) {
    case FoldOperation::Xor:
      return IntegerSpec(operation, FoldValueDomain::BitwiseInteger);
    case FoldOperation::Max:
    case FoldOperation::Min:
    case FoldOperation::SaturatingAdd:
      return IntegerSpec(operation, FoldValueDomain::UnsignedInteger);
    case FoldOperation::FixedBinaryTreeHash:
      return IntegerSpec(operation, FoldValueDomain::HashDigest);
    case FoldOperation::StrictFloat32Add:
      return StrictFloatSpec(operation,
                             FoldValueDomain::Float32Strict,
                             StrictFloat32ReductionPolicy());
    case FoldOperation::StrictFloat64Add:
      return StrictFloatSpec(operation,
                             FoldValueDomain::Float64Strict,
                             StrictFloat64ReductionPolicy());
  }
  return FoldPrimitiveSpec{
      .operation = operation,
      .value_domain = FoldValueDomain::Unsupported,
      .supported = false,
      .fixed_topology = false,
      .admits_floating_point = false,
      .requires_strict_floating_point = false,
  };
}

} // namespace rund::kernel
