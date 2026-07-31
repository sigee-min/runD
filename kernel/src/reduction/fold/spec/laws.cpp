#include "local.hpp"

namespace rund::kernel::reduction::fold::spec_detail {

FoldPaddingLaw PaddingLaw(const FoldOperation operation) {
  switch (operation) {
    case FoldOperation::Xor:
    case FoldOperation::Max:
    case FoldOperation::SaturatingAdd:
    case FoldOperation::StrictFloat32Add:
    case FoldOperation::StrictFloat64Add:
      return FoldPaddingLaw::Identity;
    case FoldOperation::Min:
      return FoldPaddingLaw::UnsignedMax;
    case FoldOperation::FixedBinaryTreeHash:
      return FoldPaddingLaw::FixedHashOdd;
  }
  return FoldPaddingLaw::None;
}

FoldOverflowLaw OverflowLaw(const FoldOperation operation) {
  switch (operation) {
    case FoldOperation::Xor:
      return FoldOverflowLaw::BitwiseInteger;
    case FoldOperation::Max:
    case FoldOperation::Min:
      return FoldOverflowLaw::UnsignedCompare;
    case FoldOperation::SaturatingAdd:
      return FoldOverflowLaw::SaturatingUnsignedAdd;
    case FoldOperation::FixedBinaryTreeHash:
      return FoldOverflowLaw::FixedHashMix;
    case FoldOperation::StrictFloat32Add:
    case FoldOperation::StrictFloat64Add:
      return FoldOverflowLaw::StrictIeee754Add;
  }
  return FoldOverflowLaw::None;
}

} // namespace rund::kernel::reduction::fold::spec_detail
