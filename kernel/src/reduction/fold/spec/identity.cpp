#include "local.hpp"

#include "../../primitives.hpp"

#include <limits>

namespace rund::kernel::reduction::fold {

u64 FoldIdentityValue(const FoldOperation operation) {
  switch (operation) {
    case FoldOperation::Xor:
    case FoldOperation::Max:
    case FoldOperation::SaturatingAdd:
    case FoldOperation::StrictFloat32Add:
    case FoldOperation::StrictFloat64Add:
      return 0u;
    case FoldOperation::Min:
      return std::numeric_limits<u64>::max();
    case FoldOperation::FixedBinaryTreeHash:
      return 0u;
  }
  return 0u;
}

u64 FoldPaddingValue(const FoldOperation operation) {
  switch (spec_detail::PaddingLaw(operation)) {
    case FoldPaddingLaw::Identity:
      return FoldIdentityValue(operation);
    case FoldPaddingLaw::UnsignedMax:
      return std::numeric_limits<u64>::max();
    case FoldPaddingLaw::FixedHashOdd:
      return reduction_primitives::kFixedBinaryTreeOddPadding;
    case FoldPaddingLaw::None:
      break;
  }
  return 0u;
}

} // namespace rund::kernel::reduction::fold
