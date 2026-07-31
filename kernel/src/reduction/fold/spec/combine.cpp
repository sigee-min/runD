#include "local.hpp"

#include "../../primitives.hpp"
#include "../../strict/float.hpp"

#include <algorithm>

namespace rund::kernel::reduction::fold {

u64 CombineFoldValues(const FoldOperation operation,
                      const u64 left,
                      const u64 right,
                      const StrictFloatReductionPolicy policy) {
  switch (operation) {
    case FoldOperation::Xor:
      return left ^ right;
    case FoldOperation::Max:
      return std::max(left, right);
    case FoldOperation::Min:
      return std::min(left, right);
    case FoldOperation::SaturatingAdd:
      return reduction_primitives::SaturatingAdd(left, right);
    case FoldOperation::FixedBinaryTreeHash:
      return reduction_primitives::CombineHashes(left, right);
    case FoldOperation::StrictFloat32Add:
      return strict_float::AddFloat32Bits(left, right, policy);
    case FoldOperation::StrictFloat64Add:
      return strict_float::AddFloat64Bits(left, right, policy);
  }
  return 0u;
}

} // namespace rund::kernel::reduction::fold
