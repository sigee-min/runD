#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class FoldOperation : u8 {
  Xor = 0,
  Max = 1,
  Min = 2,
  SaturatingAdd = 3,
  FixedBinaryTreeHash = 4,
  StrictFloat32Add = 5,
  StrictFloat64Add = 6,
};

enum class FoldValueDomain : u8 {
  BitwiseInteger = 0,
  UnsignedInteger = 1,
  HashDigest = 2,
  Float32Strict = 3,
  Float64Strict = 4,
  Unsupported = 255,
};

enum class FoldPaddingLaw : u8 {
  None = 0,
  Identity = 1,
  FixedHashOdd = 2,
  UnsignedMax = 3,
};

enum class FoldOverflowLaw : u8 {
  None = 0,
  BitwiseInteger = 1,
  UnsignedCompare = 2,
  SaturatingUnsignedAdd = 3,
  FixedHashMix = 4,
  StrictIeee754Add = 5,
};

} // namespace rund::kernel
