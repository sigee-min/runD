#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

constexpr u32 kStackPartitionCapacity = 256u;

enum class PartitionIntent : u8 {
  StaticWidth = 0,
};

enum class AllocationPolicy : u8 {
  AllowGrowth = 0,
  NoGrowth = 1,
};

enum class PlacementPolicy : u8 {
  Uniform = 0,
  ContiguousBalanced = 1,
  WeightedStable = 2,
  CapacityWeightedStatic = 3,
};

} // namespace rund::kernel
