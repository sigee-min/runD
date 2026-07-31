#include "local.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <algorithm>

namespace rund::kernel {

bool EnsureFoldGraphScratch(FoldSlots& scratch,
                            const FoldGraphView graph,
                            const AllocationPolicy allocation) {
  const u32 required = FoldGraphScratchSlotCount(graph);
  if (allocation == AllocationPolicy::NoGrowth &&
      scratch.values.capacity() < static_cast<std::size_t>(required)) {
    return false;
  }
  try {
    if (allocation == AllocationPolicy::AllowGrowth) {
      scratch.values.reserve(required);
    }
    scratch.values.resize(required);
  } catch (...) {
    return false;
  }
  std::fill(scratch.values.begin(), scratch.values.end(), 0u);
  return true;
}

} // namespace rund::kernel
