#include "local.hpp"

namespace rund::kernel::reduction::fold {

FoldGraphValidationResult FailFoldGraphValidation(const char* const reason,
                                                  const u32 scratch_slot_count) {
  return FoldGraphValidationResult{
      .ok = false,
      .reason = reason,
      .scratch_slot_count = scratch_slot_count,
  };
}

} // namespace rund::kernel::reduction::fold
