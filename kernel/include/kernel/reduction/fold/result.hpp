#pragma once

#include <kernel/reduction/fold/operation.hpp>

namespace rund::kernel {

struct FoldResult {
  bool ok = false;
  const char *reason = "not_run";
  u64 value = 0u;
  u32 slot_count = 0u;
  FoldOperation operation = FoldOperation::Xor;
  u64 fold_cost_ns = 0u;
  bool fold_cost_measured = false;
};

} // namespace rund::kernel
