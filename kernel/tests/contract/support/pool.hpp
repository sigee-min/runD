#pragma once

#include <kernel/dispatch/orchestrator.hpp>

#include <array>

namespace kernel_contract_test {

struct FakePool {
  rund::kernel::u32 workers = 1u;
  bool supports_no_alloc_stats = true;
  bool supports_strict_fp_fold = false;
  bool nested = false;
  rund::kernel::WorkerTruthLevel affinity_truth_level = rund::kernel::WorkerTruthLevel::HintOnly;
};

inline FakePool BuildStaticPool(const rund::kernel::u32 workers) {
  return FakePool{
      .workers = workers,
  };
}

} // namespace kernel_contract_test
