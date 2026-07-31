#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

struct KernelProgramDispatchContract {
  u32 backend_width = 0u;
  bool static_tile_map = false;
  bool global_claim_sync_elided = false;
  bool require_no_allocation = false;
  bool collect_worker_stats = false;
  bool require_dispatch_backend = true;
  bool no_allocation_verified = false;
};

} // namespace rund::kernel
