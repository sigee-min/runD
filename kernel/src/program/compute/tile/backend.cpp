#include "state.hpp"

namespace rund::kernel::compute_tile_detail {

Backend Select(const WorkerBackend backend, const u32 workers,
               const Mode mode) noexcept {
  if (!backend) {
    return Backend{.reason = "compute_tile_backend_invalid"};
  }
  const WorkerBackendCapabilities caps = InspectWorkerBackend(backend, workers);
  if (caps.is_nested) {
    return Backend{.reason = "pool_nested_dispatch"};
  }
  if (!caps.width_matches_request) {
    return Backend{.reason = "pool_width_mismatch"};
  }
  if (!caps.supports_static_tile_map ||
      !caps.supports_claim_free_static_tiles) {
    return Backend{.reason = "static_tile_backend_required"};
  }
  if (mode == Mode::Async && (!caps.supports_async_partitions ||
                              backend.submit_partitions == nullptr)) {
    return Backend{.reason = "async_tile_backend_required"};
  }
  if (mode == Mode::Sync && backend.execute_partitions == nullptr) {
    return Backend{.reason = "compute_tile_backend_invalid"};
  }
  return Backend{.value = backend, .ok = true, .reason = "pass"};
}

} // namespace rund::kernel::compute_tile_detail
