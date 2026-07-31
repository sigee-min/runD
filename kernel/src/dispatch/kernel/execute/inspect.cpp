#include "../local.hpp"

namespace rund::kernel {

WorkerBackendCapabilities InspectWorkerBackend(const WorkerBackend& backend, const u32 requested_width) {
  return dispatch::detail::InspectBackend(backend, requested_width);
}

} // namespace rund::kernel
