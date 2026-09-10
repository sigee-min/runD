#include "../internal.hpp"

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck preflight(const PersistentResidencySlidingRequest &request,
                           MetalAdapter *&adapter) noexcept {
  const std::uint64_t issue = first_invalid_request(request, adapter);
  if (issue == valid_request_issue())
    return {true, "ok"};
  if (issue == request_issue_key(RequestIssue::GenerationRange))
    return {false, "compute_pipeline_capacity"};
  return {false, "accel_kernel_pipeline_invalid"};
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
