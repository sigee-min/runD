#include "internal.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_persistent {

bool SubmitPersistentCase(const PreparedCase &test_case,
                          const PersistentRunOptions &options) {
  if (test_case.lowering.backend.service.submit_result == nullptr) {
    std::fprintf(stderr, "Vulkan persistent: submit result unavailable\n");
    return false;
  }
  accel::PersistentResidencySlidingControl control{};
  const std::uint64_t submit_probe_ns = accel::MonotonicNanoseconds();
  const accel::PersistentResidencySlidingSubmitResult submit_result =
      test_case.lowering.backend.service.submit_result(test_case.wait.request,
                                                       control);
  const rund::AccelCheck &submitted = submit_result.status;
  if (options.submit_device_lost) {
    return HandlePersistentDeviceLoss(test_case, submit_result, submitted,
                                      control);
  }
  if (!submitted.ok ||
      submit_result.event !=
          accel::PersistentResidencySlidingSubmitEvent::Accepted) {
    std::fprintf(stderr, "Vulkan persistent: submit reason=%s event=%u\n",
                 submitted.reason, static_cast<unsigned>(submit_result.event));
    return false;
  }
  if (!CommitAccepted(test_case.lowering.backend.capability,
                      test_case.wait.request, control)) {
    std::fprintf(stderr, "Vulkan persistent: accepted prefix commit\n");
    return false;
  }
  if (options.unknown_failure) {
    return HandlePersistentUnknown(test_case, control, submit_probe_ns);
  }
  return HandlePersistentKnown(test_case, options, control, submit_probe_ns);
}

} // namespace rund_node_test_pipeline_vulkan_persistent

#endif
