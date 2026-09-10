#include "internal.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_persistent {

bool RunPersistentCase(const std::uint64_t coordinate_count,
                       const std::uint64_t identity, const bool known_failure,
                       const bool unknown_failure,
                       const bool failed_admission_only,
                       const bool submit_device_lost) {
  using namespace rund::compute;
  const PersistentRunOptions options{
      .coordinate_count = coordinate_count,
      .identity = identity,
      .known_failure = known_failure,
      .unknown_failure = unknown_failure,
      .failed_admission_only = failed_admission_only,
      .submit_device_lost = submit_device_lost,
  };
  if ((known_failure && unknown_failure) ||
      (known_failure && failed_admission_only) ||
      (unknown_failure && failed_admission_only) ||
      (submit_device_lost &&
       (known_failure || unknown_failure || failed_admission_only))) {
    return false;
  }
  auto opened = open(Target::vulkan());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable;
  }
  Device device = std::move(opened).value();
  auto prepared = PreparePersistentCase(device, coordinate_count, identity);
  if (!prepared ||
      !ValidatePersistentBeforeSubmit(*prepared, submit_device_lost)) {
    return false;
  }
  return SubmitPersistentCase(*prepared, options);
}

} // namespace rund_node_test_pipeline_vulkan_persistent

#endif
