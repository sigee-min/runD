#pragma once

#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <cstdint>
#include <memory>

namespace rund_node_test_pipeline_vulkan_persistent {

struct PersistentRunOptions final {
  std::uint64_t coordinate_count{};
  std::uint64_t identity{};
  bool known_failure{};
  bool unknown_failure{};
  bool failed_admission_only{};
  bool submit_device_lost{};
};

struct PreparedCase final {
  std::shared_ptr<void> primary;
  std::shared_ptr<void> alternate;
  accel::VulkanPipeline *native{};
  std::shared_ptr<PersistentBacking> output_backing;
  FinalWait wait{};
  accel::PreparedResidencyPersistentSlidingPreparation lowering{};
  accel::VulkanResidencySlidingDiagnostics primary_before{};
  std::uint64_t queue_before{};
};

[[nodiscard]] std::unique_ptr<PreparedCase>
PreparePersistentCase(rund::compute::Device &, std::uint64_t coordinate_count,
                      std::uint64_t identity);

[[nodiscard]] bool
ValidatePersistentBeforeSubmit(PreparedCase &,
                               bool submit_device_lost) noexcept;

[[nodiscard]] bool SubmitPersistentCase(const PreparedCase &,
                                        const PersistentRunOptions &);

[[nodiscard]] bool HandlePersistentDeviceLoss(
    const PreparedCase &, const accel::PersistentResidencySlidingSubmitResult &,
    const rund::AccelCheck &,
    accel::PersistentResidencySlidingControl &) noexcept;
[[nodiscard]] bool
HandlePersistentUnknown(const PreparedCase &,
                        accel::PersistentResidencySlidingControl &,
                        std::uint64_t submit_probe_ns) noexcept;
[[nodiscard]] bool
HandlePersistentKnown(const PreparedCase &, const PersistentRunOptions &,
                      accel::PersistentResidencySlidingControl &,
                      std::uint64_t submit_probe_ns);

} // namespace rund_node_test_pipeline_vulkan_persistent

#endif
