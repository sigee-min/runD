#pragma once

#include "../../local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../device_vsm_product/local.hpp"
#include "../../persistent_product/local.hpp"
#include "../../residency/device_vsm/actual.hpp"
#include "../../residency/service_free_direct/product.hpp"

#include "src/accel/clock.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/kernel/residency/persistent_sliding/service_validation.hpp"
#include "src/accel/vulkan/adapter/state.hpp"
#include "src/accel/vulkan/kernel.hpp"
#include "src/accel/vulkan/kernel/pipeline/residency/generated_indirect/internal.hpp"
#include "src/accel/vulkan/kernel/pipeline/residency/local.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace rund_node_test_pipeline_vulkan_persistent {

namespace accel = rund::node::accel::detail;

[[nodiscard]] bool DeviceVsmQueueCount(const rund::AccelDevice &pick,
                                       std::uint64_t &count) noexcept;

class PersistentBacking final : public rund::compute::VirtualBacking {
public:
  explicit PersistentBacking(std::size_t bytes);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;

  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;

  [[nodiscard]] rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;

  [[nodiscard]] std::uint64_t write_count() const noexcept;

private:
  std::vector<std::byte> bytes_;
  std::atomic<std::uint64_t> write_count_;
};

struct FinalWait final {
  accel::PersistentResidencySlidingRequest request{};
  accel::PersistentResidencySlidingFinal final{};
  std::atomic<std::uint64_t> callback_count{};
};

void CompletePersistent(void *,
                        accel::PersistentResidencySlidingFinal &&) noexcept;

[[nodiscard]] std::shared_ptr<void> NativeBackend(
    const std::shared_ptr<rund::compute::detail::PipelineState> &pipeline);

[[nodiscard]] bool ProductQueueCount(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &state,
    std::uint64_t &count) noexcept;

[[nodiscard]] bool
PersistentRawControlCleared(const std::shared_ptr<void> &prepared) noexcept;

[[nodiscard]] bool PersistentControlPristine(
    const accel::PersistentResidencySlidingControl &control) noexcept;

[[nodiscard]] bool
CommitAccepted(const accel::PersistentResidencySlidingCapability &capability,
               const accel::PersistentResidencySlidingRequest &request,
               accel::PersistentResidencySlidingControl &control) noexcept;

[[nodiscard]] bool RejectOversizedCapability(
    accel::VulkanAdapter &adapter,
    const accel::PersistentResidencySlidingRequest &request) noexcept;

[[nodiscard]] bool SameFusedDirectStorage(
    const accel::VulkanFusedDirectRecurrenceDiagnostics &left,
    const accel::VulkanFusedDirectRecurrenceDiagnostics &right) noexcept;

[[nodiscard]] bool
SameCurrentCommonMemory(const accel::PreparedPipelineMemory &left,
                        const accel::PreparedPipelineMemory &right) noexcept;

[[nodiscard]] bool CheckGeneratedReadAtMap();
[[nodiscard]] bool CheckFusedDirectRecurrence();
[[nodiscard]] bool CheckFusedDirectHistory();

[[nodiscard]] bool RunPersistentCase(std::uint64_t coordinate_count,
                                     std::uint64_t identity, bool known_failure,
                                     bool unknown_failure,
                                     bool failed_admission_only,
                                     bool submit_device_lost);

} // namespace rund_node_test_pipeline_vulkan_persistent
#endif
