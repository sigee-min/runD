#pragma once

#include "../../local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_METAL) && \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "src/accel/kernel/fault.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/vulkan/kernel/pipeline/state.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/device/residency/execution/owner.hpp"
#include "src/compute/device/residency/execution/plan.hpp"
#include "src/compute/device/residency/pool.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/pipeline/execution/submit.hpp"
#include "src/compute/pipeline/local.hpp"
#include "src/compute/pipeline/residency/authority.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund_node_test_pipeline::vulkan_transfer {

namespace execution = rund::compute::detail::residency::execution;
namespace residency = rund::compute::detail::residency;

class VulkanTransferBacking final : public rund::compute::VirtualBacking {
public:
  explicit VulkanTransferBacking(std::size_t bytes);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;

  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;

  [[nodiscard]] rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;

  [[nodiscard]] std::uint64_t callbacks() const noexcept;

private:
  std::vector<std::byte> bytes_;
  std::uint64_t callbacks_{};
};

struct VulkanSelectionWait final {
  std::atomic_bool done{false};
  rund::node::accel::detail::PreparedPipelineEvidence evidence{};
};

struct VulkanExecutionWait final {
  std::atomic_bool done{false};
  execution::NativeEvidence evidence{};
};

void CompleteVulkanExecution(void *raw,
                             execution::NativeEvidence &&evidence) noexcept;

void CompleteVulkanSelection(
    void *raw,
    rund::node::accel::detail::PreparedPipelineEvidence &&evidence) noexcept;

[[nodiscard]] execution::SealResult VulkanExecutionPlan(
    const residency::Pool &pool, std::uint64_t page_count) noexcept;

[[nodiscard]] bool RunVulkanSelection(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    std::span<const std::uint32_t> locals,
    rund::node::accel::detail::PreparedPipelineEvidence &evidence);

[[nodiscard]] bool SeedVulkanSelection(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state);

[[nodiscard]] std::uint64_t VulkanQueueSubmits(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state);

[[nodiscard]] bool DownloadVulkanSelection(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    const rund::compute::Buffer<std::uint32_t> &buffer,
    std::array<std::uint32_t, 8u> &output);

[[nodiscard]] bool CheckVulkanRangeFailure(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    const std::shared_ptr<rund::compute::detail::BufferState> &buffer,
    const std::array<std::uint32_t, 8u> &expected);

[[nodiscard]] bool RejectVulkanSelection(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state,
    std::span<const std::uint32_t> locals);

[[nodiscard]] bool SameCurrentCounter(
    const rund::compute::MemoryCounter &left,
    const rund::compute::MemoryCounter &right);

[[nodiscard]] bool SameDeviceMemoryAuthority(
    const rund::compute::MemoryStats &left,
    const rund::compute::MemoryStats &right);

struct VulkanTransferProbe final {
  std::uint64_t capacity{};
  std::uint64_t staging{};
};

[[nodiscard]] int ExactVulkanResidencySelection();
[[nodiscard]] int ExactVulkanExecutionAdapter();
[[nodiscard]] int ExactVulkanTransferProbe(VulkanTransferProbe &probe);
[[nodiscard]] int VulkanTransferBudgetRollback(VulkanTransferProbe probe);

} // namespace rund_node_test_pipeline::vulkan_transfer

#endif
