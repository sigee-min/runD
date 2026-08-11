#include "../local.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU) || defined(RUND_NODE_TEST_BACKEND_METAL)

namespace rund_node_test_pipeline {

int CheckVulkanTransferAdmission() { return 0; }

} // namespace rund_node_test_pipeline

#else

#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace rund_node_test_pipeline {
namespace {

class VulkanTransferBacking final : public rund::compute::VirtualBacking {
public:
  explicit VulkanTransferBacking(const std::size_t bytes) : bytes_(bytes) {}

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override {
    return bytes_.size();
  }

  [[nodiscard]] rund::compute::Status
  read(const std::uint64_t offset,
       const std::span<std::byte> output) noexcept override {
    ++callbacks_;
    if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
    return rund::compute::Status::success();
  }

  [[nodiscard]] rund::compute::Status
  write(const std::uint64_t offset,
        const std::span<const std::byte> input) noexcept override {
    ++callbacks_;
    if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
    return rund::compute::Status::success();
  }

  [[nodiscard]] std::uint64_t callbacks() const noexcept { return callbacks_; }

private:
  std::vector<std::byte> bytes_;
  std::uint64_t callbacks_{};
};

[[nodiscard]] bool SameCounter(const rund::compute::MemoryCounter &left,
                               const rund::compute::MemoryCounter &right) {
  return left.current == right.current && left.peak == right.peak &&
         left.cumulative == right.cumulative && left.reused == right.reused &&
         left.budget == right.budget;
}

[[nodiscard]] bool
SameDeviceMemorySnapshot(const rund::compute::MemoryStats &left,
                         const rund::compute::MemoryStats &right) {
  return left.backend == right.backend && left.scope == right.scope &&
         SameCounter(left.host, right.host) &&
         SameCounter(left.device, right.device) &&
         SameCounter(left.frame, right.frame) &&
         SameCounter(left.tile, right.tile) &&
         SameCounter(left.resident, right.resident) &&
         SameCounter(left.staging, right.staging) &&
         SameCounter(left.transfer, right.transfer);
}

struct VulkanTransferProbe final {
  std::uint64_t capacity{};
  std::uint64_t staging{};
};

[[nodiscard]] int ExactVulkanTransferProbe(VulkanTransferProbe &probe) {
  using namespace rund::compute;
  constexpr std::size_t PageElements = 16u;
  constexpr std::size_t Elements = 67u;
  constexpr std::size_t Bytes = Elements * sizeof(std::int32_t);
  auto opened = open(Target::vulkan());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program =
      on(device)
          .map<std::int32_t>("vulkan-transfer-admission", PageElements,
                             [](auto value) { return value + 1; })
          .compile();
  auto input_backing = std::make_shared<VulkanTransferBacking>(Bytes);
  auto output_backing = std::make_shared<VulkanTransferBacking>(Bytes);
  auto input = virtual_buffer<std::int32_t>(Elements, input_backing);
  auto output = virtual_buffer<std::int32_t>(Elements, output_backing);
  auto prepared =
      program && input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    // The portability subset is a product capability boundary. Its exact
    // zero-mutation rejection is owned by the public product capability leaf;
    // this lower-level admission proof remains executable on native Vulkan.
    return prepared.reason() == Reason::BackendUnsupported ? 0 : 2;
  }
  const PipelinePlan plan = prepared->plan();
  const MemoryStats memory = prepared->memory();
  const DevicePipelineMemoryReport report = device.pipeline_memory();
  probe.capacity = plan.committed_peak_bytes;
  probe.staging = memory.staging.current;
  if (probe.capacity == 0u || probe.staging == 0u ||
      memory.staging.peak != probe.staging ||
      memory.staging.cumulative != probe.staging ||
      probe.staging > plan.prepared_native_bytes ||
      report.committed_bytes != probe.capacity ||
      report.preparing_bytes != 0u || input_backing->callbacks() != 0u ||
      output_backing->callbacks() != 0u) {
    return 3;
  }
  return 0;
}

[[nodiscard]] int
VulkanTransferBudgetRollback(const VulkanTransferProbe probe) {
  using namespace rund::compute;
  if (probe.capacity == 0u) {
    return 0;
  }
  constexpr std::size_t PageElements = 16u;
  constexpr std::size_t Elements = 67u;
  constexpr std::size_t Bytes = Elements * sizeof(std::int32_t);
  auto opened = open(Target::vulkan(),
                     DevicePipelineMemoryLimit{.bytes = probe.capacity - 1u});
  if (!opened) {
    return 1;
  }
  Device device = std::move(opened).value();
  auto program =
      on(device)
          .map<std::int32_t>("vulkan-transfer-admission", PageElements,
                             [](auto value) { return value + 1; })
          .compile();
  auto input_backing = std::make_shared<VulkanTransferBacking>(Bytes);
  auto output_backing = std::make_shared<VulkanTransferBacking>(Bytes);
  auto input = virtual_buffer<std::int32_t>(Elements, input_backing);
  auto output = virtual_buffer<std::int32_t>(Elements, output_backing);
  if (!program || !input || !output) {
    return 2;
  }
  const MemoryStats before = device.memory();
  auto rejected =
      virtual_pipeline(*program, *input, *output, ResidencyConfig{});
  const MemoryStats after = device.memory();
  const DevicePipelineMemoryReport report = device.pipeline_memory();
  if (rejected || rejected.reason() != Reason::DevicePipelineMemoryCapacity ||
      !SameDeviceMemorySnapshot(before, after) ||
      report.committed_bytes != 0u || report.preparing_bytes != 0u ||
      input_backing->callbacks() != 0u || output_backing->callbacks() != 0u) {
    return 3;
  }
  return 0;
}

} // namespace

int CheckVulkanTransferAdmission() {
  VulkanTransferProbe probe{};
  const int exact = ExactVulkanTransferProbe(probe);
  if (exact != 0) {
    return exact;
  }
  const int rollback = VulkanTransferBudgetRollback(probe);
  return rollback == 0 ? 0 : 100 + rollback;
}

} // namespace rund_node_test_pipeline

#endif
