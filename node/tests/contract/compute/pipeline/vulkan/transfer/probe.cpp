#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_METAL) && \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <cstdio>
#include <limits>

namespace rund_node_test_pipeline::vulkan_transfer {

int ExactVulkanTransferProbe(VulkanTransferProbe &probe) {
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
  const DevicePipelineMemoryReport before = device.pipeline_memory();
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
  if (report.committed_bytes < before.committed_bytes) {
    return 3;
  }
  const std::uint64_t admitted_delta =
      report.committed_bytes - before.committed_bytes;
  probe.capacity = plan.committed_peak_bytes;
  probe.staging = memory.staging.current;
  if (probe.capacity == 0u || probe.staging == 0u ||
      memory.staging.peak != probe.staging ||
      memory.staging.cumulative != probe.staging ||
      admitted_delta < probe.capacity ||
      report.preparing_bytes != before.preparing_bytes ||
      input_backing->callbacks() != 0u || output_backing->callbacks() != 0u) {
    std::fprintf(stderr,
                 "vulkan transfer probe capacity=%llu staging=%llu/%llu/%llu "
                 "native=%llu report=%llu/%llu callbacks=%llu/%llu\n",
                 static_cast<unsigned long long>(probe.capacity),
                 static_cast<unsigned long long>(probe.staging),
                 static_cast<unsigned long long>(memory.staging.peak),
                 static_cast<unsigned long long>(memory.staging.cumulative),
                 static_cast<unsigned long long>(plan.prepared_native_bytes),
                 static_cast<unsigned long long>(report.committed_bytes),
                 static_cast<unsigned long long>(report.preparing_bytes),
                 static_cast<unsigned long long>(input_backing->callbacks()),
                 static_cast<unsigned long long>(output_backing->callbacks()));
    return 3;
  }
  return 0;
}

int VulkanTransferBudgetRollback(const VulkanTransferProbe probe) {
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
      !SameDeviceMemoryAuthority(before, after) ||
      report.committed_bytes != 0u || report.preparing_bytes != 0u ||
      input_backing->callbacks() != 0u || output_backing->callbacks() != 0u) {
    std::fprintf(stderr,
                 "vulkan transfer rollback accepted=%u reason=%u limit=%llu "
                 "report=%llu/%llu callbacks=%llu/%llu "
                 "host=%llu/%llu/%llu:%llu/%llu/%llu "
                 "device=%llu/%llu/%llu:%llu/%llu/%llu "
                 "staging=%llu/%llu/%llu:%llu/%llu/%llu\n",
                 static_cast<unsigned>(static_cast<bool>(rejected)),
                 static_cast<unsigned>(rejected.reason()),
                 static_cast<unsigned long long>(probe.capacity - 1u),
                 static_cast<unsigned long long>(report.committed_bytes),
                 static_cast<unsigned long long>(report.preparing_bytes),
                 static_cast<unsigned long long>(input_backing->callbacks()),
                 static_cast<unsigned long long>(output_backing->callbacks()),
                 static_cast<unsigned long long>(before.host.current),
                 static_cast<unsigned long long>(before.host.peak),
                 static_cast<unsigned long long>(before.host.cumulative),
                 static_cast<unsigned long long>(after.host.current),
                 static_cast<unsigned long long>(after.host.peak),
                 static_cast<unsigned long long>(after.host.cumulative),
                 static_cast<unsigned long long>(before.device.current),
                 static_cast<unsigned long long>(before.device.peak),
                 static_cast<unsigned long long>(before.device.cumulative),
                 static_cast<unsigned long long>(after.device.current),
                 static_cast<unsigned long long>(after.device.peak),
                 static_cast<unsigned long long>(after.device.cumulative),
                 static_cast<unsigned long long>(before.staging.current),
                 static_cast<unsigned long long>(before.staging.peak),
                 static_cast<unsigned long long>(before.staging.cumulative),
                 static_cast<unsigned long long>(after.staging.current),
                 static_cast<unsigned long long>(after.staging.peak),
                 static_cast<unsigned long long>(after.staging.cumulative));
    return 3;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::vulkan_transfer

#endif
