#include "rolling.hpp"

#if !defined(RUND_NODE_HAVE_VULKAN_SDK) ||                                     \
    defined(RUND_NODE_TEST_BACKEND_CPU) ||                                     \
    defined(RUND_NODE_TEST_BACKEND_METAL)

#else

#include "src/accel/backend/resource.hpp"
#include "src/accel/backend/token.hpp"
#include "src/accel/backend/usage.hpp"
#include "src/accel/kernel/fault.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/vulkan/adapter/state.hpp"
#include "src/accel/vulkan/buffer/resident/find.hpp"
#include "src/accel/vulkan/kernel.hpp"
#include "src/accel/vulkan/kernel/pipeline/residency/local.hpp"
#include "src/accel/vulkan/kernel/pipeline/state.hpp"
#include "src/accel/vulkan/resident/access.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline {

[[nodiscard]] bool ProductLegacyPersistentFallback() {
  using namespace rund::compute;
  constexpr std::size_t epochs = 2u;
  constexpr std::size_t elements = epochs * 2u;
  auto opened = open(Target::vulkan());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .map<std::uint32_t>("vulkan-product-rolling", 1u,
                                         [](auto value) { return value + 1u; })
                     .compile();
  auto input_backing =
      std::make_shared<WindowBacking>(elements * sizeof(std::uint32_t));
  auto output_backing =
      std::make_shared<WindowBacking>(elements * sizeof(std::uint32_t));
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, input_backing);
  auto output = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, output_backing);
  auto prepared =
      program && input && output
          ? detail::prepare_virtual_pipeline(
                detail::ProgramAccess::state(*program),
                std::move(input).value(), std::move(output).value(),
                ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    return prepared.reason() == Reason::BackendUnsupported;
  }
  const auto state = prepared.value();
  detail::AccelDeviceState *const native =
      state == nullptr || state->pipeline == nullptr ||
              state->pipeline->device == nullptr
          ? nullptr
          : detail::accel_device(*state->pipeline->device);
  if (native == nullptr ||
      !rund::node::accel::detail::InjectNativeHostWriteUnavailableOnce(
          native->pick)) {
    return false;
  }
  const std::uint64_t before = ReadVulkanResidencyQueueSubmits(state->pipeline);
  const Status status = detail::run_virtual_pipeline(state);
  const std::uint64_t after = ReadVulkanResidencyQueueSubmits(state->pipeline);
  const Stats stats = detail::virtual_pipeline_stats(state);
  const bool valid = status && stats.pipeline.residency.epoch_count == epochs &&
                     stats.pipeline.residency.window_handoff_count == 0u &&
                     stats.pipeline.residency.window_batch_count == 0u &&
                     stats.pipeline.residency.window_queue_call_count == 0u &&
                     after > before + 1u && output_backing->all_u32(1u);
  if (!valid) {
    std::fprintf(
        stderr,
        "vulkan rolling status=%u epochs=%llu window=%llu/%llu/%llu "
        "submits=%llu/%llu output=%u\n",
        static_cast<unsigned>(status.reason()),
        static_cast<unsigned long long>(stats.pipeline.residency.epoch_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_handoff_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_batch_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_queue_call_count),
        static_cast<unsigned long long>(before),
        static_cast<unsigned long long>(after),
        static_cast<unsigned>(output_backing->all_u32(1u)));
  }
  return valid;
}

} // namespace rund_node_test_pipeline

#endif
