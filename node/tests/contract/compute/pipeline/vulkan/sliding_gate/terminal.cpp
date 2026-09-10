#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL)

#include "src/accel/kernel/fault.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/vulkan/adapter/state.hpp"
#include "src/accel/vulkan/kernel.hpp"
#include "src/accel/vulkan/kernel/pipeline/residency/local.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute/flow.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

namespace rund_node_test_pipeline::sliding_gate_detail {
namespace {

template <class Predicate>
[[nodiscard]] bool WaitForDiagnostic(Predicate &&ready) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (!ready()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

} // namespace

bool RunTerminalFrontier() {
  using namespace rund::compute;
  using namespace rund::node::accel::detail;
  constexpr std::size_t elements = 8u;
  auto opened = open(Target::vulkan());
  if (!opened) {
    return false;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .map<std::uint32_t>("vulkan-sliding-frontier", 1u,
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
    return false;
  }
  const auto state = prepared.value();
  const auto primary = state == nullptr ? nullptr : state->pipeline;
  const auto alternate = state == nullptr ? nullptr : state->alternate_pipeline;
  auto *const primary_common =
      primary == nullptr || !primary->prepared.ok
          ? nullptr
          : static_cast<rund::node::accel::detail::prepared::PipelineState *>(
                primary->prepared.owner.get());
  auto *const alternate_common =
      alternate == nullptr || !alternate->prepared.ok
          ? nullptr
          : static_cast<rund::node::accel::detail::prepared::PipelineState *>(
                alternate->prepared.owner.get());
  const std::shared_ptr<void> primary_backend = primary_common == nullptr
                                                    ? std::shared_ptr<void>{}
                                                    : primary_common->backend;
  const std::shared_ptr<void> peer_backend = alternate_common == nullptr
                                                 ? std::shared_ptr<void>{}
                                                 : alternate_common->backend;
  auto *const primary_pipeline =
      static_cast<VulkanPipeline *>(primary_backend.get());
  auto *const peer_pipeline = static_cast<VulkanPipeline *>(peer_backend.get());
  const auto *const native = primary == nullptr || primary->device == nullptr
                                 ? nullptr
                                 : detail::accel_device(*primary->device);
  VulkanAdapter *const adapter =
      primary_pipeline == nullptr ? nullptr : primary_pipeline->adapter;
  if (primary_backend == nullptr || peer_backend == nullptr ||
      primary_backend == peer_backend || adapter == nullptr ||
      native == nullptr || peer_pipeline == nullptr ||
      peer_pipeline->adapter != adapter) {
    return false;
  }

  VulkanResidencySlidingDiagnostics primary_before{};
  VulkanResidencySlidingDiagnostics peer_before{};
  constexpr std::uint32_t primary_control = 900'001u;
  constexpr std::uint32_t peer_control = 900'101u;
  if (!InspectVulkanResidencySliding(primary_backend, primary_before) ||
      !InspectVulkanResidencySliding(peer_backend, peer_before) ||
      !primary_before.terminal_frontier_released ||
      primary_before.gate_quarantined || primary_before.adapter_quarantined ||
      !SeedPreparedVulkanPipelineGeneration(primary_backend,
                                            primary_control - 1u)
           .ok ||
      !SeedPreparedVulkanPipelineGeneration(peer_backend, peer_control - 1u)
           .ok ||
      !InjectVulkanResidencySlidingTerminalFrontierPauseOnce(primary_backend) ||
      !InjectNativeDeviceLostOnce(native->pick)) {
    return false;
  }
  std::uint64_t queue_before = 0u;
  {
    std::lock_guard lock{adapter->mutex};
    queue_before = adapter->command_submit_count;
  }

  const std::array<std::uint32_t, 1u> locals{0u};
  const BackendResidencySlidingDescriptor primary_descriptor{
      .owner = primary_backend.get(),
      .plan_identity = 0x56'4b'4650u,
      .token = 910'001u,
      .generation = 910'002u,
      .read_mask = 1u,
      .write_mask = 1u,
      .descriptor_generation = 1u,
      .control_generation = primary_control,
      .stride = 1u,
  };
  Wait primary_wait{};
  if (!SubmitVulkanResidencySliding(
           primary_backend, primary_descriptor, Complete, &primary_wait,
           KernelTiming::Submission, PipelineSubmitMode::Residency, locals)
           .ok) {
    static_cast<void>(
        ReleaseVulkanResidencySlidingTerminalFrontier(primary_backend));
    return false;
  }
  const bool terminal_entered = WaitForDiagnostic([&] {
    VulkanResidencySlidingDiagnostics observed{};
    return InspectVulkanResidencySliding(primary_backend, observed) &&
           observed.terminal_frontier_count ==
               primary_before.terminal_frontier_count + 1u;
  });
  if (!terminal_entered) {
    static_cast<void>(
        ReleaseVulkanResidencySlidingTerminalFrontier(primary_backend));
    primary_wait.done.wait(false, std::memory_order_acquire);
    return false;
  }

  const BackendResidencySlidingDescriptor peer_descriptor{
      .owner = peer_backend.get(),
      .plan_identity = 0x56'4b'4651u,
      .token = 920'001u,
      .generation = 920'002u,
      .read_mask = 1u,
      .write_mask = 1u,
      .descriptor_generation = 1u,
      .control_generation = peer_control,
      .stride = 1u,
  };
  Wait peer_wait{};
  rund::AccelCheck peer_submit{};
  std::thread peer{[&] {
    peer_submit = SubmitVulkanResidencySliding(
        peer_backend, peer_descriptor, Complete, &peer_wait,
        KernelTiming::Submission, PipelineSubmitMode::Residency, locals);
  }};
  const bool peer_entered = WaitForDiagnostic([&] {
    VulkanResidencySlidingDiagnostics observed{};
    return InspectVulkanResidencySliding(peer_backend, observed) &&
           observed.submit_frontier_count ==
               peer_before.submit_frontier_count + 1u;
  });
  const bool released =
      ReleaseVulkanResidencySlidingTerminalFrontier(primary_backend);
  peer.join();
  primary_wait.done.wait(false, std::memory_order_acquire);

  VulkanResidencySlidingDiagnostics primary_after{};
  VulkanResidencySlidingDiagnostics peer_after{};
  std::uint64_t queue_after = 0u;
  {
    std::lock_guard lock{adapter->mutex};
    queue_after = adapter->command_submit_count;
  }
  const bool inspected =
      InspectVulkanResidencySliding(primary_backend, primary_after) &&
      InspectVulkanResidencySliding(peer_backend, peer_after);
  return peer_entered && released && inspected &&
         !primary_wait.result.check.ok &&
         primary_wait.result.terminal == NativeTerminal::UnknownMayWrite &&
         primary_wait.callback_count.load(std::memory_order_relaxed) == 1u &&
         !peer_submit.ok && peer_submit.reason != nullptr &&
         std::strcmp(peer_submit.reason, "compute_device_lost") == 0 &&
         peer_wait.callback_count.load(std::memory_order_relaxed) == 0u &&
         primary_after.gpu_result_read_count ==
             primary_before.gpu_result_read_count &&
         primary_after.gate_quarantined && primary_after.adapter_quarantined &&
         peer_after.adapter_quarantined && queue_after == queue_before + 1u &&
         !VulkanResidencySlidingCapability(primary_backend,
                                           ResidencySlidingMemory::HostCoherent)
              .check.ok &&
         !VulkanResidencySlidingCapability(peer_backend,
                                           ResidencySlidingMemory::HostCoherent)
              .check.ok;
}

} // namespace rund_node_test_pipeline::sliding_gate_detail

#endif
