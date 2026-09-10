#include "../local.hpp"
#include "src/accel/vulkan/adapter/error.hpp"

#include <cstdio>

int RunComputePipelineVulkanSlidingGateContract() {
  const int result = rund_node_test_pipeline::CheckVulkanResidencySlidingGate();
  if (result != 0) {
    std::fprintf(stderr, "pipeline Vulkan sliding gate result=%d\n", result);
  }
  return result;
}

#if defined(RUND_NODE_TEST_BACKEND_CPU) || defined(RUND_NODE_TEST_BACKEND_METAL)

namespace rund_node_test_pipeline {

int CheckVulkanResidencySlidingGate() { return 0; }

} // namespace rund_node_test_pipeline

#else

#include "residency/local.hpp"
#include "sliding_gate/local.hpp"

#include "src/accel/backend/resource.hpp"
#include "src/accel/backend/token.hpp"
#include "src/accel/backend/usage.hpp"
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
#include <rund/compute/virtual.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline {
namespace sliding_gate_detail {

void Complete(void *const raw,
              rund::node::accel::detail::KernelResult result) noexcept {
  auto *const wait = static_cast<Wait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->result = std::move(result);
  wait->callback_count.fetch_add(1u, std::memory_order_relaxed);
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

} // namespace sliding_gate_detail

namespace {

using sliding_gate_detail::Complete;
using sliding_gate_detail::Wait;

[[nodiscard]] bool
RunSlidingGateCoordinates(const std::shared_ptr<void> &backend,
                          const std::size_t count, const std::uint64_t identity,
                          const std::uint8_t stride, const std::uint8_t slot) {
  using namespace rund::node::accel::detail;
  const BackendResidencySlidingCapability capability =
      VulkanResidencySlidingCapability(backend,
                                       ResidencySlidingMemory::HostCoherent);
  auto *const pipeline = static_cast<VulkanPipeline *>(backend.get());
  const VulkanResidencySelection *const selection =
      pipeline == nullptr ? nullptr : pipeline->residency.get();
  if (!capability.check.ok || capability.descriptor_release_acquire ||
      capability.memory != ResidencySlidingMemory::HostCoherent ||
      capability.retained_bytes == 0u || selection == nullptr ||
      capability.retained_bytes != selection->sliding.retained_bytes ||
      capability.transient_bytes !=
          sizeof(VulkanResidencySlidingSubmissionStorage)) {
    std::fprintf(
        stderr,
        "vulkan sliding capability ok=%u reason=%s release=%u "
        "retained=%llu transient=%llu gate=%u buffers=%llu/%llu/%llu "
        "command=%u semaphore=%u last=%s\n",
        static_cast<unsigned>(capability.check.ok), capability.check.reason,
        static_cast<unsigned>(capability.descriptor_release_acquire),
        static_cast<unsigned long long>(capability.retained_bytes),
        static_cast<unsigned long long>(capability.transient_bytes),
        static_cast<unsigned>(selection != nullptr &&
                              selection->sliding.ready_for_submit),
        static_cast<unsigned long long>(
            selection == nullptr
                ? 0u
                : selection->sliding.descriptor.allocated_bytes),
        static_cast<unsigned long long>(
            selection == nullptr
                ? 0u
                : selection->sliding.original_arguments.allocated_bytes),
        static_cast<unsigned long long>(
            selection == nullptr
                ? 0u
                : selection->sliding.argument_owners.allocated_bytes),
        static_cast<unsigned>(selection != nullptr &&
                              selection->sliding.command.buffer !=
                                  VK_NULL_HANDLE),
        static_cast<unsigned>(selection != nullptr &&
                              selection->sliding.ready != VK_NULL_HANDLE),
        pipeline == nullptr ? "no-pipeline"
                            : VulkanLastError(pipeline->adapter));
    return false;
  }
  std::uint64_t queue_begin = 0u;
  {
    std::lock_guard lock{pipeline->adapter->mutex};
    queue_begin = pipeline->adapter->command_submit_count;
  }
  const std::array<std::uint32_t, 1u> locals{0u};
  for (std::size_t turn = 0u; turn < count; ++turn) {
    const std::uint32_t control_generation =
        static_cast<std::uint32_t>(10'000u + identity * 512u + turn);
    if (!SeedPreparedVulkanPipelineGeneration(backend, control_generation - 1u)
             .ok) {
      std::fprintf(stderr, "vulkan sliding seed q=%zu turn=%zu\n", count, turn);
      return false;
    }
    Wait wait{};
    const BackendResidencySlidingDescriptor descriptor{
        .owner = backend.get(),
        .plan_identity = 0x56'4b'5347u + identity,
        .token = 20'000u + identity,
        .generation = 30'000u + identity,
        .coordinate = turn * stride + slot,
        .turn = turn,
        .read_mask = 1u,
        .write_mask = 1u,
        .descriptor_generation = turn + 1u,
        .control_generation = control_generation,
        .stride = stride,
        .slot = slot,
    };
    const rund::AccelCheck submitted = SubmitVulkanResidencySliding(
        backend, descriptor, Complete, &wait, KernelTiming::Submission,
        PipelineSubmitMode::Residency, locals);
    if (!submitted.ok) {
      std::fprintf(stderr, "vulkan sliding submit q=%zu turn=%zu reason=%s\n",
                   count, turn, submitted.reason);
      return false;
    }
    wait.done.wait(false, std::memory_order_acquire);
    if (!wait.result.check.ok ||
        wait.result.terminal != NativeTerminal::Known ||
        !wait.result.pipeline.submitted ||
        !wait.result.pipeline.control_observed ||
        wait.result.pipeline.control.generation != control_generation ||
        wait.result.stats.run.work.command_submit_count != 1u ||
        wait.result.stats.run.work.command_inflight_peak != 1u) {
      std::fprintf(
          stderr,
          "vulkan sliding terminal q=%zu turn=%zu check=%u reason=%s term=%u "
          "submitted=%u observed=%u generation=%u/%u queues=%llu peak=%llu\n",
          count, turn, static_cast<unsigned>(wait.result.check.ok),
          wait.result.check.reason, static_cast<unsigned>(wait.result.terminal),
          static_cast<unsigned>(wait.result.pipeline.submitted),
          static_cast<unsigned>(wait.result.pipeline.control_observed),
          wait.result.pipeline.control.generation, control_generation,
          static_cast<unsigned long long>(
              wait.result.stats.run.work.command_submit_count),
          static_cast<unsigned long long>(
              wait.result.stats.run.work.command_inflight_peak));
      return false;
    }
  }
  std::uint64_t queue_end = 0u;
  {
    std::lock_guard lock{pipeline->adapter->mutex};
    queue_end = pipeline->adapter->command_submit_count;
  }
  return queue_end >= queue_begin && queue_end - queue_begin == count;
}

[[nodiscard]] bool RunSlidingGateStale(const std::shared_ptr<void> &backend) {
  using namespace rund::node::accel::detail;
  constexpr std::uint32_t control_generation = 500'001u;
  const std::array<std::uint32_t, 1u> locals{0u};
  Wait first{};
  const BackendResidencySlidingDescriptor descriptor{
      .owner = backend.get(),
      .plan_identity = 0x56'4b'5354u,
      .token = 600'001u,
      .generation = 700'001u,
      .read_mask = 1u,
      .write_mask = 1u,
      .descriptor_generation = 1u,
      .control_generation = control_generation,
      .stride = 1u,
      .slot = 0u,
  };
  if (!SeedPreparedVulkanPipelineGeneration(backend, control_generation - 1u)
           .ok ||
      !SubmitVulkanResidencySliding(backend, descriptor, Complete, &first,
                                    KernelTiming::Submission,
                                    PipelineSubmitMode::Residency, locals)
           .ok) {
    return false;
  }
  first.done.wait(false, std::memory_order_acquire);
  if (!first.result.check.ok ||
      !InjectVulkanResidencySlidingStaleDescriptorOnce(backend) ||
      !SeedPreparedVulkanPipelineGeneration(backend, control_generation).ok) {
    return false;
  }
  Wait stale{};
  BackendResidencySlidingDescriptor next = descriptor;
  next.coordinate = 1u;
  next.turn = 1u;
  next.descriptor_generation = 2u;
  next.control_generation = control_generation + 1u;
  if (!SubmitVulkanResidencySliding(backend, next, Complete, &stale,
                                    KernelTiming::Submission,
                                    PipelineSubmitMode::Residency, locals)
           .ok) {
    return false;
  }
  stale.done.wait(false, std::memory_order_acquire);
  return !stale.result.check.ok &&
         stale.result.terminal == NativeTerminal::UnknownMayWrite &&
         stale.result.pipeline.submitted &&
         !stale.result.pipeline.control_observed &&
         stale.result.stats.run.work.command_submit_count == 1u &&
         stale.result.stats.run.work.command_inflight_peak == 1u &&
         !VulkanResidencySlidingCapability(backend,
                                           ResidencySlidingMemory::HostCoherent)
              .check.ok;
}

} // namespace

int CheckVulkanResidencySlidingGate() {
  using namespace rund::compute;
  constexpr std::size_t elements = 8u;
  auto opened = open(Target::vulkan());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .map<std::uint32_t>("vulkan-sliding-gate", 1u,
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
    return prepared.reason() == Reason::BackendUnsupported ? 0 : 2;
  }
  const auto state = prepared.value();
  const auto primary = state == nullptr ? nullptr : state->pipeline;
  auto *const common =
      primary == nullptr || !primary->prepared.ok
          ? nullptr
          : static_cast<rund::node::accel::detail::prepared::PipelineState *>(
                primary->prepared.owner.get());
  const std::shared_ptr<void> backend =
      common == nullptr ? std::shared_ptr<void>{} : common->backend;
  if (backend == nullptr) {
    return 3;
  }
  const auto *const native = primary->device == nullptr
                                 ? nullptr
                                 : detail::accel_device(*primary->device);
  auto *const adapter =
      native == nullptr
          ? nullptr
          : static_cast<rund::node::accel::detail::VulkanAdapter *>(
                native->pick.backend.context);
  // On a discrete target the physical sliding gate is deliberately outside
  // the admitted HostCoherent route. MoltenVK must execute every row below.
  if (adapter == nullptr) {
    return 3;
  }
  struct NativeAllocations final {
    std::uint64_t pipelines{};
    std::uint64_t descriptor_pools{};
    std::uint64_t descriptor_sets{};
    std::uint64_t buffers{};
  };
  const auto allocations = [&] {
    std::lock_guard lock{adapter->mutex};
    return NativeAllocations{
        .pipelines = adapter->pipeline_compile_count,
        .descriptor_pools = adapter->descriptor_pool_create_count,
        .descriptor_sets = adapter->descriptor_set_allocate_count,
        .buffers = adapter->buffer_allocation_count,
    };
  };
  const NativeAllocations cold = allocations();
  struct GateSample final {
    std::size_t count{};
    std::uint64_t identity{};
    std::uint8_t stride{};
    std::uint8_t slot{};
  };
  for (const GateSample sample :
       {GateSample{5u, 1u, 1u, 0u}, GateSample{9u, 2u, 2u, 1u},
        GateSample{257u, 3u, 3u, 2u}, GateSample{17u, 4u, 4u, 3u}}) {
    if (!RunSlidingGateCoordinates(backend, sample.count, sample.identity,
                                   sample.stride, sample.slot)) {
      return 4;
    }
  }
  if (!RunSlidingGateStale(backend)) {
    return 5;
  }
  const NativeAllocations warm = allocations();
  if (cold.pipelines != warm.pipelines ||
      cold.descriptor_pools != warm.descriptor_pools ||
      cold.descriptor_sets != warm.descriptor_sets ||
      cold.buffers != warm.buffers) {
    return 6;
  }
  return sliding_gate_detail::RunTerminalFrontier() ? 0 : 7;
}

} // namespace rund_node_test_pipeline

#endif
