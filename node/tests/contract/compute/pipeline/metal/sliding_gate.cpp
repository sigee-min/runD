#include "../local.hpp"

#include <cstdio>

int RunComputePipelineMetalSlidingGateContract() {
  const int result = rund_node_test_pipeline::CheckMetalResidencySlidingGate();
  if (result != 0) {
    std::fprintf(stderr, "pipeline Metal sliding gate result=%d\n", result);
  }
  return result;
}

#if defined(RUND_NODE_TEST_BACKEND_CPU) ||                                     \
    defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline {

int CheckMetalResidencySlidingGate() { return 0; }

} // namespace rund_node_test_pipeline

#else

#include "residency/local.hpp"

#include "src/accel/kernel/fault.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/metal/kernel.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace rund_node_test_pipeline {
namespace {

struct SlidingGateWait final {
  std::atomic_bool done{false};
  std::atomic<std::uint32_t> callback_count{};
  rund::node::accel::detail::KernelResult result{};
};

void CompleteSlidingGate(
    void *const raw, rund::node::accel::detail::KernelResult result) noexcept {
  auto *const wait = static_cast<SlidingGateWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->result = std::move(result);
  wait->callback_count.fetch_add(1u, std::memory_order_relaxed);
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

[[nodiscard]] bool
RunSlidingGateCoordinates(const std::shared_ptr<void> &backend,
                          const std::size_t count, const std::uint64_t identity,
                          const std::uint8_t stride, const std::uint8_t slot) {
  using namespace rund::node::accel::detail;
  const BackendResidencySlidingCapability capability =
      MetalResidencySlidingCapability(backend,
                                      ResidencySlidingMemory::HostCoherent);
  if (!capability.check.ok || capability.descriptor_release_acquire ||
      capability.memory != ResidencySlidingMemory::HostCoherent ||
      capability.retained_bytes == 0u || capability.transient_bytes != 480u) {
    return false;
  }
  const std::array<std::uint32_t, 1u> locals{0u};
  for (std::size_t turn = 0u; turn < count; ++turn) {
    const std::uint32_t control_generation =
        static_cast<std::uint32_t>(10'000u + identity * 512u + turn);
    if (!SeedPreparedMetalPipelineGeneration(backend, control_generation - 1u)
             .ok) {
      return false;
    }
    SlidingGateWait wait{};
    const BackendResidencySlidingDescriptor descriptor{
        .owner = backend.get(),
        .plan_identity = 0x4d'54'4c'47u + identity,
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
    const rund::AccelCheck submitted = SubmitMetalResidencySliding(
        backend, descriptor, CompleteSlidingGate, &wait,
        KernelTiming::Submission, PipelineSubmitMode::Residency, locals);
    if (!submitted.ok) {
      return false;
    }
    wait.done.wait(false, std::memory_order_acquire);
    if (!wait.result.check.ok ||
        wait.result.terminal != NativeTerminal::Known ||
        wait.callback_count.load(std::memory_order_relaxed) != 1u ||
        !wait.result.pipeline.submitted ||
        !wait.result.pipeline.control_observed ||
        wait.result.pipeline.control.generation != control_generation ||
        wait.result.stats.run.work.command_submit_count != 1u ||
        wait.result.stats.run.work.command_inflight_peak != 1u) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RunSlidingGateStale(const std::shared_ptr<void> &backend) {
  using namespace rund::node::accel::detail;
  constexpr std::uint32_t control_generation = 500'001u;
  const std::array<std::uint32_t, 1u> locals{0u};
  const BackendResidencySlidingDescriptor descriptor{
      .owner = backend.get(),
      .plan_identity = 0x4d'54'4c'53u,
      .token = 600'001u,
      .generation = 700'001u,
      .read_mask = 1u,
      .write_mask = 1u,
      .descriptor_generation = 1u,
      .control_generation = control_generation,
      .stride = 1u,
      .slot = 0u,
  };
  SlidingGateWait first{};
  if (!SeedPreparedMetalPipelineGeneration(backend, control_generation - 1u)
           .ok ||
      !SubmitMetalResidencySliding(backend, descriptor, CompleteSlidingGate,
                                   &first, KernelTiming::Submission,
                                   PipelineSubmitMode::Residency, locals)
           .ok) {
    return false;
  }
  first.done.wait(false, std::memory_order_acquire);
  if (!first.result.check.ok ||
      !InjectMetalResidencySlidingStaleDescriptorOnce(backend) ||
      !SeedPreparedMetalPipelineGeneration(backend, control_generation).ok) {
    return false;
  }
  SlidingGateWait stale{};
  BackendResidencySlidingDescriptor next = descriptor;
  next.coordinate = 1u;
  next.turn = 1u;
  next.descriptor_generation = 2u;
  next.control_generation = control_generation + 1u;
  if (!SubmitMetalResidencySliding(backend, next, CompleteSlidingGate, &stale,
                                   KernelTiming::Submission,
                                   PipelineSubmitMode::Residency, locals)
           .ok) {
    return false;
  }
  stale.done.wait(false, std::memory_order_acquire);
  return !stale.result.check.ok &&
         stale.result.terminal == NativeTerminal::UnknownMayWrite &&
         stale.callback_count.load(std::memory_order_relaxed) == 1u &&
         stale.result.pipeline.submitted &&
         !stale.result.pipeline.control_observed &&
         stale.result.stats.run.work.command_submit_count == 1u &&
         stale.result.stats.run.work.command_inflight_peak == 1u &&
         !MetalResidencySlidingCapability(backend,
                                          ResidencySlidingMemory::HostCoherent)
              .check.ok;
}

[[nodiscard]] bool
RunSlidingGateTerminalLoss(const std::shared_ptr<void> &backend,
                           const rund::AccelDevice &pick) {
  using namespace rund::node::accel::detail;
  MetalResidencySlidingDiagnostics before{};
  if (!InspectMetalResidencySliding(backend, before) || before.quarantined ||
      !InjectNativeResidencyTerminalLossOnce(pick)) {
    return false;
  }
  constexpr std::uint32_t control_generation = 800'001u;
  if (!SeedPreparedMetalPipelineGeneration(backend, control_generation - 1u)
           .ok) {
    return false;
  }
  const std::array<std::uint32_t, 1u> locals{0u};
  const BackendResidencySlidingDescriptor descriptor{
      .owner = backend.get(),
      .plan_identity = 0x4d'54'4c'4cu,
      .token = 800'002u,
      .generation = 800'003u,
      .read_mask = 1u,
      .write_mask = 1u,
      .descriptor_generation = 1u,
      .control_generation = control_generation,
      .stride = 1u,
  };
  SlidingGateWait loss{};
  if (!SubmitMetalResidencySliding(backend, descriptor, CompleteSlidingGate,
                                   &loss, KernelTiming::Submission,
                                   PipelineSubmitMode::Residency, locals)
           .ok) {
    return false;
  }
  loss.done.wait(false, std::memory_order_acquire);
  MetalResidencySlidingDiagnostics after{};
  if (!InspectMetalResidencySliding(backend, after)) {
    return false;
  }
  // The injected path intentionally omits the physical event.  Leave enough
  // time for any erroneously retained late handler to attempt a second close.
  std::this_thread::sleep_for(std::chrono::milliseconds{30});
  return !loss.result.check.ok &&
         loss.result.terminal == NativeTerminal::UnknownMayWrite &&
         loss.result.pipeline.submitted &&
         loss.result.stats.run.work.command_submit_count == 1u &&
         loss.callback_count.load(std::memory_order_relaxed) == 1u &&
         after.gpu_result_read_count == before.gpu_result_read_count &&
         after.command_submit_count == before.command_submit_count + 1u &&
         after.command_active == before.command_active + 1u &&
         after.quarantined &&
         !MetalResidencySlidingCapability(backend,
                                          ResidencySlidingMemory::HostCoherent)
              .check.ok &&
         loss.callback_count.load(std::memory_order_relaxed) == 1u;
}

} // namespace

int CheckMetalResidencySlidingGate() {
  using namespace rund::compute;
  constexpr std::size_t elements = 8u;
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .map<std::int32_t>("metal-sliding-gate", 1u,
                                        [](auto value) { return value + 1; })
                     .compile();
  auto input_backing =
      std::make_shared<AdmissionBacking>(elements * sizeof(std::int32_t));
  auto output_backing =
      std::make_shared<AdmissionBacking>(elements * sizeof(std::int32_t));
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::int32_t), detail::Type::I32, {}, input_backing);
  auto output = detail::make_virtual_buffer(
      elements, sizeof(std::int32_t), detail::Type::I32, {}, output_backing);
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
  rund::node::accel::detail::MetalResidencySlidingDiagnostics cold{};
  if (!rund::node::accel::detail::InspectMetalResidencySliding(backend, cold) ||
      !cold.ready || cold.quarantined || cold.retained_bytes == 0u ||
      cold.descriptor_identity == 0u || cold.pipeline_identity == 0u ||
      cold.gate_command_identity == 0u || cold.ready_event_identity == 0u ||
      cold.command_identity == 0u || cold.allocator_identity == 0u) {
    return 4;
  }
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
      return 5;
    }
  }
  if (!RunSlidingGateStale(backend)) {
    return 6;
  }
  rund::node::accel::detail::MetalResidencySlidingDiagnostics warm{};
  if (!rund::node::accel::detail::InspectMetalResidencySliding(backend, warm)) {
    return 7;
  }
  if (!warm.quarantined || cold.retained_bytes != warm.retained_bytes ||
      cold.descriptor_identity != warm.descriptor_identity ||
      cold.pipeline_identity != warm.pipeline_identity ||
      cold.gate_command_identity != warm.gate_command_identity ||
      cold.ready_event_identity != warm.ready_event_identity ||
      cold.command_identity != warm.command_identity ||
      cold.allocator_identity != warm.allocator_identity ||
      cold.pipeline_compile_count != warm.pipeline_compile_count ||
      cold.buffer_allocation_count != warm.buffer_allocation_count ||
      warm.command_submit_count < cold.command_submit_count ||
      // 5 + 9 + 257 + 17 authenticated coordinates, followed by the two
      // stale-row probes. Every coordinate owns exactly one native queue
      // submission.
      warm.command_submit_count - cold.command_submit_count != 290u ||
      warm.command_active != 0u) {
    return 8;
  }

  // Terminal-loss is necessarily a fresh adapter: the stale-row proof above
  // intentionally leaves its adapter sticky-quarantined.
  auto loss_opened = open(Target::metal());
  if (!loss_opened) {
    return loss_opened.reason() == Reason::AdapterUnavailable ? 0 : 9;
  }
  Device loss_device = std::move(loss_opened).value();
  auto loss_program =
      on(loss_device)
          .map<std::int32_t>("metal-sliding-gate-loss", 1u,
                             [](auto value) { return value + 1; })
          .compile();
  auto loss_input_backing =
      std::make_shared<AdmissionBacking>(elements * sizeof(std::int32_t));
  auto loss_output_backing =
      std::make_shared<AdmissionBacking>(elements * sizeof(std::int32_t));
  auto loss_input =
      detail::make_virtual_buffer(elements, sizeof(std::int32_t),
                                  detail::Type::I32, {}, loss_input_backing);
  auto loss_output =
      detail::make_virtual_buffer(elements, sizeof(std::int32_t),
                                  detail::Type::I32, {}, loss_output_backing);
  auto loss_prepared =
      loss_program && loss_input && loss_output
          ? detail::prepare_virtual_pipeline(
                detail::ProgramAccess::state(*loss_program),
                std::move(loss_input).value(), std::move(loss_output).value(),
                ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!loss_prepared) {
    return loss_prepared.reason() == Reason::BackendUnsupported ? 0 : 10;
  }
  const auto loss_state = loss_prepared.value();
  const auto loss_primary =
      loss_state == nullptr ? nullptr : loss_state->pipeline;
  auto *const loss_common =
      loss_primary == nullptr || !loss_primary->prepared.ok
          ? nullptr
          : static_cast<rund::node::accel::detail::prepared::PipelineState *>(
                loss_primary->prepared.owner.get());
  const std::shared_ptr<void> loss_backend =
      loss_common == nullptr ? std::shared_ptr<void>{} : loss_common->backend;
  const std::shared_ptr<detail::DeviceState> &loss_device_state =
      detail::DeviceAccess::state(loss_device);
  detail::AccelDeviceState *const loss_native =
      loss_device_state == nullptr ? nullptr
                                   : detail::accel_device(*loss_device_state);
  return loss_backend != nullptr && loss_native != nullptr &&
                 RunSlidingGateTerminalLoss(loss_backend, loss_native->pick)
             ? 0
             : 11;
}

} // namespace rund_node_test_pipeline

#endif
