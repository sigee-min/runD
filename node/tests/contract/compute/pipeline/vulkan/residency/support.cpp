#include "internal.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_METAL) && \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <cstdio>
#include <mutex>
#include <utility>

namespace rund_node_test_pipeline {

[[nodiscard]] std::uint64_t ReadVulkanResidencyQueueSubmits(
    const std::shared_ptr<rund::compute::detail::PipelineState> &pipeline) {
  const auto *const native =
      pipeline == nullptr || pipeline->device == nullptr
          ? nullptr
          : rund::compute::detail::accel_device(*pipeline->device);
  return native == nullptr ? 0u
                           : rund::node::accel::ReadRuntimeStats(native->pick)
                                 .run.work.command_submit_count;
}

} // namespace rund_node_test_pipeline

namespace rund_node_test_pipeline_vulkan_residency {

[[nodiscard]] std::uint32_t
ScheduleControlGeneration(const ScheduleWait &wait,
                          const std::uint64_t epoch) noexcept {
  const std::size_t role = static_cast<std::size_t>(epoch % 4u);
  return wait.first_generation[role] +
         static_cast<std::uint32_t>(epoch / 4u) * wait.generation_stride;
}

void CompleteScheduleRelease(
    void *const raw,
    accel::PreparedResidencyScheduleRelease &&release) noexcept {
  auto *const wait = static_cast<ScheduleWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  const std::uint64_t epoch = release.receipt.epoch;
  const bool valid =
      epoch < wait->epoch_count && release.receipt.check.ok &&
      release.receipt.terminal == accel::NativeTerminal::Known &&
      release.receipt.backend_sequence == epoch + 1u &&
      release.receipt.bank == epoch % 2u && release.receipt.dispatched &&
      release.receipt.completed && release.receipt.may_write &&
      release.evidence.check.ok && release.evidence.submitted &&
      release.evidence.terminal == accel::NativeTerminal::Known &&
      release.evidence.control_observed && release.evidence.control_valid &&
      release.evidence.control.generation ==
          ScheduleControlGeneration(*wait, epoch);
  if (!valid) {
    wait->valid.store(false, std::memory_order_release);
  }
  wait->releases.fetch_add(1u, std::memory_order_acq_rel);
  const std::uint64_t next = epoch + 2u;
  if (next >= wait->epoch_count) {
    return;
  }
  const std::size_t role = static_cast<std::size_t>(next % 4u);
  if (!accel::SignalPreparedKernelPipelineSchedule(
           wait->context, wait->pipelines[next % 2u],
           accel::BackendResidencyWindowSignal{
               .plan_identity = wait->plan,
               .token = wait->token,
               .generation = wait->generation,
               .epoch = next,
               .control_generation = ScheduleControlGeneration(*wait, next),
               .bank = static_cast<std::uint8_t>(role % 2u),
           })
           .ok) {
    wait->valid.store(false, std::memory_order_release);
  }
}

void CompleteScheduleFinal(
    void *const raw, accel::PreparedResidencyScheduleFinal &&final) noexcept {
  auto *const wait = static_cast<ScheduleWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->final = std::move(final.evidence);
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

void CompleteRelease(
    void *const raw, accel::PreparedResidencyWindowRelease &&release) noexcept {
  auto *const wait = static_cast<WindowWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  const std::uint64_t epoch = release.receipt.epoch;
  const bool in_range =
      epoch >= wait->first_epoch && epoch - wait->first_epoch < wait->count;
  const std::size_t slot =
      in_range ? static_cast<std::size_t>(epoch - wait->first_epoch)
               : wait->count;
  const bool suppressed = epoch == wait->suppressed;
  const auto terminal =
      wait->aborted ? accel::NativeTerminal::UnknownMayWrite
                    : accel::NativeTerminal::Known;
  const bool valid =
      in_range && release.receipt.backend_sequence == epoch + 1u &&
      release.receipt.bank == epoch % 2u && release.receipt.dispatched &&
      release.receipt.completed == !wait->aborted &&
      release.receipt.terminal == terminal &&
      release.receipt.check.ok == (!suppressed && !wait->aborted) &&
      release.receipt.may_write == (wait->aborted || !suppressed) &&
      release.evidence.submitted &&
      release.evidence.check.ok == (!suppressed && !wait->aborted) &&
      release.evidence.terminal == terminal &&
      (wait->aborted ||
       (release.evidence.control_observed &&
        release.evidence.control.generation == wait->control + slot));
  if (!valid) {
    std::fprintf(
        stderr,
        "vulkan release e=%zu suppressed=%u rcheck=%u term=%u dispatched=%u "
        "completed=%u may=%u echeck=%u submitted=%u observed=%u valid=%u "
        "generation=%u/%u\n",
        static_cast<std::size_t>(epoch), static_cast<unsigned>(suppressed),
        static_cast<unsigned>(release.receipt.check.ok),
        static_cast<unsigned>(release.receipt.terminal),
        static_cast<unsigned>(release.receipt.dispatched),
        static_cast<unsigned>(release.receipt.completed),
        static_cast<unsigned>(release.receipt.may_write),
        static_cast<unsigned>(release.evidence.check.ok),
        static_cast<unsigned>(release.evidence.submitted),
        static_cast<unsigned>(release.evidence.control_observed),
        static_cast<unsigned>(release.evidence.control_valid),
        release.evidence.control.generation,
        wait->control + static_cast<std::uint32_t>(slot));
    wait->valid.store(false, std::memory_order_release);
  }
  wait->releases.fetch_add(1u, std::memory_order_acq_rel);
  if (!wait->aborted && in_range && slot + 2u < wait->count) {
    const std::uint64_t next = epoch + 2u;
    const std::size_t next_slot = slot + 2u;
    accel::BackendResidencyWindowSignal signal{
        .plan_identity = wait->plan,
        .token = wait->token,
        .generation = wait->generation,
        .epoch = next,
        .control_generation =
            wait->control + static_cast<std::uint32_t>(next_slot),
        .bank = static_cast<std::uint8_t>(next % 2u),
    };
    if (next == wait->suppressed) {
      signal.admission = {false, "compute_backend_failed"};
    }
    if (!accel::SignalPreparedKernelPipelineWindow(
             wait->context, wait->pipelines[next_slot % 2u], signal)
             .ok) {
      wait->valid.store(false, std::memory_order_release);
    }
  }
}

void CompleteFinal(void *const raw,
                   accel::PreparedResidencyWindowFinal &&final) noexcept {
  auto *const wait = static_cast<WindowWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  if (final.evidence.first_epoch != wait->first_epoch) {
    wait->valid.store(false, std::memory_order_release);
  }
  wait->final = std::move(final.evidence);
  wait->done.store(true, std::memory_order_release);
  wait->done.notify_one();
}

[[nodiscard]] int InitializeResidencyFixture(ResidencyFixture &fixture) {
  using namespace rund::compute;
  constexpr std::size_t elements = 8u;
  auto opened = open(Target::vulkan());
  if (!opened) {
    if (opened.reason() == Reason::AdapterUnavailable) {
      fixture.skipped = true;
      return 0;
    }
    return 1;
  }
  fixture.device =
      std::make_unique<Device>(std::move(opened).value());
  auto program = on(*fixture.device)
                     .map<std::uint32_t>("vulkan-window", 1u,
                                         [](auto value) { return value + 1u; })
                     .compile();
  fixture.input_backing =
      std::make_shared<rund_node_test_pipeline::WindowBacking>(
          elements * sizeof(std::uint32_t));
  fixture.output_backing =
      std::make_shared<rund_node_test_pipeline::WindowBacking>(
          elements * sizeof(std::uint32_t));
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {},
      fixture.input_backing);
  auto output = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {},
      fixture.output_backing);
  auto prepared =
      program && input && output
          ? detail::prepare_virtual_pipeline(
                detail::ProgramAccess::state(*program),
                std::move(input).value(), std::move(output).value(),
                ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    if (prepared.reason() == Reason::BackendUnsupported) {
      fixture.skipped = true;
      return 0;
    }
    return 2;
  }
  fixture.program =
      std::make_unique<U32Program>(std::move(program).value());
  fixture.state = prepared.value();
  fixture.primary = fixture.state == nullptr ? nullptr : fixture.state->pipeline;
  fixture.alternate = fixture.state == nullptr
                          ? nullptr
                          : fixture.state->alternate_pipeline;
  fixture.native =
      fixture.primary == nullptr || fixture.primary->device == nullptr
          ? nullptr
          : detail::accel_device(*fixture.primary->device);
  if (fixture.primary == nullptr || fixture.alternate == nullptr ||
      fixture.native == nullptr || !fixture.primary->prepared.ok ||
      !fixture.alternate->prepared.ok) {
    return 3;
  }

  // Ordinary resident storage remains the private DeviceLocal allocation
  // class even when this adapter also exposes coherent VSM banks. Resolve the
  // actual native record and prove it was neither tagged nor mapped as the
  // VSM-only HostVisiblePreferred class.
  const std::shared_ptr<accel::PickToken> ordinary_token =
      accel::AdmitPick(fixture.native->pick);
  rund::Buffer ordinary = accel::CreateBackendBuffer(
      ordinary_token, rund::BufferDesc{.bytes = 64u,
                                       .usage = rund::BufferUsage::ReadWrite,
                                       .alignment = 16u});
  auto *const ordinary_adapter =
      static_cast<accel::VulkanAdapter *>(fixture.native->pick.backend.context);
  bool ordinary_private = ordinary_token != nullptr && ordinary.check.ok &&
                          ordinary_adapter != nullptr;
  if (ordinary_private) {
    auto &resident = accel::VulkanResidents(*ordinary_adapter);
    std::lock_guard lock{resident.mutex};
    const auto resolved = accel::ResolveVulkanResidentBuffer(
        resident,
        rund::kernel::ResidentBufferRef{
            .id = ordinary.id,
            .bytes = ordinary.bytes,
            .element_bytes = ordinary.element_bytes,
            .stride_bytes = ordinary.stride_bytes,
            .count = ordinary.count,
            .usage = accel::ResidentUsage(ordinary.usage),
        },
        ordinary.handle, "compute_resident_id_invalid");
    ordinary_private =
        resolved.check.ok && resolved.device_buffer != nullptr &&
        resolved.device_buffer->memory_use == accel::VulkanMemoryUse::Resident &&
        resolved.device_buffer->mapped == nullptr;
  }
  if (!ordinary_private) {
    return 3;
  }

  fixture.wait.context = fixture.native->context;
  fixture.wait.pipelines = {fixture.primary->prepared,
                            fixture.alternate->prepared};
  fixture.wait.plan = 0x56'4b'34u;
  bool ready = false;
  const rund::AccelCheck queried = accel::PreparedKernelPipelineWindowReady(
      fixture.wait.context, fixture.wait.pipelines, ready);
  if (!queried.ok || !ready) {
    // This is truthful native capability unavailability, not a passing W4
    // execution claim. Product MoltenVK coverage requires this branch not to
    // be taken on a HostVisible coherent adapter.
    fixture.skipped = true;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_vulkan_residency

#else

namespace rund_node_test_pipeline {

[[nodiscard]] std::uint64_t ReadVulkanResidencyQueueSubmits(
    const std::shared_ptr<rund::compute::detail::PipelineState> &) {
  return 0u;
}

} // namespace rund_node_test_pipeline

#endif
