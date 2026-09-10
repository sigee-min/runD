#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/backend/run.hpp"
#include "../kernel/callback.hpp"
#include "../kernel/memory.hpp"
#include "../kernel/preparation.hpp"
#include "../kernel/residency/device_vsm.hpp"
#include "../kernel/residency/persistent_sliding.hpp"
#include "../kernel/residency/schedule.hpp"
#include "../kernel/residency/sliding.hpp"
#include "../kernel/residency/window.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;

using MetalDeviceVsmPreparation = DeviceVsmPreparation;
[[nodiscard]] MetalDeviceVsmPreparation
PrepareMetalDeviceVsm(const rund::AccelDevice &,
                      const std::shared_ptr<const DeviceVsmProof> &) noexcept;

[[nodiscard]] rund::AccelCheck PrepareMetalResources(
    const rund::AccelDevice &pick, const BoundStep *steps,
    std::size_t step_count, std::uint64_t dispatch_count,
    KernelPreparationMode mode, const BoundResets *resets,
    const KernelViewLayout *views, const RunBinds *view_binds,
    const KernelScratchLayout *scratch, const BackendRun *template_probe,
    PreparedKernelTemplateRegistry *templates, std::uint32_t *failed_node,
    std::shared_ptr<void> &prepared, PreparedMemory &memory);
[[nodiscard]] std::uint64_t
MetalKernelTraffic(const std::shared_ptr<void> &prepared) noexcept;
[[nodiscard]] rund::AccelCheck
RunMetalResources(const rund::AccelDevice &pick,
                  const std::shared_ptr<void> &prepared);
[[nodiscard]] rund::AccelCheck
SubmitMetalResources(const rund::AccelDevice &pick,
                     const std::shared_ptr<void> &prepared,
                     KernelCompletion completion, void *user,
                     PreparedMemoryMeter *memory, KernelTiming timing) noexcept;

[[nodiscard]] rund::AccelCheck
SeedPreparedMetalPipelineGeneration(const std::shared_ptr<void> &prepared,
                                    std::uint32_t generation) noexcept;

[[nodiscard]] rund::AccelCheck
MetalPipelineResidencyReady(const std::shared_ptr<void> &prepared,
                            bool &ready) noexcept;
[[nodiscard]] BackendResidencySlidingCapability
MetalResidencySlidingCapability(const std::shared_ptr<void> &prepared,
                                ResidencySlidingMemory memory) noexcept;
[[nodiscard]] rund::AccelCheck SubmitMetalResidencySliding(
    const std::shared_ptr<void> &prepared,
    const BackendResidencySlidingDescriptor &, KernelCompletion, void *,
    KernelTiming, PipelineSubmitMode, std::span<const std::uint32_t>) noexcept;
struct MetalResidencySlidingDiagnostics final {
  std::uint64_t retained_bytes{};
  std::uint64_t descriptor_identity{};
  std::uint64_t pipeline_identity{};
  std::uint64_t gate_command_identity{};
  std::uint64_t ready_event_identity{};
  std::uint64_t command_identity{};
  std::uint64_t allocator_identity{};
  std::uint64_t pipeline_compile_count{};
  std::uint64_t buffer_allocation_count{};
  std::uint64_t command_submit_count{};
  std::uint64_t command_active{};
  std::uint64_t gpu_result_read_count{};
  bool ready{};
  bool quarantined{};
};
[[nodiscard]] bool
InspectMetalResidencySliding(const std::shared_ptr<void> &,
                             MetalResidencySlidingDiagnostics &) noexcept;
[[nodiscard]] bool InjectMetalResidencySlidingStaleDescriptorOnce(
    const std::shared_ptr<void> &) noexcept;
using MetalPersistentResidencySlidingPreparation =
    PersistentResidencySlidingPreparation;
enum class MetalPersistentResidencySlidingDiagnosticStage : std::uint8_t {
  Unknown,
  StageRearm,
  Claims,
  Control,
  Prepare,
  Encode,
  QueueCommit,
  CommonCommit,
  WaitSignal,
  WaitRetire,
  WaitPayload,
};
enum class MetalPersistentResidencySlidingDiagnosticPredicate : std::uint8_t {
  None,
  State,
  Request,
  Ticket,
  Coordinates,
  Sequence,
  Control,
  Command,
  Encode,
  Commit,
  Identity,
  Signal,
  Retire,
  Payload,
  DescriptorGeneration,
  AcceptedReason,
  ControlGeneration,
};
struct MetalPersistentResidencySlidingDiagnosticTrace final {
  MetalPersistentResidencySlidingDiagnosticStage stage{
      MetalPersistentResidencySlidingDiagnosticStage::Unknown};
  std::uint64_t stage_key{};
  MetalPersistentResidencySlidingDiagnosticPredicate predicate{
      MetalPersistentResidencySlidingDiagnosticPredicate::None};
  std::uint64_t predicate_key{};
  std::uint64_t coordinate{};
  bool signaled{};
  bool retired{};
  bool admission{};
  bool accepted{};
  std::uint32_t reason{};
  std::uint64_t expected_descriptor_generation{};
  std::uint64_t observed_descriptor_generation{};
  std::uint64_t expected_control_generation{};
  std::uint64_t observed_control_generation{};
};
struct MetalPersistentResidencySlidingDiagnostics final {
  std::uint64_t encoded_coordinate_count{};
  std::uint64_t encoded_intermediate_bytes{};
  std::uint64_t native_submit_count{};
  std::uint64_t epoch_native_submit_count{};
  std::uint64_t backend_epoch_callback_count{};
  std::uint64_t queue_commit_count{};
  std::uint64_t final_callback_count{};
  bool submitted{};
  bool final_sent{};
  bool quarantined{};
  MetalPersistentResidencySlidingDiagnosticTrace first_failure{};
  MetalPersistentResidencySlidingDiagnosticTrace last_wait{};
};
enum class MetalFusedDirectRecurrenceRetention : std::uint8_t {
  Unknown,
  Terminal,
  History,
};
// Exact Metal-private shape of the already admitted fused Direct Map
// recurrence. The iteration count is a four-byte kernel argument; it must not
// change the retained route or ICB storage. This is the executable O(1)
// backend primitive that a future service-free persistent ABI may own.
struct MetalFusedDirectRecurrenceDiagnostics final {
  std::uint64_t iteration_count{};
  std::uint64_t native_command_count{};
  std::uint64_t native_dispatch_count{};
  std::uint64_t native_storage_bytes{};
  std::uint64_t route_host_bytes{};
  std::uint64_t retained_bytes{};
  std::uint64_t iteration_argument_bytes{};
  MetalFusedDirectRecurrenceRetention retention{
      MetalFusedDirectRecurrenceRetention::Unknown};
};
[[nodiscard]] MetalPersistentResidencySlidingPreparation
PrepareMetalPersistentResidencySliding(
    const PersistentResidencySlidingRequest &) noexcept;
[[nodiscard]] const PersistentResidencySlidingServiceOps &
MetalPersistentResidencySlidingServiceOps() noexcept;
[[nodiscard]] bool InspectMetalPersistentResidencySliding(
    const std::shared_ptr<void> &,
    MetalPersistentResidencySlidingDiagnostics &) noexcept;
[[nodiscard]] bool InspectMetalFusedDirectRecurrence(
    const std::shared_ptr<void> &,
    MetalFusedDirectRecurrenceDiagnostics &) noexcept;
[[nodiscard]] rund::AccelCheck
SubmitMetalResidencyWindow(const BackendResidencyWindowRequest &) noexcept;
[[nodiscard]] rund::AccelCheck
SignalMetalResidencyWindow(const std::shared_ptr<void> &,
                           const BackendResidencyWindowSignal &) noexcept;
[[nodiscard]] rund::AccelCheck
AbortMetalResidencyWindow(const std::shared_ptr<void> &,
                          const BackendResidencyWindowAbort &) noexcept;

[[nodiscard]] BackendResidencySchedulePreparation
PrepareMetalResidencySchedule(std::span<const BackendResidencyScheduleRole>,
                              std::uint64_t epoch_count,
                              std::size_t tail_local_count) noexcept;
[[nodiscard]] rund::AccelCheck
SubmitMetalResidencySchedule(const BackendResidencyScheduleRequest &) noexcept;
[[nodiscard]] rund::AccelCheck
SignalMetalResidencySchedule(const std::shared_ptr<void> &,
                             const BackendResidencyWindowSignal &) noexcept;
[[nodiscard]] rund::AccelCheck
AbortMetalResidencySchedule(const std::shared_ptr<void> &,
                            const BackendResidencyWindowAbort &) noexcept;

} // namespace rund::node::accel::detail
