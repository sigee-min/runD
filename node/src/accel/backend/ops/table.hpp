#pragma once

#include "../../kernel/callback.hpp"
#include "../../kernel/fault/domain.hpp"
#include "../../kernel/residency/device_vsm/preparation.hpp"
#include "../../kernel/residency/persistent_sliding.hpp"
#include "../../kernel/residency/schedule.hpp"
#include "../../kernel/residency/service_free_direct.hpp"
#include "../../kernel/residency/sliding.hpp"
#include "../../kernel/residency/window.hpp"
#include "../result.hpp"

#include <accel/api.hpp>
#include <accel/check.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace rund {
struct AccelDevice;
struct AccelContext;
struct Buffer;
struct BufferDesc;
struct AccelRunFacts;
struct RuntimeStats;
} // namespace rund

namespace rund::kernel {
struct ComputePlan;
}

namespace rund::node::accel {
struct AccelMemoryStats;
} // namespace rund::node::accel

namespace rund::node::accel::detail {

struct BackendRun;
struct BackendBatchEntry;
struct BackendPublish;
struct TileTransducer;
struct NestedAggregate;
struct PreparedMemory;
struct PreparedPipelineFailure;
struct PreparedPipelineMemory;
struct PreparedKernelPipelineReservation;
struct PreparedKernelRouteReservation;
struct PreparedKernelProgramRoute;
struct PreparedKernelTemplateRegistry;
struct PreparedMapRecurrenceReservation;
struct MapRecurrencePreparationPlan;
struct KernelExecution;
struct KernelExecutionStep;
struct BoundStep;
struct PreparedResidencyPersistentSlidingRole;
struct PreparedBackendManifest;
class PreparedMemoryMeter;
class PreparedPipelineMemoryMeter;
struct PreparedPipelineStatusLayout;
class RangeCaps;

enum class BackendBufferInitialization : std::uint8_t {
  Zeroed,
  FullOverwrite,
};

struct BackendOps final {
  rund::AccelApi api = rund::AccelApi::Auto;
  bool resident = false;
  std::uint32_t nested_aggregate_command_count = 0u;
  rund::Buffer (*create)(const rund::AccelDevice &, const rund::BufferDesc &,
                         BackendBufferInitialization, BackendBufferMemory,
                         std::uint64_t exact_storage_bytes) = nullptr;
  // Exact backend allocation charge for one logical storage Buffer. This is
  // also the pre-materialization Pipeline admission projection.
  std::uint64_t (*buffer_storage_bytes)(const rund::AccelDevice &,
                                        std::uint64_t) noexcept = nullptr;
  // Exact allocation charge for one reusable prepared-Pipeline staging
  // Buffer. The query creates no memory allocation and is consumed by the
  // existing Pipeline admission plan before backend materialization.
  std::uint64_t (*pipeline_transfer_storage_bytes)(
      const rund::AccelDevice &, std::uint64_t) noexcept = nullptr;
  rund::AccelCheck (*upload)(const rund::AccelDevice &,
                             const rund::kernel::ResidentBufferRef &,
                             const std::shared_ptr<void> &, const void *,
                             std::uint64_t, std::uint64_t) = nullptr;
  BackendUpload (*upload_batch)(const rund::AccelDevice &,
                                std::span<const UploadRoute>,
                                TransferCompletion,
                                TransferAuthority) = nullptr;
  BackendDownload (*download)(const rund::AccelDevice &,
                              const rund::kernel::ResidentBufferRef &,
                              const std::shared_ptr<void> &, void *,
                              std::uint64_t, std::uint64_t, bool) = nullptr;
  BackendDownload (*download_batch)(const rund::AccelDevice &,
                                    std::span<const DownloadRoute>,
                                    TransferAuthority) = nullptr;
  BackendCopy (*copy_batch)(const rund::AccelDevice &,
                            std::span<const CopyRoute>,
                            TransferAuthority) = nullptr;
  BackendLookup (*lookup)(const rund::AccelDevice &,
                          const rund::kernel::ResidentBufferRef &,
                          const std::shared_ptr<void> &) = nullptr;
  // Timing-free capability projection. The pointer is read-only, remains
  // stable while the authenticated resident handle lives, and is coherent
  // only after an exact native command terminal. Backends that cannot prove
  // all three properties leave this null or return an empty view.
  BackendHostView (*host_read)(
      const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
      const std::shared_ptr<void> &) noexcept = nullptr;
  // Timing-free coherent supply capability. The pointer is writable and
  // stable while the authenticated resident handle lives. It grants no
  // scheduling or range authority; Compute must separately authenticate the
  // exact private Input owner and live Authority lease.
  BackendHostWriteView (*host_write)(
      const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
      const std::shared_ptr<void> &) noexcept = nullptr;
  rund::RuntimeStats (*stats)(const rund::AccelDevice &) = nullptr;
  void (*reset)(const rund::AccelDevice &) = nullptr;
  rund::node::accel::AccelMemoryStats (*memory)(
      const rund::AccelDevice &) noexcept = nullptr;
  // Timing- and allocation-free backend fact consumed before the compute
  // VirtualPipeline materializes any physical owner. Unsupported adapters
  // return the product's typed capability failure through this owner.
  rund::AccelCheck (*virtual_pipeline_capability)(
      const rund::AccelDevice &) noexcept = nullptr;
  // The selected backend projects immutable, timing-free range-execution
  // capability facts once. Graph admission freezes the resulting plan before
  // any scratch, source, or pipeline identity is materialized.
  RangeCaps (*range_caps)(const rund::AccelDevice &) noexcept = nullptr;
  rund::AccelCheck (*run)(const BackendRun &) = nullptr;
  rund::AccelCheck (*prepare)(const BackendRun &, std::shared_ptr<void> &,
                              PreparedMemory &) = nullptr;
  rund::AccelCheck (*plan_pipeline_private)(
      const BackendRun &, PreparedKernelRouteReservation &) noexcept = nullptr;
  rund::AccelCheck (*plan_pipeline_program)(
      const KernelExecution &, const PreparedKernelProgramRoute &,
      PreparedKernelRouteReservation &) noexcept = nullptr;
  rund::AccelCheck (*plan_pipeline_recurrence)(
      const MapRecurrencePreparationPlan &,
      PreparedMapRecurrenceReservation &) noexcept = nullptr;
  rund::AccelCheck (*plan_pipeline_structure)(
      const rund::AccelContext &,
      PreparedKernelPipelineReservation &) noexcept = nullptr;
  PreparedBackendManifest (*build_step_manifest)(
      const KernelExecutionStep &, const rund::kernel::ComputePlan &,
      const BoundStep *, std::uint64_t) noexcept = nullptr;
  bool (*same_pipeline_program_template)(
      const KernelExecution &, const PreparedKernelProgramRoute &,
      const PreparedKernelProgramRoute &) noexcept = nullptr;
  bool (*same_pipeline_template)(const BackendRun &,
                                 const BackendRun &) noexcept = nullptr;
  // Exact runD-owned host structure retained by one immutable registry entry.
  // Adapter-global source caches and opaque driver allocations keep their own
  // device/cache authority and are deliberately not guessed here.
  PreparedMemory (*observe_pipeline_template)(const void *) noexcept = nullptr;
  rund::AccelCheck (*prepare_pipeline_private)(const BackendRun &,
                                               std::shared_ptr<void> &,
                                               PreparedMemory &) = nullptr;
  std::uint64_t (*traffic)(const std::shared_ptr<void> &) noexcept = nullptr;
  rund::AccelCheck (*run_batch)(std::span<const BackendBatchEntry>,
                                std::span<rund::AccelCheck>,
                                std::shared_ptr<void> &,
                                rund::RuntimeStats &) = nullptr;
  rund::AccelCheck (*prepare_pipeline)(
      std::span<const BackendBatchEntry>, std::span<const BackendBatchEntry>,
      std::span<const std::uint8_t>, std::span<const TileTransducer>,
      std::span<const NestedAggregate>, std::span<const BackendPublish>,
      PreparedKernelTemplateRegistry &, PreparedPipelineStatusLayout &, bool,
      std::shared_ptr<void> &, PreparedPipelineMemory &,
      PreparedPipelineMemoryMeter *, AccelRunFacts &,
      PreparedPipelineFailure &) = nullptr;
  rund::AccelCheck (*seed_prepared_pipeline_generation)(
      const std::shared_ptr<void> &, std::uint32_t) noexcept = nullptr;
  rund::AccelCheck (*prepare_pipeline_transfer)(
      const std::shared_ptr<void> &, const UploadRoute &, const DownloadRoute &,
      std::uint64_t) noexcept = nullptr;
  rund::AccelCheck (*query_pipeline_residency)(const std::shared_ptr<void> &,
                                               bool &) noexcept = nullptr;
  // True only when the retained residency owner is committed and can accept
  // an exact selected-local submission now. Timing/allocation free.
  rund::AccelCheck (*pipeline_residency_ready)(const std::shared_ptr<void> &,
                                               bool &) noexcept = nullptr;
  rund::AccelCheck (*stage_pipeline_residency)(
      const std::shared_ptr<void> &, std::shared_ptr<void> &,
      std::uint64_t &) noexcept = nullptr;
  void (*commit_pipeline_residency)(const std::shared_ptr<void> &,
                                    std::shared_ptr<void>) noexcept = nullptr;
  BackendUpload (*upload_prepared_pipeline)(const std::shared_ptr<void> &,
                                            const void *,
                                            std::uint64_t) noexcept = nullptr;
  BackendDownload (*download_prepared_pipeline)(
      const std::shared_ptr<void> &, void *, std::uint64_t,
      std::uint64_t *) noexcept = nullptr;
  rund::AccelCheck (*submit_prepared_pipeline)(
      const std::shared_ptr<void> &, KernelCompletion, void *, KernelTiming,
      PipelineSubmitMode, std::span<const std::uint32_t>) noexcept = nullptr;
  // One fixed backend-neutral handoff for W<=4 exact selected Pipeline
  // batches. Backends copy the fixed request and emit internal Releases plus
  // one Final; Host-service policy and cache selection never cross this seam.
  rund::AccelCheck (*submit_prepared_window)(
      const BackendResidencyWindowRequest &) noexcept = nullptr;
  // True only when every window Release/Final callback is delivered from a
  // backend-owned cold service lane after submit returns. The recurrent
  // arbitrary-Q controller never admits a backend that could run Host I/O on
  // its caller thread.
  bool residency_window_callbacks_async = false;
  rund::AccelCheck (*signal_prepared_window)(
      const std::shared_ptr<void> &,
      const BackendResidencyWindowSignal &) noexcept = nullptr;
  rund::AccelCheck (*abort_prepared_window)(
      const std::shared_ptr<void> &,
      const BackendResidencyWindowAbort &) noexcept = nullptr;
  // Immutable whole-run recurrence. Prepare may materialize a cold candidate,
  // but it reports every runD-owned retained/transient byte and returns its
  // owner before Authority begins. Submit consumes that exact candidate,
  // accepts all Q native batches, and Signal only opens already queued work.
  BackendResidencySchedulePreparation (*prepare_prepared_schedule)(
      std::span<const BackendResidencyScheduleRole>, std::uint64_t epoch_count,
      std::size_t tail_local_count) noexcept = nullptr;
  rund::AccelCheck (*submit_prepared_schedule)(
      const BackendResidencyScheduleRequest &) noexcept = nullptr;
  rund::AccelCheck (*signal_prepared_schedule)(
      const std::shared_ptr<void> &,
      const BackendResidencyWindowSignal &) noexcept = nullptr;
  rund::AccelCheck (*abort_prepared_schedule)(
      const std::shared_ptr<void> &,
      const BackendResidencyWindowAbort &) noexcept = nullptr;
  // Fixed-state recurrent selection capability. Submission reuses the
  // existing prepared-Pipeline native owner after its exact prior terminal;
  // no Q-sized descriptor or command owner is materialized here.
  BackendResidencySlidingCapability (*prepared_sliding_capability)(
      const std::shared_ptr<void> &, ResidencySlidingMemory) noexcept = nullptr;
  rund::AccelCheck (*submit_prepared_sliding)(
      const std::shared_ptr<void> &, const BackendResidencySlidingDescriptor &,
      KernelCompletion, void *, KernelTiming, PipelineSubmitMode,
      std::span<const std::uint32_t>) noexcept = nullptr;
  // Whole-run product lowering. The request already contains backend-native
  // strong owners projected from authenticated common Prepared Pipelines.
  // Preparation encodes the complete recurrence but performs no queue submit.
  // This pure companion is queried before any persistent Authority lease or
  // backend owner is materialized. It must not allocate, claim, or submit.
  PersistentResidencySlidingCapability (*query_persistent_sliding_capability)(
      std::span<const PreparedResidencyPersistentSlidingRole>,
      std::uint64_t, ResidencySlidingMemory,
      PersistentResidencySlidingMode) noexcept = nullptr;
  PersistentResidencySlidingPreparation (*prepare_persistent_sliding)(
      const PersistentResidencySlidingRequest &) noexcept = nullptr;
  // One aggregate page-coordinate recurrence. The backend receives an exact
  // common proof and exposes no per-page submit or callback surface.
  DeviceVsmPreparation (*prepare_device_vsm)(
      const rund::AccelDevice &,
      const std::shared_ptr<const DeviceVsmProof> &) noexcept = nullptr;
  // Backend-native proof for one already prepared aggregate recurrence. The
  // common prepared owner independently supplies fixed_common_storage.
  ServiceFreeDirectCapability (*service_free_direct_capability)(
      const std::shared_ptr<void> &,
      const ServiceFreeDirectProof &) noexcept = nullptr;
  rund::AccelCheck (*submit_prepared)(const BackendRun &,
                                      const std::shared_ptr<void> &,
                                      KernelCompletion, void *,
                                      PreparedMemoryMeter *,
                                      const std::shared_ptr<void> &,
                                      KernelTiming) noexcept = nullptr;
  bool (*inject_device_lost_once)(const rund::AccelDevice &,
                                  SubmitKind) noexcept = nullptr;
  bool (*inject_download_failure_once)(const rund::AccelDevice &) noexcept =
      nullptr;
  bool (*inject_host_read_unavailable_once)(
      const rund::AccelDevice &) noexcept = nullptr;
  bool (*inject_host_write_unavailable_once)(
      const rund::AccelDevice &) noexcept = nullptr;
  bool (*inject_residency_terminal_loss_once)(
      const rund::AccelDevice &) noexcept = nullptr;
  bool (*inject_trace_unavailable_once)(const rund::AccelDevice &) noexcept =
      nullptr;
  bool (*inject_trace_resolve_device_lost_once)(
      const rund::AccelDevice &) noexcept = nullptr;
};

} // namespace rund::node::accel::detail
