#pragma once

#include "residency.hpp"
#include "../run.hpp"
#include "../template/registry.hpp"
#include "../../backend/run.hpp"
#include "../../preparation.hpp"
#include "../../residency/service_free_direct.hpp"
#include "../../scratch.hpp"
#include "../../view.hpp"
#include <accel/context/value.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] PreparedKernelRun
PrepareKernelRun(const rund::AccelContext &context,
                 const rund::AccelKernel &kernel, const rund::AccelRun &run,
                 KernelPreparationMode mode = KernelPreparationMode::Standalone,
                 const KernelViewLayout *views = nullptr,
                 const RunBinds *view_binds = nullptr,
                 const KernelScratchLayout *scratch = nullptr);

[[nodiscard]] rund::AccelEvidence
RunPreparedKernel(const rund::AccelContext &context,
                  const PreparedKernelRun &prepared);

[[nodiscard]] PreparedBatchEvidence
RunPreparedKernelBatch(const rund::AccelContext &context,
                       std::span<const PreparedKernelRun *const> prepared,
                       std::span<rund::AccelEvidence> jobs,
                       std::shared_ptr<void> &workspace,
                       PreparedBatchStart start, void *user);

[[nodiscard]] PreparedKernelPipeline
PrepareKernelPipeline(const rund::AccelContext &context,
                      std::span<const PreparedKernelRun *const> prepared,
                      std::span<const std::uint8_t> barriers,
                      std::span<const std::uint32_t> declared_steps,
                      std::span<const BackendRecurrence> recurrences,
                      std::span<const BackendPublish> publications,
                      std::uint32_t declared_step_count,
                      std::uint32_t generation_stride, bool profile_steps,
                      PreparedKernelTemplateRegistry *templates = nullptr);

[[nodiscard]] std::shared_ptr<const ServiceFreeDirectProof>
PreparedKernelPipelineServiceFreeDirectProof(
    const PreparedKernelPipeline &prepared) noexcept;

[[nodiscard]] ServiceFreeDirectPreparation
PreparePreparedKernelPipelineServiceFreeDirect(
    const PreparedKernelPipeline &prepared) noexcept;

[[nodiscard]] PreparedKernelPipelineReservation PlanPreparedKernelPipeline(
    const rund::AccelContext &context,
    std::span<const PreparedKernelRun *const> prepared,
    std::span<const BackendRecurrence> recurrences,
    PreparedKernelPipelineShape shape,
    PreparedKernelTemplateRegistry *templates = nullptr) noexcept;

[[nodiscard]] PreparedKernelPipelineReservation PlanPreparedKernelPipelineLimit(
    const rund::AccelContext &context,
    std::span<const PreparedKernelProgramRoute> routes,
    PreparedKernelPipelineShape shape,
    PreparedKernelTemplateRegistry &templates) noexcept;

[[nodiscard]] bool PreparedKernelPipelineReservationWithin(
    const PreparedKernelPipelineReservation &reservation,
    const PreparedKernelPipelineReservation &limit) noexcept;

[[nodiscard]] PreparedPipelineEvidence RunPreparedKernelPipeline(
    const rund::AccelContext &context, const PreparedKernelPipeline &prepared,
    PipelineSubmitMode mode = PipelineSubmitMode::Standard);

[[nodiscard]] rund::AccelCheck
SeedPreparedKernelPipelineGeneration(const PreparedKernelPipeline &prepared,
                                     std::uint32_t generation) noexcept;

[[nodiscard]] PersistentResidencySlidingCapability
QueryPreparedKernelPipelinePersistentSlidingCapability(
    std::span<const PreparedResidencyPersistentSlidingRole>,
    std::uint64_t coordinate_count, ResidencySlidingMemory,
    PersistentResidencySlidingMode) noexcept;

[[nodiscard]] PreparedResidencyPersistentSlidingPreparation
PrepareKernelPipelinePersistentSliding(
    const PersistentResidencySlidingRequest &,
    std::span<const PreparedResidencyPersistentSlidingRole>) noexcept;

[[nodiscard]] rund::AccelCheck PreparePreparedKernelPipelineTransfer(
    const PreparedKernelPipeline &prepared, const UploadRoute &upload,
    const DownloadRoute &download, std::uint64_t exact_storage_bytes) noexcept;

[[nodiscard]] rund::AccelCheck
StagePreparedKernelPipelineResidency(const PreparedKernelPipeline &prepared,
                                     std::shared_ptr<void> &candidate,
                                     std::uint64_t &retained_bytes) noexcept;

void CommitPreparedKernelPipelineResidency(
    const PreparedKernelPipeline &prepared,
    std::shared_ptr<void> candidate) noexcept;

[[nodiscard]] rund::AccelCheck
QueryPreparedKernelPipelineResidency(const PreparedKernelPipeline &prepared,
                                     bool &supported) noexcept;

[[nodiscard]] rund::AccelCheck
PreparedKernelPipelineResidencyReady(const PreparedKernelPipeline &prepared,
                                     bool &ready) noexcept;

[[nodiscard]] rund::AccelCheck
PreparedKernelPipelineWindowReady(const rund::AccelContext &,
                                  std::span<const PreparedKernelPipeline>,
                                  bool &ready) noexcept;

[[nodiscard]] BackendUpload
UploadPreparedKernelPipeline(const PreparedKernelPipeline &prepared,
                             const void *data, std::uint64_t bytes) noexcept;

[[nodiscard]] BackendDownload
DownloadPreparedKernelPipeline(const PreparedKernelPipeline &prepared,
                               void *data, std::uint64_t bytes,
                               std::uint64_t *payload_hash) noexcept;

[[nodiscard]] rund::AccelCheck SubmitPreparedKernelPipeline(
    const rund::AccelContext &context, const PreparedKernelPipeline &prepared,
    std::shared_ptr<void> lifetime, PreparedPipelineCompletion completion,
    void *user, KernelTiming timing = KernelTiming::Submission,
    PipelineSubmitMode mode = PipelineSubmitMode::Standard) noexcept;

[[nodiscard]] rund::AccelCheck SubmitPreparedKernelPipelineSelection(
    const rund::AccelContext &context, const PreparedKernelPipeline &prepared,
    std::shared_ptr<void> lifetime, PreparedPipelineCompletion completion,
    void *user, KernelTiming timing, PipelineSubmitMode mode,
    std::span<const std::uint32_t> selected_steps) noexcept;

[[nodiscard]] rund::AccelCheck
SubmitPreparedKernelPipelineWindow(const rund::AccelContext &,
                                   const PreparedResidencyWindowRequest &,
                                   PreparedResidencyWindowControl &) noexcept;

[[nodiscard]] rund::AccelCheck ClaimPreparedKernelPipelineStream(
    const rund::AccelContext &, std::span<const PreparedKernelPipeline>,
    std::uint64_t plan_identity, std::uint64_t token, std::uint64_t generation,
    PreparedResidencyStreamControl &) noexcept;

[[nodiscard]] rund::AccelCheck SubmitPreparedKernelPipelineStreamWindow(
    const rund::AccelContext &, const PreparedResidencyWindowRequest &,
    PreparedResidencyWindowControl &,
    PreparedResidencyStreamControl &) noexcept;

[[nodiscard]] rund::AccelCheck ReleasePreparedKernelPipelineStream(
    PreparedResidencyStreamControl &, std::uint64_t plan_identity,
    std::uint64_t token, std::uint64_t generation, bool quarantine) noexcept;

[[nodiscard]] rund::AccelCheck QuarantinePreparedKernelPipelineStream(
    PreparedResidencyStreamControl &, std::uint64_t plan_identity,
    std::uint64_t token, std::uint64_t generation) noexcept;

[[nodiscard]] rund::AccelCheck SignalPreparedKernelPipelineWindow(
    const rund::AccelContext &, const PreparedKernelPipeline &,
    const BackendResidencyWindowSignal &) noexcept;

[[nodiscard]] rund::AccelCheck
AbortPreparedKernelPipelineWindow(const rund::AccelContext &,
                                  PreparedResidencyWindowControl &,
                                  const BackendResidencyWindowAbort &) noexcept;

[[nodiscard]] BackendResidencySchedulePreparation PrepareKernelPipelineSchedule(
    const rund::AccelContext &, std::span<const PreparedResidencyScheduleRole>,
    std::uint64_t epoch_count, std::size_t tail_local_count) noexcept;

[[nodiscard]] rund::AccelCheck
SubmitPreparedKernelPipelineSchedule(const rund::AccelContext &,
                                     const PreparedResidencyScheduleRequest &,
                                     PreparedResidencyScheduleControl &,
                                     PreparedResidencyStreamControl &) noexcept;

[[nodiscard]] rund::AccelCheck SignalPreparedKernelPipelineSchedule(
    const rund::AccelContext &, const PreparedKernelPipeline &,
    const BackendResidencyWindowSignal &) noexcept;

[[nodiscard]] rund::AccelCheck AbortPreparedKernelPipelineSchedule(
    const rund::AccelContext &, PreparedResidencyScheduleControl &,
    const BackendResidencyWindowAbort &) noexcept;

[[nodiscard]] rund::AccelCheck PrepareKernelPipelineSliding(
    const rund::AccelContext &, std::span<const PreparedResidencySlidingRole>,
    ResidencySlidingMemory, PreparedResidencySlidingControl &) noexcept;

[[nodiscard]] rund::AccelCheck
SubmitPreparedKernelPipelineSliding(const rund::AccelContext &,
                                    const PreparedResidencySlidingRequest &,
                                    PreparedResidencySlidingControl &) noexcept;

[[nodiscard]] rund::AccelCheck
WakePreparedKernelPipelineSliding(PreparedResidencySlidingControl &) noexcept;

[[nodiscard]] rund::AccelCheck SubmitPreparedKernel(
    const rund::AccelContext &context, const PreparedKernelRun &prepared,
    std::shared_ptr<void> lifetime, PreparedKernelCompletion completion,
    void *user, KernelTiming timing = KernelTiming::Submission) noexcept;

[[nodiscard]] PreparedMemory
ReadPreparedKernelMemory(const PreparedKernelRun &prepared) noexcept;

[[nodiscard]] PreparedPipelineMemory ReadPreparedKernelPipelineMemory(
    const PreparedKernelPipeline &prepared) noexcept;

[[nodiscard]] PreparedMemory ReadPreparedKernelTemplateRegistryMemory(
    const PreparedKernelTemplateRegistry &registry) noexcept;

} // namespace rund::node::accel::detail
