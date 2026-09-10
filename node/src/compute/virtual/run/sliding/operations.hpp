#pragma once

#include "pipeline/rebase.hpp"

namespace rund::compute::detail::sliding_product_detail {

[[nodiscard]] bool control_range(const PipelineExecutionSnapshot &, std::size_t,
                                 std::uint64_t) noexcept;
[[nodiscard]] Status status_from(rund::AccelCheck) noexcept;
[[nodiscard]] residency::execution::TerminalKind
    terminal_from(node::accel::detail::NativeTerminal) noexcept;
void record_failure(SlidingProductRun &, Status, std::uint64_t) noexcept;
[[nodiscard]] bool reset_work(SlidingProductWork &, std::uint64_t) noexcept;
[[nodiscard]] bool discard_unissued_projection(SlidingProductWork &) noexcept;
[[nodiscard]] bool valid_project_request(const SlidingProductRun *,
                                         std::uint64_t, std::uint64_t,
                                         std::uint8_t) noexcept;
[[nodiscard]] bool run_accepts_projection(SlidingProductRun &) noexcept;
[[nodiscard]] bool ensure_projection(SlidingProductRun &, SlidingProductWork &,
                                     std::uint64_t) noexcept;

[[nodiscard]] Status begin_pipeline_sliding(SlidingProductRun &) noexcept;
[[nodiscard]] bool consume_persistent_pipeline_begin_failure_once() noexcept;
void inject_persistent_pipeline_begin_failure_once() noexcept;
[[nodiscard]] Status finish_pipeline_sliding(SlidingProductRun &,
                                             Status) noexcept;

[[nodiscard]] InputServiceResult
service_fetches(SlidingProductRun &, SlidingProductWork &, Status &) noexcept;
void record_fetch_io(SlidingProductRun &, const FetchIo &) noexcept;
[[nodiscard]] InputServiceResult service_promote(SlidingProductRun &,
                                                 SlidingProductWork &,
                                                 std::uint64_t,
                                                 Status &) noexcept;
[[nodiscard]] Status
copy_promote_frames(SlidingProductRun &, std::uint64_t,
                    const residency::execution::SlidingPromote &) noexcept;
[[nodiscard]] ProjectionResult select_native(
    SlidingProductRun &, SlidingProductWork &, std::uint64_t, std::uint8_t,
    node::accel::detail::PreparedResidencySlidingSelection &) noexcept;
[[nodiscard]] OutputServiceResult
service_drain(SlidingProductRun &, SlidingProductWork &, std::size_t,
              std::size_t, BufferReadView, residency::FrameRegion,
              Status &) noexcept;
[[nodiscard]] Status
copy_drain_frame(SlidingProductRun &, BufferReadView, residency::FrameRegion,
                 const residency::execution::SlidingDrain &) noexcept;
[[nodiscard]] OutputServiceResult service_persist(SlidingProductRun &,
                                                  SlidingProductWork &,
                                                  std::size_t, std::size_t,
                                                  Status &) noexcept;
void require_persist_recovery(SlidingProductRun &) noexcept;
[[nodiscard]] PersistIo
perform_persist_io(SlidingProductRun &,
                   const residency::execution::SlidingPersist &) noexcept;
void record_persist_io(SlidingProductRun &, const PersistIo &) noexcept;
[[nodiscard]] OutputServiceResult
service_output(SlidingProductRun &, SlidingProductWork &, Status &) noexcept;
[[nodiscard]] inline bool
requires_returned_output_service(SlidingProductRun &state,
                                 const std::uint64_t coordinate) noexcept {
  std::lock_guard lock{state.gate};
  return state.failure || coordinate < state.failure_coordinate;
}

[[nodiscard]] std::shared_ptr<SlidingProductOwner>
validate_execution_owner(VirtualPipelineState &,
                         const VirtualExecutionSlidingPrepared &) noexcept;
[[nodiscard]] residency::execution::Sliding
bind_controller(residency::Pool &, const SlidingProductOwner &,
                residency::ExecutionLease) noexcept;
void initialize_run(SlidingProductRun &, SlidingProductOwner &,
                    std::shared_ptr<void>, VirtualPipelineState &,
                    VirtualBacking &, VirtualBacking &,
                    const VirtualRunProjection &, Stats &, residency::Pool &,
                    residency::execution::Sliding) noexcept;
void release_run_links(SlidingProductRun &) noexcept;
[[nodiscard]] bool close_rejected_start(SlidingProductRun &, residency::Pool &,
                                        const SlidingProductOwner &,
                                        Status) noexcept;

[[nodiscard]] node::accel::detail::PersistentResidencySlidingRequest
make_persistent_request(SlidingProductRun &, const SlidingProductOwner &,
                        residency::ExecutionLease) noexcept;
[[nodiscard]] Status
prepare_persistent_lowering(SlidingProductRun &, SlidingProductOwner &,
                            residency::Pool &,
                            residency::ExecutionLease) noexcept;
[[nodiscard]] Status submit_persistent(SlidingProductRun &) noexcept;
[[nodiscard]] bool
persistent_submission_started(const SlidingProductRun &) noexcept;
[[nodiscard]] Status
service_persistent_recurrence(SlidingProductRun &) noexcept;
void final_persistent(
    void *, node::accel::detail::PersistentResidencySlidingFinal &&) noexcept;
void complete_persistent_product(SlidingProductRun &, Status) noexcept;
[[nodiscard]] Status prepare_persistent_publication(
    SlidingProductRun &,
    const node::accel::detail::PersistentResidencySlidingFinal &,
    PersistentPublication &) noexcept;
void commit_persistent_publication(void *) noexcept;
void inject_persistent_publication_preflight_failure_once() noexcept;
void inject_persistent_publication_pause_once(void (*)() noexcept) noexcept;
void fold_native_stats(
    SlidingProductRun &,
    const node::accel::detail::PersistentResidencySlidingFinal &) noexcept;
void fold_service_stats(SlidingProductRun &,
                        const residency::execution::SlidingEvidence &) noexcept;

void fold_sliding_stats(
    SlidingProductRun &, const residency::execution::SlidingEvidence &,
    const node::accel::detail::BackendResidencySlidingFinal &) noexcept;
[[nodiscard]] Status merged_run_status(SlidingProductRun &, Status) noexcept;
void quarantine_final(SlidingProductRun &) noexcept;
[[nodiscard]] GenerationClose close_generation(SlidingProductRun &,
                                               Status) noexcept;
void complete_known_final(SlidingProductRun &, Status) noexcept;

[[nodiscard]] bool admit_preparation(
    VirtualPipelineState &, const VirtualRunProjection &,
    residency::execution::SealResult &,
    node::accel::detail::PersistentResidencySlidingMode &) noexcept;
[[nodiscard]] Status
allocate_preparation_owner(const residency::execution::Plan &,
                           storage::Reservation &&,
                           std::shared_ptr<SlidingProductOwner> &) noexcept;
[[nodiscard]] Status
reserve_persistent_capacity(VirtualPipelineState &,
                            storage::Reservation &) noexcept;
[[nodiscard]] Status renew_persistent_memory(VirtualPipelineState &,
                                             SlidingProductOwner &,
                                             SlidingProductRun &) noexcept;
[[nodiscard]] Status commit_persistent_memory(
    SlidingProductOwner &,
    node::accel::detail::PreparedResidencyPersistentSlidingPreparation &&,
    node::accel::detail::PreparedResidencyPersistentSlidingPreparation
        &) noexcept;
[[nodiscard]] Status
snapshot_preparation_pipelines(VirtualPipelineState &,
                               SlidingProductOwner &) noexcept;
[[nodiscard]] Status build_preparation_roles(SlidingProductOwner &) noexcept;
[[nodiscard]] Status
verify_persistent_backend(const SlidingProductOwner &) noexcept;

} // namespace rund::compute::detail::sliding_product_detail
