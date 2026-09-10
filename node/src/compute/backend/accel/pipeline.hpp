#pragma once

#include "../../../accel/kernel/backend/run.hpp"

#include "../../backend.hpp"
#include "../../stats.hpp"

namespace rund::compute::detail::accel_backend {

[[nodiscard]] MemoryCounter
device_staging(const DeviceState &device) noexcept;

[[nodiscard]] MemoryCounter job_staging(const JobState &job) noexcept;

[[nodiscard]] node::accel::detail::PreparedPipelineMemory
pipeline_memory(const PipelineState &pipeline) noexcept;

[[nodiscard]] node::accel::detail::PreparedKernelPipelineReservation
plan_pipeline_preparation(
    const DeviceState &device,
    std::span<const node::accel::detail::PreparedKernelProgramRoute> routes,
    node::accel::detail::PreparedKernelPipelineShape shape,
    node::accel::detail::PreparedKernelTemplateRegistry &templates) noexcept;

[[nodiscard]] node::accel::detail::PreparedKernelPipeline prepare_pipeline(
    const DeviceState &device,
    std::span<const node::accel::detail::PreparedKernelRun *const> prepared,
    std::span<const std::uint8_t> barriers,
    std::span<const std::uint32_t> declared_steps,
    std::span<const node::accel::detail::BackendRecurrence> recurrences,
    std::span<const node::accel::detail::BackendPublish> publications,
    std::uint32_t declared_step_count, std::uint32_t generation_stride,
    bool profile_steps,
    node::accel::detail::PreparedKernelTemplateRegistry *templates);

[[nodiscard]] node::accel::detail::PreparedPipelineEvidence run_pipeline(
    const DeviceState &device,
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    node::accel::detail::PipelineSubmitMode mode);

[[nodiscard]] rund::AccelCheck submit_residency_pipeline(
    const DeviceState &device,
    const node::accel::detail::PreparedKernelPipeline &prepared,
    std::span<const std::uint32_t> locals, std::shared_ptr<void> lifetime,
    node::accel::detail::PreparedPipelineCompletion completion,
    void *user) noexcept;

[[nodiscard]] Status virtual_pipeline_capability(
    const DeviceState &device) noexcept;

[[nodiscard]] rund::AccelCheck submit_pipeline(
    const DeviceState &device,
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    std::shared_ptr<void> lifetime,
    node::accel::detail::PreparedPipelineCompletion completion, void *user,
    node::accel::detail::KernelTiming timing,
    node::accel::detail::PipelineSubmitMode mode) noexcept;

[[nodiscard]] Status submit_residency_window(
    const DeviceState &device,
    const node::accel::detail::PreparedResidencyWindowRequest &request,
    node::accel::detail::PreparedResidencyWindowControl &control) noexcept;

[[nodiscard]] Status submit_residency_stream_window(
    const DeviceState &device,
    const node::accel::detail::PreparedResidencyWindowRequest &request,
    node::accel::detail::PreparedResidencyWindowControl &window,
    node::accel::detail::PreparedResidencyStreamControl &stream) noexcept;

[[nodiscard]] Status claim_residency_stream(
    const DeviceState &device,
    std::span<const node::accel::detail::PreparedKernelPipeline> pipelines,
    std::uint64_t plan_identity, std::uint64_t token, std::uint64_t generation,
    node::accel::detail::PreparedResidencyStreamControl &control) noexcept;

[[nodiscard]] Status release_residency_stream(
    const DeviceState &device,
    node::accel::detail::PreparedResidencyStreamControl &control,
    std::uint64_t plan_identity, std::uint64_t token, std::uint64_t generation,
    bool quarantine) noexcept;

[[nodiscard]] Status quarantine_residency_stream(
    const DeviceState &device,
    node::accel::detail::PreparedResidencyStreamControl &control,
    std::uint64_t plan_identity, std::uint64_t token,
    std::uint64_t generation) noexcept;

[[nodiscard]] Status residency_window_capability(
    const DeviceState &device,
    std::span<const node::accel::detail::PreparedKernelPipeline> pipelines,
    bool &ready) noexcept;

[[nodiscard]] Status signal_residency_window(
    const DeviceState &device,
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    const node::accel::detail::BackendResidencyWindowSignal &signal) noexcept;

[[nodiscard]] Status abort_residency_window(
    const DeviceState &device,
    node::accel::detail::PreparedResidencyWindowControl &control,
    const node::accel::detail::BackendResidencyWindowAbort &abort) noexcept;

[[nodiscard]] node::accel::detail::BackendResidencySchedulePreparation
prepare_residency_schedule(
    const DeviceState &device,
    std::span<const node::accel::detail::PreparedResidencyScheduleRole> roles,
    std::uint64_t epoch_count, std::size_t tail_local_count) noexcept;

[[nodiscard]] Status submit_residency_schedule(
    const DeviceState &device,
    const node::accel::detail::PreparedResidencyScheduleRequest &request,
    node::accel::detail::PreparedResidencyScheduleControl &schedule,
    node::accel::detail::PreparedResidencyStreamControl &stream) noexcept;

[[nodiscard]] Status signal_residency_schedule(
    const DeviceState &device,
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    const node::accel::detail::BackendResidencyWindowSignal &signal) noexcept;

[[nodiscard]] Status abort_residency_schedule(
    const DeviceState &device,
    node::accel::detail::PreparedResidencyScheduleControl &control,
    const node::accel::detail::BackendResidencyWindowAbort &abort) noexcept;

[[nodiscard]] Status prepare_residency_sliding(
    const DeviceState &device,
    std::span<const node::accel::detail::PreparedResidencySlidingRole> roles,
    node::accel::detail::ResidencySlidingMemory memory,
    node::accel::detail::PreparedResidencySlidingControl &control) noexcept;

[[nodiscard]] Status submit_residency_sliding(
    const DeviceState &device,
    const node::accel::detail::PreparedResidencySlidingRequest &request,
    node::accel::detail::PreparedResidencySlidingControl &control) noexcept;

[[nodiscard]] Status wake_residency_sliding(
    const DeviceState &device,
    node::accel::detail::PreparedResidencySlidingControl &control) noexcept;

[[nodiscard]] rund::AccelCheck seed_pipeline_generation(
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    std::uint32_t generation) noexcept;

} // namespace rund::compute::detail::accel_backend
