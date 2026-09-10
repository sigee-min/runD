#include "../interface/api.hpp"

#include "../model.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck
SeedPreparedKernelPipelineGeneration(const PreparedKernelPipeline &prepared,
                                     const std::uint32_t generation) noexcept {
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->seed_prepared_pipeline_generation == nullptr) {
    return rund::AccelCheck{
        false, "accel_prepared_pipeline_generation_seed_precondition"};
  }
  return pipeline->ops->seed_prepared_pipeline_generation(pipeline->backend,
                                                          generation);
}

rund::AccelCheck PreparePreparedKernelPipelineTransfer(
    const PreparedKernelPipeline &prepared, const UploadRoute &upload,
    const DownloadRoute &download,
    const std::uint64_t exact_storage_bytes) noexcept {
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->prepare_pipeline_transfer == nullptr) {
    return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
  }
  return pipeline->ops->prepare_pipeline_transfer(
      pipeline->backend, upload, download, exact_storage_bytes);
}

rund::AccelCheck
StagePreparedKernelPipelineResidency(const PreparedKernelPipeline &prepared,
                                     std::shared_ptr<void> &candidate,
                                     std::uint64_t &retained_bytes) noexcept {
  candidate.reset();
  retained_bytes = 0u;
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->stage_pipeline_residency == nullptr) {
    return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
  }
  return pipeline->ops->stage_pipeline_residency(pipeline->backend, candidate,
                                                 retained_bytes);
}

void CommitPreparedKernelPipelineResidency(
    const PreparedKernelPipeline &prepared,
    std::shared_ptr<void> candidate) noexcept {
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->commit_pipeline_residency == nullptr) {
    return;
  }
  pipeline->ops->commit_pipeline_residency(pipeline->backend,
                                           std::move(candidate));
}

rund::AccelCheck
QueryPreparedKernelPipelineResidency(const PreparedKernelPipeline &prepared,
                                     bool &supported) noexcept {
  supported = false;
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->query_pipeline_residency == nullptr) {
    return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
  }
  return pipeline->ops->query_pipeline_residency(pipeline->backend, supported);
}

rund::AccelCheck
PreparedKernelPipelineResidencyReady(const PreparedKernelPipeline &prepared,
                                     bool &ready) noexcept {
  ready = false;
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->pipeline_residency_ready == nullptr) {
    return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
  }
  return pipeline->ops->pipeline_residency_ready(pipeline->backend, ready);
}

rund::AccelCheck PreparedKernelPipelineWindowReady(
    const rund::AccelContext &context,
    const std::span<const PreparedKernelPipeline> pipelines,
    bool &ready) noexcept {
  ready = false;
  if (pipelines.empty() || pipelines.size() > ResidencyWindowCapacity) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const BackendOps *ops = nullptr;
  for (const PreparedKernelPipeline &pipeline : pipelines) {
    auto *const state =
        static_cast<prepared::PipelineState *>(pipeline.owner.get());
    if (!pipeline.ok || state == nullptr ||
        !prepared::ValidPipeline(context, *state) || state->ops == nullptr ||
        state->backend == nullptr ||
        state->ops->pipeline_residency_ready == nullptr ||
        state->ops->submit_prepared_window == nullptr ||
        state->ops->signal_prepared_window == nullptr ||
        state->ops->abort_prepared_window == nullptr ||
        !state->ops->residency_window_callbacks_async ||
        (ops != nullptr && ops != state->ops)) {
      return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
    }
    {
      std::lock_guard submission_lock{state->submission.mutex};
      if (state->submission.active()) {
        return rund::AccelCheck{true, "ok"};
      }
    }
    bool native_ready = false;
    const rund::AccelCheck queried =
        state->ops->pipeline_residency_ready(state->backend, native_ready);
    if (!queried.ok || !native_ready) {
      return queried;
    }
    ops = state->ops;
  }
  ready = true;
  return rund::AccelCheck{true, "ok"};
}

BackendUpload
UploadPreparedKernelPipeline(const PreparedKernelPipeline &prepared,
                             const void *const data,
                             const std::uint64_t bytes) noexcept {
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->upload_prepared_pipeline == nullptr) {
    return {};
  }
  return pipeline->ops->upload_prepared_pipeline(pipeline->backend, data,
                                                 bytes);
}

BackendDownload
DownloadPreparedKernelPipeline(const PreparedKernelPipeline &prepared,
                               void *const data, const std::uint64_t bytes,
                               std::uint64_t *const payload_hash) noexcept {
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->download_prepared_pipeline == nullptr) {
    return {};
  }
  return pipeline->ops->download_prepared_pipeline(pipeline->backend, data,
                                                   bytes, payload_hash);
}

PreparedPipelineMemory ReadPreparedKernelPipelineMemory(
    const PreparedKernelPipeline &prepared) noexcept {
  const auto *const pipeline =
      static_cast<const prepared::PipelineState *>(prepared.owner.get());
  return prepared.ok && pipeline != nullptr ? pipeline->memory.read()
                                            : PreparedPipelineMemory{};
}

} // namespace rund::node::accel::detail
