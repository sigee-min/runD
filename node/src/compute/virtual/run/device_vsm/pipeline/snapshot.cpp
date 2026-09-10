#include "../internal.hpp"

#include <algorithm>
#include <mutex>

namespace rund::compute::detail::device_vsm_product_detail {

Status snapshot_pipelines(
    const std::span<const std::shared_ptr<PipelineState>> pipelines,
    const std::span<const std::uint32_t> stages, const bool graph,
    DeviceVsmPipelineSnapshot &snapshot) noexcept {
  snapshot = {};
  if (pipelines.size() < 2u || pipelines.size() > DeviceVsmPipelineCapacity ||
      pipelines.size() != stages.size()) {
    return Status::fail(Reason::BackendUnsupported);
  }
  if (!graph) {
    const bool shared = pipelines[0u] != nullptr &&
                        pipelines[0u]->residency == nullptr &&
                        pipelines[0u]->residency_pool == nullptr;
    if (!shared) {
      if (pipelines.size() != residency::execution::BankCapacity) {
        return Status::fail(Reason::BackendUnsupported);
      }
      std::array<std::shared_ptr<PipelineState>,
                 residency::execution::BankCapacity>
          pair{pipelines[0u], pipelines[1u]};
      PipelineExecutionSnapshot pair_snapshot{};
      const Status captured = snapshot_pipeline_execution(pair, pair_snapshot);
      if (!captured) {
        return captured;
      }
      snapshot.generation[0u] = pair_snapshot.generation[0u];
      snapshot.generation[1u] = pair_snapshot.generation[1u];
      snapshot.parity[0u] = pair_snapshot.parity[0u];
      snapshot.parity[1u] = pair_snapshot.parity[1u];
      snapshot.count = pipelines.size();
      return Status::success();
    }
    if (pipelines.size() != residency::execution::BankCapacity ||
        pipelines[1u] == nullptr || pipelines[0u] == pipelines[1u] ||
        pipelines[0u]->device != pipelines[1u]->device) {
      return Status::fail(Reason::BackendUnsupported);
    }
    for (std::size_t index = 0u; index < pipelines.size(); ++index) {
      const std::shared_ptr<PipelineState> &pipeline = pipelines[index];
      if (!valid_pipeline(pipeline) || pipeline->device == nullptr ||
          pipeline->device->backend == Backend::Cpu ||
          pipeline->device->ops == nullptr || pipeline->transactional ||
          pipeline->residency != nullptr ||
          pipeline->residency_pool != nullptr || pipeline->steps.size() != 1u ||
          pipeline->logical_step_count != 1u ||
          pipeline->outputs.size() != 1u || pipeline->publication == nullptr) {
        return Status::fail(Reason::BackendUnsupported);
      }
      std::lock_guard pipeline_lock{pipeline->gate};
      std::lock_guard publication_lock{pipeline->publication->gate};
      if (pipeline->phase != PipelinePhase::Ready ||
          pipeline->publication->attempt_active ||
          pipeline->publication->device_lost) {
        return Status::fail(pipeline->publication->device_lost
                                ? Reason::DeviceLost
                                : Reason::PipelineBusy);
      }
      snapshot.generation[index] = pipeline->publication->generation;
      snapshot.parity[index] = pipeline->publication->parity;
    }
    snapshot.count = pipelines.size();
    return Status::success();
  }
  if (pipelines[0u] == nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  for (std::size_t index = 0u; index < pipelines.size(); ++index) {
    const std::shared_ptr<PipelineState> &pipeline = pipelines[index];
    if (pipeline == nullptr ||
        std::find(pipelines.begin(), pipelines.begin() + index, pipeline) !=
            pipelines.begin() + index ||
        pipeline->device != pipelines[0u]->device ||
        pipeline->residency != pipelines[0u]->residency ||
        pipeline->residency_pool != pipelines[0u]->residency_pool ||
        !valid_pipeline(pipeline) || pipeline->device == nullptr ||
        pipeline->device->backend == Backend::Cpu ||
        pipeline->device->ops == nullptr || pipeline->residency_bank != 0u ||
        pipeline->residency_stage != PipelineResidencyStage::Graph ||
        pipeline->residency_graph_stage != stages[index] ||
        pipeline->publication == nullptr) {
      return Status::fail(Reason::BackendUnsupported);
    }
    std::lock_guard pipeline_lock{pipeline->gate};
    std::lock_guard publication_lock{pipeline->publication->gate};
    if (pipeline->phase != PipelinePhase::Ready ||
        pipeline->publication->attempt_active ||
        pipeline->publication->device_lost) {
      return Status::fail(pipeline->publication->device_lost
                              ? Reason::DeviceLost
                              : Reason::PipelineBusy);
    }
    snapshot.generation[index] = pipeline->publication->generation;
    snapshot.parity[index] = pipeline->publication->parity;
  }
  snapshot.count = pipelines.size();
  return Status::success();
}

} // namespace rund::compute::detail::device_vsm_product_detail
