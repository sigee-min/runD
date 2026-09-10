#include "internal.hpp"

#include "../../pipeline/local.hpp"
#include "../stats.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/runtime.hpp>

#include <algorithm>
#include <new>

namespace rund::compute::detail {

Result<std::shared_ptr<VirtualPipelineState>> prepare_virtual_multi_device_vsm(
    const std::shared_ptr<ProgramState> &program,
    const std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output,
    const ResidencyConfig config, const VirtualGeometry &geometry) noexcept {
  using namespace virtual_multi_detail;
  Schema schema{};
  const Status inspected = inspect(program, inputs, output, geometry, schema);
  if (!inspected) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        inspected.reason());
  }
  if ((config.device_resident_bytes != 0u &&
       config.device_resident_bytes < schema.logical_bytes) ||
      (config.host_resident_bytes != 0u &&
       config.host_resident_bytes < schema.logical_bytes)) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineMemoryBudget);
  }
  auto primary = prepare_pipeline(schema.semantic);
  auto alternate = primary ? prepare_pipeline(schema.semantic)
                           : Result<std::shared_ptr<PipelineState>>::fail(
                                 primary.reason(), primary.location());
  if (!primary || !alternate) {
    const auto &failed = primary ? alternate : primary;
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        failed.reason(), failed.location());
  }
  try {
    auto state = std::make_shared<VirtualPipelineState>();
    std::copy(inputs.begin(), inputs.end(), state->inputs.begin());
    state->input_count = inputs.size();
    state->output = output;
    state->pipeline = std::move(primary).value();
    state->alternate_pipeline = std::move(alternate).value();
    state->geometry = schema.geometry;
    state->stats = pipeline_stats(state->pipeline);
    const Status accumulated = accumulate_virtual_epoch(
        state->stats, pipeline_stats(state->alternate_pipeline));
    std::uint64_t plan_page_bytes = 0u;
    if (!accumulated ||
        !kernel::checked::mul(schema.page_bytes, inputs.size() + 1u,
                              plan_page_bytes)) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          accumulated ? Reason::PipelineCapacity : accumulated.reason());
    }
    state->stats.pipeline.residency = ResidencyStats{
        .logical_bytes = schema.logical_bytes,
        .page_bytes = plan_page_bytes,
        .page_count = schema.page_count,
        .frame_capacity = schema.geometry.route == VirtualRoute::Scan ? 2u : 1u,
        .resident_frames_peak = 0u,
        .plan_identity_hi = schema.geometry.materialization_hi,
        .plan_identity_lo = schema.geometry.materialization_lo,
    };
    return Result<std::shared_ptr<VirtualPipelineState>>::success(
        std::move(state));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail
