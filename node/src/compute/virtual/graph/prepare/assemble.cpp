#include "internal.hpp"
#include "../../../backend.hpp"

#include <rund/compute/pipeline/builder.hpp>
#include <rund/compute/pipeline/capacity.hpp>
#include <rund/compute/pipeline/runtime.hpp>

#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail::virtual_graph_prepare_detail {
namespace {

using VirtualResult = Result<std::shared_ptr<VirtualPipelineState>>;

[[nodiscard]] Result<std::shared_ptr<PipelineState>> prepare_graph_bank(
    const std::shared_ptr<ProgramState> &program,
    const std::shared_ptr<const residency::ResidencyPlan> &pages,
    const std::shared_ptr<residency::Pool> &pool,
    const std::uint64_t logical_bytes, const std::uint64_t resident_bytes,
    const std::uint32_t frame_count, const std::uint32_t bank,
    const std::uint32_t graph_stage,
    const PipelineResidencySemantic semantic = PipelineResidencySemantic::None,
    const std::span<const std::uint32_t> semantic_inputs = {},
    const std::uint32_t semantic_output = 0u) noexcept {
  auto build = make_pipeline(program == nullptr ? nullptr : program->device);
  if (build == nullptr) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  if (semantic == PipelineResidencySemantic::None) {
    append_pipeline_graph_residency(build, program, pages, pool, logical_bytes,
                                    resident_bytes, frame_count, bank,
                                    graph_stage);
  } else {
    append_pipeline_graph_semantic_residency(
        build, program, pages, pool, logical_bytes, resident_bytes, frame_count,
        bank, graph_stage, semantic, semantic_inputs, semantic_output);
  }
  auto prepared = prepare_pipeline(std::move(build));
  if (!prepared) {
    return Result<std::shared_ptr<PipelineState>>::fail(prepared.reason(),
                                                        prepared.location());
  }
  std::shared_ptr<PipelineState> pipeline = std::move(prepared).value();
  const residency::TiledGraphPlan &graph = pages->tiled_graph();
  const residency::TiledGraphStage *const stage =
      graph_stage < graph.stages().size() ? &graph.stages()[graph_stage]
                                          : nullptr;
  const auto first_resource = [&](const residency::Access access) noexcept
      -> const residency::TiledGraphResource * {
    if (stage == nullptr) {
      return nullptr;
    }
    const auto found =
        std::find_if(stage->ports.begin(), stage->ports.end(),
                     [access](const residency::TiledGraphPort port) {
                       return port.access == access;
                     });
    return found == stage->ports.end() ? nullptr
                                       : graph.resource(found->resource);
  };
  const residency::TiledGraphResource *input_resource =
      first_resource(residency::Access::Read);
  const residency::TiledGraphResource *output_resource =
      first_resource(residency::Access::Write);
  if (semantic != PipelineResidencySemantic::None) {
    const residency::TiledGraphPlan &residency_graph = pages->tiled_graph();
    input_resource = semantic_inputs.empty()
                         ? nullptr
                         : residency_graph.resource(semantic_inputs.front());
    output_resource = residency_graph.resource(semantic_output);
  }
  const std::uint64_t expected_input =
      input_resource == nullptr ? 0u : input_resource->page_bytes;
  const std::uint64_t expected_output =
      output_resource == nullptr ? 0u : output_resource->page_bytes;
  if (pipeline->residency != pages || pipeline->residency_pool != pool ||
      pipeline->residency_bank != bank ||
      pipeline->residency_stage != PipelineResidencyStage::Graph ||
      pipeline->residency_graph_stage != graph_stage ||
      pipeline->residency_input_page_bytes != expected_input ||
      pipeline->residency_output_page_bytes != expected_output ||
      pipeline->residency_semantic != semantic ||
      (semantic != PipelineResidencySemantic::None &&
       pipeline->residency_semantic_port_count !=
           semantic_inputs.size() + 1u) ||
      pipeline->residency_input == std::numeric_limits<std::uint32_t>::max() ||
      pipeline->residency_output == std::numeric_limits<std::uint32_t>::max() ||
      pipeline->residency_input == pipeline->residency_output ||
      pipeline->steps.size() != frame_count ||
      pipeline->logical_step_count != frame_count) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineInvalid);
  }
  if (pipeline->device->backend != Backend::Cpu) {
    if (pipeline->device->ops == nullptr ||
        pipeline->device->ops->residency.prepare_pipeline_residency ==
            nullptr) {
      return Result<std::shared_ptr<PipelineState>>::fail(
          Reason::BackendUnsupported);
    }
    const Status prepared =
        pipeline->device->ops->residency.prepare_pipeline_residency(*pipeline);
    if (!prepared) {
      return Result<std::shared_ptr<PipelineState>>::fail(prepared.reason());
    }
  }
  return Result<std::shared_ptr<PipelineState>>::success(std::move(pipeline));
}

} // namespace

Result<std::shared_ptr<VirtualPipelineState>>
assemble_graph(GraphPreparationDraft &draft) noexcept {
  using VirtualResult = Result<std::shared_ptr<VirtualPipelineState>>;
  try {
    std::shared_ptr<graph_reduce::CpuReceiptBook> cpu_receipts;
    std::shared_ptr<graph_reduce::CpuGraphQuarantine> cpu_quarantine_hold;
    if (draft.program->device->backend == Backend::Cpu) {
      cpu_receipts = std::make_shared<graph_reduce::CpuReceiptBook>();
      if (!cpu_receipts->valid()) {
        return VirtualResult::fail(Reason::PipelineCapacity);
      }
      cpu_quarantine_hold =
          std::make_shared<graph_reduce::CpuGraphQuarantine>(cpu_receipts);
    }
    draft.pages = std::make_shared<residency::ResidencyPlan>(
        std::move(draft.planned.plan));
    draft.pool = draft.program->device->residency->acquire(
        draft.program->device,
        residency::PoolLayout{
            .input_type = draft.input->type,
            .input_format = draft.input->format,
            .intermediate_type = draft.terminal_input.type,
            .intermediate_format = draft.terminal_input.format,
            .control_type = Type::U64,
            .control_format = {},
            .output_type = draft.output->type,
            .output_format = draft.output->format,
            .input_page_bytes = draft.input_page_bytes,
            .intermediate_page_bytes = draft.intermediate_page_bytes,
            .control_page_bytes = draft.control_page_bytes,
            .output_page_bytes = draft.output_page_bytes,
            .frame_capacity = static_cast<std::uint32_t>(draft.frames),
            .graph_host_input_count =
                static_cast<std::uint32_t>(draft.inputs.size()),
            .host_frame_capacity =
                static_cast<std::uint32_t>(draft.host_frames),
            .host_output_frame_capacity =
                static_cast<std::uint32_t>(draft.host_output_frames),
            .graph_host_service =
                draft.program->device->backend != Backend::Cpu,
        },
        draft.pages->tiled_graph().physical_classes());
    if (draft.pool == nullptr ||
        draft.pool->host_storage_bytes != draft.host_storage_bytes) {
      return VirtualResult::fail(Reason::PipelineMemoryBudget);
    }

    try {
      draft.graph_pipelines.reserve(draft.sliced.stages.size() *
                                    residency::Pool::BankCount);
    } catch (const std::bad_alloc &) {
      return VirtualResult::fail(Reason::PipelineCapacity);
    }
    for (std::size_t stage = 0u; stage < draft.sliced.stages.size(); ++stage) {
      for (std::uint32_t bank = 0u; bank < residency::Pool::BankCount; ++bank) {
        auto prepared = prepare_graph_bank(
            draft.sliced.stages[stage].program, draft.pages, draft.pool,
            draft.logical_bytes, draft.resident_bytes,
            static_cast<std::uint32_t>(draft.frames), bank,
            static_cast<std::uint32_t>(stage));
        if (!prepared) {
          return VirtualResult::fail(prepared.reason(), prepared.location());
        }
        draft.graph_pipelines.push_back(std::move(prepared).value());
      }
    }
    if (draft.program->device->backend != Backend::Cpu &&
        draft.graph_reduction && draft.sliced.service_free_prefix != nullptr) {
      auto prepared = prepare_graph_bank(
          draft.sliced.service_free_prefix, draft.pages, draft.pool,
          draft.logical_bytes, draft.resident_bytes,
          static_cast<std::uint32_t>(draft.frames), 0u, 0u,
          PipelineResidencySemantic::GraphReduction,
          {draft.sliced.input_resources.data(),
           draft.sliced.input_resources.size()},
          draft.sliced.stages.back().inputs.front());
      if (!prepared) {
        return VirtualResult::fail(prepared.reason(), prepared.location());
      }
      draft.device_vsm_semantic_pipeline = std::move(prepared).value();
    }

    auto state = std::make_shared<VirtualPipelineState>();
    std::copy(draft.inputs.begin(), draft.inputs.end(), state->inputs.begin());
    state->input_count = draft.inputs.size();
    std::copy(draft.sliced.input_resources.begin(),
              draft.sliced.input_resources.end(),
              state->graph_input_resources.begin());
    state->graph_input_resource_count = draft.sliced.input_resources.size();
    state->output = draft.output;
    state->graph_pipelines = std::move(draft.graph_pipelines);
    state->device_vsm_semantic_pipeline =
        std::move(draft.device_vsm_semantic_pipeline);
    state->cpu_receipts = std::move(cpu_receipts);
    state->cpu_quarantine_hold = std::move(cpu_quarantine_hold);
    state->pipeline = graph_stage_pipeline(*state, 0u, 0u);
    state->alternate_pipeline = graph_stage_pipeline(*state, 0u, 1u);
    if (state->cpu_quarantine_hold != nullptr &&
        !state->cpu_quarantine_hold->bind(state->pipeline.get(),
                                          draft.pool.get(),
                                          &draft.pool->authority())) {
      return VirtualResult::fail(Reason::PipelineCapacity);
    }
    state->geometry = draft.geometry;
    const std::uint64_t graph_identity =
        draft.program->graph_info.fingerprint.hi ^
        std::rotl(draft.program->graph_info.fingerprint.lo, 1);
    for (const std::shared_ptr<PipelineState> &stage : state->graph_pipelines) {
      const Status accumulated = accumulate_virtual_graph_stage(
          state->stats, pipeline_stats(stage), graph_identity);
      if (!accumulated) {
        return VirtualResult::fail(accumulated.reason());
      }
    }
    state->stats.pipeline.residency = ResidencyStats{
        .logical_bytes = draft.logical_bytes,
        .page_bytes = draft.pages->page_bytes(),
        .page_count = draft.page_count,
        .frame_capacity = draft.frames,
        .resident_frames_peak = 0u,
        .plan_identity_hi = draft.pages->identity().hi,
        .plan_identity_lo = draft.pages->identity().lo,
    };
    return VirtualResult::success(std::move(state));
  } catch (const std::bad_alloc &) {
    return VirtualResult::fail(Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail::virtual_graph_prepare_detail
