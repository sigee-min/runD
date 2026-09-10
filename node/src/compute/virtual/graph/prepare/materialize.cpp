#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <new>
#include <utility>

namespace rund::compute::detail::virtual_graph_prepare_detail {

Status materialize_graph_inputs(GraphPreparationDraft &draft) noexcept {
  try {
    std::size_t remap_copied = 0u;
    draft.graph_resources.reserve(draft.sliced.resources.size());
    for (const graph_compile::TiledGraphSliceResource &resource :
         draft.sliced.resources) {
      std::uint64_t page_bytes = 0u;
      std::uint64_t materialized_bytes = 0u;
      if (!kernel::checked::mul(resource.count, type_bytes(resource.type),
                                page_bytes) ||
          !kernel::checked::mul(draft.page_count, page_bytes,
                                materialized_bytes)) {
        return Status::fail(Reason::PipelineCapacity);
      }
      const bool external_input =
          resource.kind == graph_compile::SliceResourceKind::ExternalInput;
      const bool external_output =
          resource.kind == graph_compile::SliceResourceKind::ExternalOutput;
      const auto input_resource =
          std::find(draft.sliced.input_resources.begin(),
                    draft.sliced.input_resources.end(), resource.resource);
      const std::size_t input_index = static_cast<std::size_t>(
          std::distance(draft.sliced.input_resources.begin(), input_resource));
      if (external_input &&
          (input_resource == draft.sliced.input_resources.end() ||
           input_index >= draft.inputs.size())) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const std::uint64_t logical_resource_bytes =
          external_input ? draft.inputs[input_index]->bytes
          : external_output && draft.graph_pointwise
              ? draft.output->bytes
              : (resource.count == draft.geometry.input_frame_elements
                     ? draft.input->bytes
                     : materialized_bytes);
      if (!kernel::checked::add(draft.logical_bytes, logical_resource_bytes,
                                draft.logical_bytes)) {
        return Status::fail(Reason::PipelineCapacity);
      }
      std::vector<residency::GraphPageRemap> remaps;
      if (external_input && !draft.page_map.entries.empty()) {
        for (const GraphPageMapEntry entry : draft.page_map.entries) {
          if (draft.sliced.input_resources[entry.input] != resource.resource) {
            continue;
          }
          remaps.push_back(residency::GraphPageRemap{
              .source_local = entry.source_local,
              .target_local = entry.target_local,
              .source_origin = entry.origin == PageOrigin::Begin
                                   ? residency::GraphPageOrigin::Begin
                                   : residency::GraphPageOrigin::End});
          ++remap_copied;
        }
      }
      draft.graph_resources.push_back(residency::TiledGraphResourceInput{
          .resource = resource.resource,
          .type = resource.type,
          .format = resource.format,
          .page_bytes = page_bytes,
          .logical_bytes = external_input ? draft.inputs[input_index]->bytes
                           : external_output && draft.graph_pointwise
                               ? draft.output->bytes
                               : materialized_bytes,
          .kind = external_input ? residency::GraphResourceKind::ExternalInput
                  : external_output
                      ? residency::GraphResourceKind::ExternalOutput
                      : residency::GraphResourceKind::Internal,
          .persistence =
              external_input || (external_output && draft.graph_pointwise)
                  ? residency::ResourcePersistence::Backing
                  : residency::ResourcePersistence::Transient,
          .remaps = std::move(remaps),
      });
    }
    if (remap_copied != draft.page_map.entries.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    draft.graph_stages.reserve(draft.sliced.stages.size());
    for (const graph_compile::TiledGraphStageSlice &stage :
         draft.sliced.stages) {
      residency::TiledGraphStageInput planned_stage{
          .node = stage.node,
          .domain = stage.tile_partial ? residency::StageDomain::TilePartial
                                       : residency::StageDomain::Tile,
      };
      planned_stage.ports.reserve(stage.inputs.size() + stage.outputs.size());
      for (const std::uint32_t resource : stage.inputs) {
        planned_stage.ports.push_back(residency::TiledGraphPortInput{
            .resource = resource, .access = residency::Access::Read});
      }
      for (const std::uint32_t resource : stage.outputs) {
        planned_stage.ports.push_back(residency::TiledGraphPortInput{
            .resource = resource, .access = residency::Access::Write});
      }
      draft.graph_stages.push_back(std::move(planned_stage));
    }
    for (const residency::TiledGraphResourceInput &resource :
         draft.graph_resources) {
      if (resource.remaps.empty()) {
        continue;
      }
      bool read = false;
      for (const residency::TiledGraphStageInput &stage : draft.graph_stages) {
        for (const residency::TiledGraphPortInput port : stage.ports) {
          if (port.resource != resource.resource) {
            continue;
          }
          read = true;
          if (port.access != residency::Access::Read) {
            return Status::fail(Reason::PipelineInvalid);
          }
        }
      }
      if (resource.kind != residency::GraphResourceKind::ExternalInput ||
          resource.persistence != residency::ResourcePersistence::Backing ||
          !read) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
  return Status::success();
}

} // namespace rund::compute::detail::virtual_graph_prepare_detail
