#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::virtual_graph_prepare_detail {

bool valid_graph_map_shape(
    const std::span<const residency::GraphPageRemap> remaps,
    const std::size_t frame_count, const std::size_t page_count) noexcept {
  if (frame_count == 0u || frame_count > PipelineLeafCapacity ||
      remaps.size() != frame_count ||
      remaps.size() > residency::GraphPageRemapCapacity) {
    return false;
  }
  std::array<bool, PipelineLeafCapacity> sources{};
  std::array<bool, PipelineLeafCapacity> targets{};
  for (const residency::GraphPageRemap remap : remaps) {
    if (remap.source_local >= frame_count ||
        remap.target_local >= frame_count ||
        static_cast<std::uint8_t>(remap.source_origin) >
            static_cast<std::uint8_t>(residency::GraphPageOrigin::End) ||
        sources[remap.source_local] || targets[remap.target_local]) {
      return false;
    }
    sources[remap.source_local] = true;
    targets[remap.target_local] = true;
  }
  const std::size_t tail = page_count % frame_count;
  if (tail == 0u) {
    return true;
  }
  std::array<bool, PipelineLeafCapacity> used{};
  for (std::size_t target = 0u; target < tail; ++target) {
    const auto found =
        std::find_if(remaps.begin(), remaps.end(),
                     [target](const residency::GraphPageRemap remap) {
                       return remap.target_local == target;
                     });
    if (found == remaps.end() || found->source_local >= tail) {
      return false;
    }
    const std::size_t source =
        found->source_origin == residency::GraphPageOrigin::Begin
            ? found->source_local
            : tail - 1u - found->source_local;
    if (source >= tail || used[source]) {
      return false;
    }
    used[source] = true;
  }
  return true;
}

Status validate_graph_request(GraphPreparationDraft &draft) noexcept {
  draft.input = draft.inputs.empty() ? nullptr : draft.inputs.front().get();
  draft.graph_reduction = draft.geometry.route == VirtualRoute::GraphReduction;
  draft.graph_pointwise = draft.geometry.route == VirtualRoute::GraphPointwise;
  if (draft.program == nullptr || draft.program->device == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!draft.page_map.entries.empty()) {
    if (!draft.graph_pointwise ||
        draft.page_map.graph_hi != draft.program->graph_info.fingerprint.hi ||
        draft.page_map.graph_lo != draft.program->graph_info.fingerprint.lo ||
        draft.page_map.entries.size() > residency::GraphPageRemapCapacity) {
      return Status::fail(Reason::PipelineInvalid);
    }
    for (const GraphPageMapEntry entry : draft.page_map.entries) {
      if (entry.input >= draft.inputs.size() ||
          (entry.origin != PageOrigin::Begin &&
           entry.origin != PageOrigin::End)) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
  } else if (draft.page_map.graph_hi != 0u || draft.page_map.graph_lo != 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }

  const auto operation =
      static_cast<kernel::ReduceOp>(draft.geometry.operation);
  const bool operation_supported =
      draft.graph_pointwise || operation == kernel::ReduceOp::Sum ||
      operation == kernel::ReduceOp::CountNonzero ||
      operation == kernel::ReduceOp::Min || operation == kernel::ReduceOp::Max;
  if (draft.program == nullptr || draft.program->device == nullptr ||
      draft.input == nullptr || draft.inputs.empty() ||
      draft.inputs.size() > VirtualPipelineState::InputCapacity ||
      (draft.program->device->backend == Backend::Cpu &&
       !draft.graph_pointwise && draft.inputs.size() != 1u) ||
      draft.output == nullptr || draft.program->device->residency == nullptr ||
      (!draft.graph_reduction && !draft.graph_pointwise) ||
      !operation_supported || draft.geometry.input_frame_elements == 0u ||
      draft.geometry.intermediate_frame_elements !=
          draft.geometry.input_frame_elements ||
      (draft.graph_reduction ? draft.geometry.output_frame_elements != 1u
                             : draft.geometry.output_frame_elements !=
                                   draft.geometry.input_frame_elements) ||
      (draft.graph_reduction && draft.input->type != Type::U64) ||
      (draft.graph_reduction && draft.output->type != Type::U64) ||
      (draft.graph_pointwise && draft.output->type != draft.input->type) ||
      (draft.graph_pointwise && draft.input->type != Type::U32 &&
       draft.input->type != Type::U64) ||
      (draft.graph_pointwise && draft.output->count != draft.input->count) ||
      draft.input->element_bytes != type_bytes(draft.input->type) ||
      draft.output->element_bytes != type_bytes(draft.output->type) ||
      std::any_of(draft.inputs.begin(), draft.inputs.end(),
                  [input = draft.input](const auto &candidate) {
                    return candidate == nullptr ||
                           candidate->type != input->type ||
                           candidate->format != input->format ||
                           candidate->count != input->count ||
                           candidate->element_bytes != input->element_bytes ||
                           candidate->bytes != input->bytes;
                  })) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  return Status::success();
}

} // namespace rund::compute::detail::virtual_graph_prepare_detail
