#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>

namespace rund::compute::detail::virtual_graph_prepare_detail {

Status calculate_graph_bytes(GraphPreparationDraft &draft) noexcept {
  const std::uint64_t page_elements = draft.geometry.input_payload_elements;
  draft.page_count =
      draft.input->count / page_elements +
      static_cast<std::uint64_t>(draft.input->count % page_elements != 0u);
  draft.control_page_bytes = sizeof(std::uint64_t);
  if (draft.page_count == 0u ||
      !kernel::checked::mul(draft.geometry.input_frame_elements,
                            draft.input->element_bytes,
                            draft.input_page_bytes) ||
      !kernel::checked::mul(draft.geometry.intermediate_frame_elements,
                            draft.input->element_bytes,
                            draft.intermediate_page_bytes) ||
      !kernel::checked::mul(draft.geometry.output_frame_elements,
                            draft.output->element_bytes,
                            draft.output_page_bytes) ||
      !kernel::checked::mul(draft.input_page_bytes, draft.inputs.size(),
                            draft.host_input_page_bytes) ||
      !kernel::checked::add(draft.host_input_page_bytes,
                            draft.output_page_bytes, draft.host_frame_bytes) ||
      !kernel::checked::mul(draft.host_frame_bytes, residency::Pool::BankCount,
                            draft.host_frame_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }

  draft.prefetch_distance =
      draft.program->device->backend != Backend::Cpu &&
              std::all_of(draft.inputs.begin(), draft.inputs.end(),
                          [](const auto &value) {
                            return value->backing->tier() ==
                                       VirtualBackingTier::Persistent &&
                                   value->backing->max_parallel_reads() >= 2u;
                          })
          ? 2u
          : 1u;
  return Status::success();
}

} // namespace rund::compute::detail::virtual_graph_prepare_detail
