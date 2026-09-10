#include "internal.hpp"

#include "../../buffer/local.hpp"
#include "../../pipeline/local.hpp"
#include "../../type.hpp"

#include <rund/compute/pipeline/builder.hpp>

#include <array>
#include <new>
#include <vector>

namespace rund::compute::detail::virtual_multi_detail {

Result<std::shared_ptr<PipelineState>>
prepare_pipeline(const std::shared_ptr<ProgramState> &semantic) noexcept {
  if (semantic == nullptr || semantic->device == nullptr ||
      semantic->input_types.empty() || semantic->output_types.size() != 1u) {
    return Result<std::shared_ptr<PipelineState>>::fail(Reason::ProgramInvalid);
  }
  try {
    std::vector<ResourceView> inputs;
    inputs.reserve(semantic->input_types.size());
    for (std::size_t index = 0u; index < semantic->input_types.size();
         ++index) {
      auto buffer = make_input_binding_buffer(semantic->device,
                                              semantic->input_types[index],
                                              semantic->input_sizes[index]);
      if (!buffer) {
        return Result<std::shared_ptr<PipelineState>>::fail(buffer.reason());
      }
      inputs.push_back(ResourceView{
          .buffer = std::move(buffer).value(),
          .type = semantic->input_types[index],
          .format = semantic->input_formats[index],
          .count = semantic->input_sizes[index],
          .element_bytes = type_bytes(semantic->input_types[index]),
          .alignment = type_bytes(semantic->input_types[index]),
          .access = ResourceAccess::Read,
      });
    }
    auto output = make_input_binding_buffer(semantic->device,
                                            semantic->output_types.front(),
                                            semantic->output_sizes.front());
    if (!output) {
      return Result<std::shared_ptr<PipelineState>>::fail(output.reason());
    }
    const std::array outputs{ResourceView{
        .buffer = std::move(output).value(),
        .type = semantic->output_types.front(),
        .format = semantic->output_formats.front(),
        .count = semantic->output_sizes.front(),
        .element_bytes = type_bytes(semantic->output_types.front()),
        .alignment = type_bytes(semantic->output_types.front()),
        .access = ResourceAccess::Write,
    }};
    auto build = make_pipeline(semantic->device);
    append_pipeline(build, semantic, inputs, outputs);
    auto prepared = ::rund::compute::detail::prepare_pipeline(std::move(build));
    if (!prepared) {
      return Result<std::shared_ptr<PipelineState>>::fail(prepared.reason(),
                                                          prepared.location());
    }
    std::shared_ptr<PipelineState> pipeline = std::move(prepared).value();
    if (!valid_pipeline(pipeline) || pipeline->transactional ||
        pipeline->residency != nullptr || pipeline->residency_pool != nullptr ||
        pipeline->steps.size() != 1u || pipeline->logical_step_count != 1u ||
        pipeline->outputs.size() != 1u || !pipeline->prepared.ok) {
      return Result<std::shared_ptr<PipelineState>>::fail(
          Reason::PipelineInvalid);
    }
    return Result<std::shared_ptr<PipelineState>>::success(std::move(pipeline));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail::virtual_multi_detail
