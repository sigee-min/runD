#include "local.hpp"

#include "../../../backend.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/residency/integration.hpp"

#include <rund/compute/pipeline/builder.hpp>
#include <rund/compute/pipeline/runtime.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace rund::compute::detail::virtual_prepare_detail {

Result<std::shared_ptr<PipelineState>> prepare_virtual_bank(
    const std::shared_ptr<ProgramState> &program,
    const std::shared_ptr<ProgramState> &residency_program,
    const std::shared_ptr<residency::ResidencyPlan> &pages,
    const std::shared_ptr<residency::Pool> &pool,
    const std::uint64_t logical_bytes, const std::uint64_t input_frame_bytes,
    const std::uint64_t output_frame_bytes, const std::uint64_t resident_bytes,
    const std::uint32_t frames, const std::uint32_t bank) noexcept {
  auto build = make_pipeline(program->device);
  if (build == nullptr) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  append_pipeline_residency(build, residency_program, pages, pool,
                            logical_bytes, input_frame_bytes,
                            output_frame_bytes, resident_bytes,
                            static_cast<std::uint32_t>(frames), bank);
  auto physical = prepare_pipeline(std::move(build));
  if (!physical) {
    return Result<std::shared_ptr<PipelineState>>::fail(physical.reason(),
                                                        physical.location());
  }
  std::shared_ptr<PipelineState> pipeline = std::move(physical).value();
  if (pipeline->residency != pages || pipeline->residency_pool != pool ||
      (program->device->backend != Backend::Cpu &&
       pool->host_storage == nullptr) ||
      pipeline->residency_input_page_bytes != input_frame_bytes ||
      pipeline->residency_output_page_bytes != output_frame_bytes ||
      pipeline->residency_input == std::numeric_limits<std::uint32_t>::max() ||
      pipeline->residency_output == std::numeric_limits<std::uint32_t>::max() ||
      pipeline->residency_input == pipeline->residency_output ||
      pipeline->outputs.size() != 1u) {
    return Result<std::shared_ptr<PipelineState>>::fail(
        Reason::PipelineInvalid);
  }
  pipeline->residency_bank = bank;
  if (pipeline->device->backend != Backend::Cpu) {
    if (pipeline->device->ops == nullptr ||
        pipeline->device->ops->residency.prepare_residency_selection ==
            nullptr) {
      return Result<std::shared_ptr<PipelineState>>::fail(
          Reason::BackendUnsupported);
    }
    const Status prepared =
        pipeline->device->ops->residency.prepare_residency_selection(*pipeline);
    if (!prepared) {
      return Result<std::shared_ptr<PipelineState>>::fail(prepared.reason());
    }
  }
  return Result<std::shared_ptr<PipelineState>>::success(std::move(pipeline));
}

} // namespace rund::compute::detail::virtual_prepare_detail
