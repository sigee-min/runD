#include "../../backend.hpp"
#include "../../host.hpp"
#include "../../program/state.hpp"
#include "../../size.hpp"
#include "../../type.hpp"
#include "../graph.hpp"
#include "../view.hpp"
#include "state.hpp"

#include <rund/compute/resource/plan.hpp>

#include <cstring>

namespace rund::compute::detail {

namespace {

[[nodiscard]] Result<bool> overlaps(const BufferState &buffer,
                                    const JobBufferView left,
                                    const JobBufferView right) noexcept {
  if (left.count == 0u || right.count == 0u) {
    return Result<bool>::success(false);
  }
  std::size_t left_offset = 0u;
  std::size_t left_stride = 0u;
  std::size_t right_offset = 0u;
  std::size_t right_stride = 0u;
  if (left.element_bytes == 0u || right.element_bytes == 0u ||
      !size::multiply(left.offset, left.element_bytes, left_offset) ||
      !size::multiply(left.stride, left.element_bytes, left_stride) ||
      !size::multiply(right.offset, right.element_bytes, right_offset) ||
      !size::multiply(right.stride, right.element_bytes, right_stride)) {
    return Result<bool>::fail(Reason::ResourceRangeCapacity);
  }
  const resource::Resource owner{
      .id = 1u,
      .bytes = buffer.bytes,
      .alias_group = 1u,
  };
  const resource::Access a{
      .resource = 1u,
      .offset_bytes = left_offset,
      .element_bytes = left.element_bytes,
      .element_count = left.count,
      .stride_bytes = left_stride,
  };
  const resource::Access b{
      .resource = 1u,
      .offset_bytes = right_offset,
      .element_bytes = right.element_bytes,
      .element_count = right.count,
      .stride_bytes = right_stride,
  };
  return resource::intersects(owner, a, owner, b);
}

[[nodiscard]] Status reset_cpu_route(JobState &job,
                                     const ResetRoute &reset) noexcept {
  if (job.program == nullptr || job.cpu == nullptr ||
      job.cpu->graph == nullptr ||
      reset.value_index >= job.program->graph_value_routes.size()) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  BufferState *const buffer =
      graph_value_buffer(*job.program, reset.value_index, job.inputs,
                         job.outputs, job.cpu->graph->buffers);
  if (buffer == nullptr) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  const JobBufferView view =
      job_value_view(job, reset.value_index + 1u, *buffer);
  for (std::size_t index = 0u; index < job.inputs.size(); ++index) {
    if (job.inputs[index].get() != buffer) {
      continue;
    }
    const JobBufferView input = index < job.input_views.size()
                                    ? job.input_views[index]
                                    : job_whole_view(*job.inputs[index]);
    auto overlap = overlaps(*buffer, input, view);
    if (!overlap) {
      return Status::fail(overlap.reason());
    }
    if (*overlap) {
      return Status::fail(Reason::BindingAliasUnsupported);
    }
  }
  const std::optional<CpuView> target = cpu_view(buffer, view);
  if (!target) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  const CpuFootprint footprint = target->footprint;
  if (footprint.bytes != 0u) {
    if (footprint.dense()) {
      std::memset(target->data, 0, footprint.bytes);
    } else {
      std::byte *element = target->data;
      for (std::size_t remaining = footprint.count; remaining > 1u;
           --remaining) {
        std::memset(element, 0, footprint.width);
        element += footprint.stride;
      }
      std::memset(element, 0, footprint.width);
    }
  }
  ::rund::detail::counter::Accumulate(job.cpu->stats.reset_bytes,
                                      footprint.bytes);
  ::rund::detail::counter::Accumulate(job.cpu->stats.reset_commands, 1u);
  return Status::success();
}

} // namespace

Status reset_cpu(JobState &job, const std::size_t step) noexcept {
  if (job.program == nullptr || job.cpu == nullptr) {
    return Status::fail(Reason::ProgramInvalid);
  }
  std::size_t &cursor = job.cpu->reset;
  if (job.program->cpu_graph == nullptr) {
    return Status::fail(Reason::CpuProgramInvalid);
  }
  const std::vector<ResetRoute> &resets = job.program->cpu_graph->resets;
  if (cursor < resets.size() && resets[cursor].step < step) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  while (cursor < resets.size() && resets[cursor].step == step) {
    const ResetRoute &reset = resets[cursor];
    const Status status = reset_cpu_route(job, reset);
    if (!status) {
      return status;
    }
    ++cursor;
  }
  return Status::success();
}

} // namespace rund::compute::detail
