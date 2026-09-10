#include "../../state/program.hpp"
#include "local.hpp"

#include "../../../program/state.hpp"
#include "../../../status.hpp"
#include "../../bounded.hpp"
#include "../../view.hpp"
#include "../state.hpp"

namespace rund::compute::detail {

namespace {

[[nodiscard]] RawCpuBuffer raw_cpu_buffer(BufferState *const buffer,
                                          const JobBufferView view) noexcept {
  const std::optional<CpuView> bound = cpu_view(buffer, view);
  return !bound || bound->data == nullptr || !bound->footprint.dense()
             ? RawCpuBuffer{}
             : RawCpuBuffer{.data = bound->data,
                            .bytes = bound->footprint.bytes};
}

} // namespace

Status bind_cpu_primitive_ports(JobState &job, CpuGraphProgram &cpu,
                                CpuGraphRun &run, const std::size_t step,
                                std::span<RawCpuBuffer> ports) noexcept {
  const std::shared_ptr<ProgramState> &program = job.program;
  if (program == nullptr || ports.empty() || step >= cpu.bind_begin.size() ||
      step >= cpu.bind_count.size()) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  const std::size_t begin = cpu.bind_begin[step];
  const std::size_t count = cpu.bind_count[step];
  if (count == 0u || count > ports.size() ||
      begin > program->graph_bindings.size() ||
      count > program->graph_bindings.size() - begin) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  for (std::size_t index = 0u; index < count; ++index) {
    const GraphRunBinding &binding = program->graph_bindings[begin + index];
    BufferState *const buffer = graph_binding_buffer(
        *program, binding, job.inputs, job.outputs, run.buffers);
    if (buffer == nullptr) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    ports[index] =
        raw_cpu_buffer(buffer, job_binding_view(job, binding, *buffer));
    if (!ports[index]) {
      return Status::fail(Reason::CpuBufferInvalid);
    }
  }
  return Status::success();
}

Status read_cpu_primitive_count(const PrimitiveContext &context,
                                const std::uint32_t value,
                                const std::size_t capacity,
                                std::uint32_t &count) noexcept {
  const std::shared_ptr<ProgramState> &program = context.job.program;
  if (program == nullptr || context.cpu.runtime == nullptr || value == 0u ||
      value > context.cpu.runtime->values.size()) {
    return Status::fail(Reason::BoundedCountInvalid);
  }
  BufferState *const buffer =
      graph_value_buffer_id(*program, value, context.job.inputs,
                            context.job.outputs, context.run.buffers);
  const JobBufferView view = buffer == nullptr
                                 ? JobBufferView{}
                                 : job_value_view(context.job, value, *buffer);
  kernel::u32 logical_count = 0u;
  const Status result = read_bounded_count(
      buffer, view, context.cpu.runtime->values[value - 1u].type, capacity,
      logical_count);
  count = static_cast<std::uint32_t>(logical_count);
  return result;
}

Status finish_cpu_primitive(const bool ok,
                            const std::string_view reason) noexcept {
  return ok ? Status::success()
            : Status::fail(
                  project_reason(reason, Reason::PrimitiveBackendFailed));
}

} // namespace rund::compute::detail
