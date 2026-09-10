#include "local.hpp"

namespace rund_node_test_pipeline::view {

[[nodiscard]] std::size_t
CpuViewTransferCount(const rund::compute::detail::JobState &job) noexcept {
  return job.cpu_view_inputs.size() + job.cpu_view_outputs.size();
}

[[nodiscard]] const rund::compute::detail::BufferState *
FirstCpuViewBuffer(const rund::compute::detail::JobState &job) noexcept {
  if (!job.cpu_view_inputs.empty()) {
    const auto &transfer = job.cpu_view_inputs.front();
    return transfer.binding < job.inputs.size()
               ? job.inputs[transfer.binding].get()
               : nullptr;
  }
  if (!job.cpu_view_outputs.empty()) {
    const auto &transfer = job.cpu_view_outputs.front();
    return transfer.binding < job.outputs.size()
               ? job.outputs[transfer.binding].get()
               : nullptr;
  }
  return nullptr;
}

} // namespace rund_node_test_pipeline::view
