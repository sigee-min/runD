#include "local.hpp"

namespace rund::compute::detail {

using cpu_run_state_detail::add_count;

bool append_cpu_job_binding_slice(CpuPreparedArenaPlan &arena,
                                  const CpuJobBindingCounts &counts,
                                  CpuJobBindingSlice &slice) noexcept {
  if (arena.layout.sealed) {
    return false;
  }
  std::size_t owners = arena.buffer_owner_count;
  std::size_t views = arena.buffer_view_count;
  std::size_t kernel_views = arena.kernel_view_count;
  std::size_t view_transfers = arena.view_transfer_count;
  CpuJobBindingSlice next{
      .input_begin = owners,
      .input_count = counts.inputs,
  };
  if (!add_count(owners, counts.inputs)) {
    return false;
  }
  next.output_begin = owners;
  next.output_count = counts.outputs;
  if (!add_count(owners, counts.outputs)) {
    return false;
  }
  next.input_view_begin = views;
  next.input_view_count = counts.inputs;
  if (!add_count(views, counts.inputs)) {
    return false;
  }
  next.output_view_begin = views;
  next.output_view_count = counts.outputs;
  if (!add_count(views, counts.outputs)) {
    return false;
  }
  next.kernel_view_begin = kernel_views;
  next.kernel_view_count = counts.kernel_views;
  if (!add_count(kernel_views, counts.kernel_views)) {
    return false;
  }
  next.input_transfer_begin = view_transfers;
  next.input_transfer_count = counts.input_transfers;
  if (!add_count(view_transfers, counts.input_transfers)) {
    return false;
  }
  next.output_transfer_begin = view_transfers;
  next.output_transfer_count = counts.output_transfers;
  if (!add_count(view_transfers, counts.output_transfers)) {
    return false;
  }
  arena.buffer_owner_count = owners;
  arena.buffer_view_count = views;
  arena.kernel_view_count = kernel_views;
  arena.view_transfer_count = view_transfers;
  slice = next;
  return true;
}

bool append_cpu_workspace_slice(CpuPreparedArenaPlan &arena,
                                const std::size_t buffer_count,
                                CpuWorkspaceSlice &slice) noexcept {
  if (arena.layout.sealed) {
    return false;
  }
  const CpuWorkspaceSlice next{
      .workspace_begin = arena.workspace_count,
      .workspace_count = 1u,
      .buffer_begin = arena.buffer_owner_count,
      .buffer_count = buffer_count,
      .offset_begin = arena.workspace_offset_count,
      .offset_count = buffer_count,
  };
  std::size_t workspaces = arena.workspace_count;
  std::size_t buffers = arena.buffer_owner_count;
  std::size_t offsets = arena.workspace_offset_count;
  if (!add_count(workspaces, 1u) || !add_count(buffers, buffer_count) ||
      !add_count(offsets, buffer_count)) {
    return false;
  }
  arena.workspace_count = workspaces;
  arena.buffer_owner_count = buffers;
  arena.workspace_offset_count = offsets;
  slice = next;
  return true;
}

} // namespace rund::compute::detail
