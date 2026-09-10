#include "internal.hpp"

#include <memory>
#include <new>
#include <utility>

namespace rund::compute::detail {

Result<std::shared_ptr<CpuPreparedArena>>
make_cpu_prepared_arena(const CpuPreparedArenaPlan &plan) {
  CpuPreparedArenaPlan expected{};
  expected.execution = plan.execution;
  expected.map_count = plan.map_count;
  expected.read_count = plan.read_count;
  expected.write_count = plan.write_count;
  expected.buffer_owner_count = plan.buffer_owner_count;
  expected.buffer_view_count = plan.buffer_view_count;
  expected.kernel_view_count = plan.kernel_view_count;
  expected.view_transfer_count = plan.view_transfer_count;
  expected.workspace_count = plan.workspace_count;
  expected.workspace_offset_count = plan.workspace_offset_count;
  if (!plan.layout.sealed ||
      !seal_cpu_prepared_arena_plan(expected, plan.layout.page_bytes) ||
      expected != plan) {
    return Result<std::shared_ptr<CpuPreparedArena>>::fail(
        Reason::CpuRuntimeInvalid);
  }
  try {
    auto arena = std::make_shared<CpuPreparedArena>();
    if (!arena->materialize(plan) || !arena->supports(plan.execution) ||
        arena->extent_bytes() != plan.layout.extent_bytes ||
        arena->committed_bytes() != plan.layout.committed_bytes) {
      return Result<std::shared_ptr<CpuPreparedArena>>::fail(
          Reason::BufferCapacity);
    }
    return Result<std::shared_ptr<CpuPreparedArena>>::success(std::move(arena));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<CpuPreparedArena>>::fail(
        Reason::BufferCapacity);
  }
}

} // namespace rund::compute::detail
