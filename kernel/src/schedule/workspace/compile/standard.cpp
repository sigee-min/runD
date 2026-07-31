#include "local.hpp"

#include <kernel/internal/schedule/builder.hpp>

namespace rund::kernel::workspace_compile_detail {
namespace {

bool ShouldMaterializePacketHints(const ScheduleCompileRequest& request) {
  return request.placement == PlacementPolicy::ContiguousBalanced &&
         request.packet_work_units.empty() &&
         request.packet_hints.size() == static_cast<std::size_t>(request.packet_count);
}

} // namespace

PartitionBuild CompileStandardSchedule(Workspace& workspace,
                                       const ScheduleCompileRequest& request,
                                       const PartitionProjection& projection) {
  ScheduleCompileRequest compile_request = request;
  if (!ShouldMaterializePacketHints(request)) {
    return rund::kernel::internal::CompileSchedule(workspace.schedule, request);
  }
  if (request.allocation == AllocationPolicy::NoGrowth &&
      workspace.packet_work_units.capacity() < static_cast<std::size_t>(request.packet_count)) {
    return BuildProjectedFailure(workspace, request, projection, "packet_work_unit_capacity_exceeded");
  }
  if (request.allocation == AllocationPolicy::AllowGrowth) {
    workspace.packet_work_units.reserve(request.packet_count);
  }
  workspace.packet_work_units.resize(request.packet_count);
  for (u32 packet = 0u; packet < request.packet_count; ++packet) {
    workspace.packet_work_units[packet] = workspace_detail::PacketWorkUnitsAt(request, packet);
  }
  compile_request.packet_work_units = ViewPacketWorkUnits(workspace);
  return rund::kernel::internal::CompileSchedule(workspace.schedule, compile_request);
}

} // namespace rund::kernel::workspace_compile_detail
