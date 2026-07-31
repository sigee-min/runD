#include "local.hpp"
#include "../work/units.hpp"

namespace rund::kernel::workspace_detail {

bool HasPacketHints(const ScheduleCompileRequest& request) {
  return !request.packet_hints.empty();
}

bool PacketHintsValid(const ScheduleCompileRequest& request) {
  return request.packet_hints.empty() ||
         request.packet_hints.size() == static_cast<std::size_t>(request.packet_count);
}

bool HasWorkUnits(const ScheduleCompileRequest& request) {
  return request.packet_work_units.size() == static_cast<std::size_t>(request.packet_count) ||
         request.packet_hints.size() == static_cast<std::size_t>(request.packet_count);
}

u64 PacketWorkUnitsAt(const ScheduleCompileRequest& request, const u32 packet) {
  if (request.packet_work_units.size() == static_cast<std::size_t>(request.packet_count)) {
    return schedule_detail::NormalizeWorkUnits(request.packet_work_units[packet]);
  }
  if (request.packet_hints.size() == static_cast<std::size_t>(request.packet_count)) {
    return schedule_detail::NormalizeWorkUnits(request.packet_hints[packet].work_units);
  }
  return 1u;
}

} // namespace rund::kernel::workspace_detail
