#include "cases.hpp"

#include "test/assert.hpp"

#include <kernel/schedule/workspace.hpp>

#include <span>

int RunWorkspacePacketWorkUnitsContract() {
  rund::kernel::Workspace workspace{};
  TEST_ASSERT(rund::kernel::ReservePacketWorkUnits(workspace, 3u));
  TEST_ASSERT(rund::kernel::AppendPacketWorkUnit(workspace, 3u));
  TEST_ASSERT(rund::kernel::AppendPacketWorkUnit(workspace, 5u));
  const std::span<const rund::kernel::u64> packet_work_units = rund::kernel::ViewPacketWorkUnits(workspace);
  TEST_ASSERT(packet_work_units.size() == 2u);
  TEST_ASSERT(packet_work_units[0u] == 3u);
  TEST_ASSERT(packet_work_units[1u] == 5u);
  rund::kernel::ClearPacketWorkUnits(workspace);
  TEST_ASSERT(rund::kernel::ViewPacketWorkUnits(workspace).empty());
  return 0;
}
