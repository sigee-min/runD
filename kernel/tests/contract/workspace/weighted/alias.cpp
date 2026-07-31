#include "local.hpp"

namespace workspace_weighted_contract {

int Alias() {
  const std::vector<rund::kernel::u64> zero_units{0u, 2u, 0u, 1u};
  const rund::kernel::ScheduleCompileRequest zero_request = Request(
      4u, 2u, rund::kernel::AllocationPolicy::NoGrowth,
      std::span<const rund::kernel::u64>(zero_units.data(), zero_units.size()));

  rund::kernel::Workspace alias_workspace{};
  TEST_ASSERT(Reserve(alias_workspace, zero_request) == 0);
  alias_workspace.packet_work_units.resize(4u);
  alias_workspace.packet_work_units[0u] = 4u;
  alias_workspace.packet_work_units[1u] = 0u;
  alias_workspace.packet_work_units[2u] = 3u;
  alias_workspace.packet_work_units[3u] = 2u;
  const rund::kernel::u64 *const alias_data =
      alias_workspace.packet_work_units.data();
  rund::kernel::ScheduleCompileRequest alias_request = zero_request;
  alias_request.packet_work_units =
      std::span<const rund::kernel::u64>(alias_data, 4u);
  const rund::kernel::PartitionBuild alias_build =
      Compile(alias_workspace, alias_request);
  TEST_ASSERT(alias_build.ok);
  TEST_ASSERT(alias_workspace.packet_work_units.data() == alias_data);
  TEST_ASSERT(alias_workspace.packet_work_units[1u] == 1u);

  rund::kernel::Workspace partial_alias_workspace{};
  TEST_ASSERT(Reserve(partial_alias_workspace, zero_request) == 0);
  partial_alias_workspace.packet_work_units.resize(5u);
  rund::kernel::ScheduleCompileRequest partial_alias_request = zero_request;
  partial_alias_request.packet_work_units = std::span<const rund::kernel::u64>(
      partial_alias_workspace.packet_work_units.data() + 1u, 4u);
  const rund::kernel::PartitionBuild partial_alias_build =
      Compile(partial_alias_workspace, partial_alias_request);
  TEST_ASSERT(!partial_alias_build.ok);
  TEST_ASSERT(std::string_view{partial_alias_build.reason} ==
              "invalid_packet_work_units");

  rund::kernel::Workspace reserved_alias_workspace{};
  TEST_ASSERT(Reserve(reserved_alias_workspace, zero_request) == 0);
  reserved_alias_workspace.packet_work_units.clear();
  TEST_ASSERT(reserved_alias_workspace.packet_work_units.capacity() >= 4u);
  TEST_ASSERT(reserved_alias_workspace.packet_work_units.data() != nullptr);
  rund::kernel::ScheduleCompileRequest reserved_alias_request = zero_request;
  reserved_alias_request.packet_work_units = std::span<const rund::kernel::u64>(
      reserved_alias_workspace.packet_work_units.data(), 4u);
  const rund::kernel::PartitionBuild reserved_alias_build =
      Compile(reserved_alias_workspace, reserved_alias_request);
  TEST_ASSERT(!reserved_alias_build.ok);
  TEST_ASSERT(std::string_view{reserved_alias_build.reason} ==
              "invalid_packet_work_units");

  rund::kernel::Workspace partition_load_alias_workspace{};
  TEST_ASSERT(Reserve(partition_load_alias_workspace, zero_request) == 0);
  partition_load_alias_workspace.partition_loads.resize(4u);
  rund::kernel::ScheduleCompileRequest partition_load_alias_request =
      zero_request;
  partition_load_alias_request.packet_work_units =
      std::span<const rund::kernel::u64>(
          partition_load_alias_workspace.partition_loads.data(), 4u);
  const rund::kernel::PartitionBuild partition_load_alias_build =
      Compile(partition_load_alias_workspace, partition_load_alias_request);
  TEST_ASSERT(!partition_load_alias_build.ok);
  TEST_ASSERT(std::string_view{partition_load_alias_build.reason} ==
              "invalid_packet_work_units");

  rund::kernel::Workspace fold_slot_alias_workspace{};
  TEST_ASSERT(Reserve(fold_slot_alias_workspace, zero_request) == 0);
  fold_slot_alias_workspace.fold_slots.values.resize(4u);
  rund::kernel::ScheduleCompileRequest fold_slot_alias_request = zero_request;
  fold_slot_alias_request.packet_work_units =
      std::span<const rund::kernel::u64>(
          fold_slot_alias_workspace.fold_slots.values.data(), 4u);
  const rund::kernel::PartitionBuild fold_slot_alias_build =
      Compile(fold_slot_alias_workspace, fold_slot_alias_request);
  TEST_ASSERT(!fold_slot_alias_build.ok);
  TEST_ASSERT(std::string_view{fold_slot_alias_build.reason} ==
              "invalid_packet_work_units");

  rund::kernel::Workspace stale_prefix_workspace{};
  const std::vector<rund::kernel::u64> larger_units{1u, 2u, 3u, 4u, 5u, 6u};
  const rund::kernel::ScheduleCompileRequest larger_request =
      Request(6u, 2u, rund::kernel::AllocationPolicy::AllowGrowth,
              std::span<const rund::kernel::u64>(larger_units.data(),
                                                 larger_units.size()));
  TEST_ASSERT(Compile(stale_prefix_workspace, larger_request).ok);
  rund::kernel::ScheduleCompileRequest stale_prefix_request = zero_request;
  stale_prefix_request.packet_work_units = std::span<const rund::kernel::u64>(
      stale_prefix_workspace.packet_work_units.data(),
      stale_prefix_request.packet_count);
  const rund::kernel::PartitionBuild stale_prefix_build =
      Compile(stale_prefix_workspace, stale_prefix_request);
  TEST_ASSERT(!stale_prefix_build.ok);
  TEST_ASSERT(std::string_view{stale_prefix_build.reason} ==
              "invalid_packet_work_units");
  return 0;
}

} // namespace workspace_weighted_contract
