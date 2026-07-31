#pragma once

#include "contract/program/allocation/none/request.hpp"
#include "contract/support.hpp"
#include "test/assert.hpp"

#include <kernel/dispatch/orchestrator.hpp>
#include <kernel/program/build.hpp>
#include <kernel/program/report.hpp>
#include <kernel/schedule/workspace.hpp>

#include <array>
#include <string_view>

namespace program_no_allocation_contract {

[[nodiscard]] inline rund::kernel::KernelProgramCompileRequest
BaseRequest(kernel_contract_test::FakePool &pool) {
  return rund::kernel::KernelProgramCompileRequest{
      .schedule = kernel_contract_test::program_no_allocation::ScheduleRequest(),
      .worker_backend = kernel_contract_test::MakeFakeBackend(&pool),
      .require_no_allocation = true,
      .collect_worker_stats = true,
      .fold_operation = rund::kernel::FoldOperation::FixedBinaryTreeHash,
  };
}

[[nodiscard]] inline rund::kernel::KernelProgramCompileRequest
TiledRequest(kernel_contract_test::FakePool &pool, rund::kernel::u32 width,
             rund::kernel::u32 packet_count = 1u << 24u) {
  return rund::kernel::KernelProgramCompileRequest{
      .schedule =
          rund::kernel::ScheduleCompileRequest{
              .packet_count = packet_count,
              .execution_width = width,
              .intent = rund::kernel::PartitionIntent::StaticWidth,
              .placement = rund::kernel::PlacementPolicy::Uniform,
              .allocation = rund::kernel::AllocationPolicy::NoGrowth,
          },
      .worker_backend = kernel_contract_test::MakeFakeBackend(&pool),
      .require_no_allocation = true,
      .collect_worker_stats = true,
      .fold_operation = rund::kernel::FoldOperation::FixedBinaryTreeHash,
  };
}

[[nodiscard]] inline int
ReserveProgram(rund::kernel::Workspace &workspace,
               const rund::kernel::KernelProgramCompileRequest &request) {
  TEST_ASSERT(rund::kernel::ReserveWorkspace(
      workspace, rund::kernel::KernelProgramWorkspaceReservation(request)));
  return 0;
}

int Policy();
int Failure();

} // namespace program_no_allocation_contract
