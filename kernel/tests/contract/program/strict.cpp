#include "cases.hpp"

#include "contract/support.hpp"
#include "test/assert.hpp"

#include <kernel/program/build.hpp>
#include <kernel/schedule/workspace.hpp>

#include <string_view>

int RunProgramStrictFloatContract() {
  using namespace kernel_contract_test;

  FakePool pool = BuildStaticPool(2u);
  rund::kernel::Workspace workspace{};
  rund::kernel::KernelProgramCompileRequest missing_floating_point_law_request{
      .schedule = rund::kernel::ScheduleCompileRequest{
          .packet_count = 4u,
          .execution_width = 2u,
          .intent = rund::kernel::PartitionIntent::StaticWidth,
          .placement = rund::kernel::PlacementPolicy::Uniform,
          .allocation = rund::kernel::AllocationPolicy::NoGrowth,
      },
      .worker_backend = MakeFakeBackend(&pool),
      .require_no_allocation = true,
      .collect_worker_stats = false,
      .fold_operation = rund::kernel::FoldOperation::StrictFloat32Add,
  };
  TEST_ASSERT(rund::kernel::ReserveWorkspace(workspace,
                                       rund::kernel::KernelProgramWorkspaceReservation(missing_floating_point_law_request)));
  const rund::kernel::KernelProgramBuild missing_law =
      rund::kernel::CompileKernelProgram(workspace, missing_floating_point_law_request);
  TEST_ASSERT(!missing_law.ok);
  TEST_ASSERT(std::string_view{missing_law.reason} == "floating_point_fold_forbidden");
  TEST_ASSERT(!workspace.telemetry.strict_fp_software_reference);
  TEST_ASSERT(!workspace.program.telemetry_schema.strict_fp_software_reference);

  rund::kernel::KernelProgramCompileRequest strict_float_reduction_request =
      missing_floating_point_law_request;
  strict_float_reduction_request.strict_float_reduction =
      rund::kernel::StrictFloat32ReductionPolicy();
  const rund::kernel::KernelProgramBuild strict_build =
      rund::kernel::CompileKernelProgram(workspace, strict_float_reduction_request);
  TEST_ASSERT(strict_build.ok);
  TEST_ASSERT(strict_build.program.fold_graph.strict_floating_point);
  TEST_ASSERT(strict_build.program.fold_graph.value_domain == rund::kernel::FoldValueDomain::Float32Strict);
  TEST_ASSERT(workspace.telemetry.strict_fp_software_reference);
  TEST_ASSERT(!workspace.telemetry.strict_fp_backend_supported);
  TEST_ASSERT(workspace.program.telemetry_schema.strict_fp_software_reference);
  return 0;
}
