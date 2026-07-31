#include "local.hpp"

namespace program_no_allocation_contract {

int Failure() {
  using namespace kernel_contract_test;

  FakePool pool = BuildStaticPool(3u);
  rund::kernel::Workspace workspace{};
  const rund::kernel::KernelProgramCompileRequest request = BaseRequest(pool);
  TEST_ASSERT(ReserveProgram(workspace, request) == 0);
  TEST_ASSERT(rund::kernel::CompileKernelProgram(workspace, request).ok);

  rund::kernel::KernelProgramCompileRequest capacity_request = request;
  capacity_request.schedule.execution_width = 64u;
  const rund::kernel::KernelProgramBuild capacity_failure =
      rund::kernel::CompileKernelProgram(workspace, capacity_request);
  TEST_ASSERT(!capacity_failure.ok);
  TEST_ASSERT(capacity_failure.program.schedule.partition_count == 0u);
  TEST_ASSERT(capacity_failure.program.fold_graph.partition_count == 0u);
  TEST_ASSERT(
      rund::kernel::ViewKernelProgram(workspace).schedule.partition_count ==
      0u);
  TEST_ASSERT(
      rund::kernel::ViewKernelProgram(workspace).fold_graph.partition_count ==
      0u);

  rund::kernel::KernelProgramCompileRequest schedule_request = request;
  schedule_request.schedule.execution_width = 0u;
  schedule_request.collect_worker_stats = false;
  const rund::kernel::KernelProgramBuild schedule_failure =
      rund::kernel::CompileKernelProgram(workspace, schedule_request);
  TEST_ASSERT(!schedule_failure.ok);
  TEST_ASSERT(schedule_failure.program.schedule.partition_count == 0u);
  TEST_ASSERT(schedule_failure.program.fold_graph.partition_count == 0u);
  TEST_ASSERT(
      rund::kernel::ViewKernelProgram(workspace).schedule.partition_count ==
      0u);
  TEST_ASSERT(
      rund::kernel::ViewKernelProgram(workspace).fold_graph.partition_count ==
      0u);

  rund::kernel::KernelProgramCompileRequest fold_request = request;
  fold_request.fold_operation = rund::kernel::FoldOperation::StrictFloat32Add;
  fold_request.strict_float_reduction =
      rund::kernel::StrictFloatReductionPolicy{};
  const rund::kernel::KernelProgramBuild fold_failure =
      rund::kernel::CompileKernelProgram(workspace, fold_request);
  TEST_ASSERT(!fold_failure.ok);
  TEST_ASSERT(std::string_view{fold_failure.reason} ==
              "floating_point_fold_forbidden");
  TEST_ASSERT(fold_failure.program.schedule.partition_count == 3u);
  TEST_ASSERT(fold_failure.program.fold_graph.partition_count == 0u);
  TEST_ASSERT(
      rund::kernel::ViewKernelProgram(workspace).schedule.partition_count ==
      3u);
  TEST_ASSERT(
      rund::kernel::ViewKernelProgram(workspace).fold_graph.partition_count ==
      0u);

  rund::kernel::KernelProgramCompileRequest dispatch_request = request;
  dispatch_request.worker_backend = rund::kernel::WorkerBackend{};
  dispatch_request.collect_worker_stats = false;
  dispatch_request.require_dispatch_backend = true;
  const rund::kernel::KernelProgramBuild dispatch_failure =
      rund::kernel::CompileKernelProgram(workspace, dispatch_request);
  TEST_ASSERT(!dispatch_failure.ok);
  TEST_ASSERT(std::string_view{dispatch_failure.reason} ==
              "pool_width_mismatch");
  TEST_ASSERT(dispatch_failure.program.schedule.partition_count == 3u);
  TEST_ASSERT(dispatch_failure.program.fold_graph.partition_count == 3u);
  TEST_ASSERT(
      rund::kernel::ViewKernelProgram(workspace).schedule.partition_count ==
      3u);
  TEST_ASSERT(
      rund::kernel::ViewKernelProgram(workspace).fold_graph.partition_count ==
      3u);
  return 0;
}

} // namespace program_no_allocation_contract
