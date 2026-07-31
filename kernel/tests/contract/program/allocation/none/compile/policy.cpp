#include "local.hpp"

namespace program_no_allocation_contract {

int Policy() {
  using namespace kernel_contract_test;

  rund::kernel::Workspace two_worker_workspace{};
  FakePool two_worker_pool = BuildStaticPool(2u);
  const rund::kernel::KernelProgramCompileRequest two_worker_request =
      TiledRequest(two_worker_pool, 2u);
  TEST_ASSERT(ReserveProgram(two_worker_workspace, two_worker_request) == 0);
  const rund::kernel::KernelProgramBuild two_worker_build =
      rund::kernel::CompileKernelProgram(two_worker_workspace,
                                         two_worker_request);
  TEST_ASSERT(two_worker_build.ok);
  TEST_ASSERT(two_worker_build.program.exec_kernel.physical_tiling_enabled);
  TEST_ASSERT(two_worker_build.program.exec_kernel.physical_tile_units ==
              2796203u);
  TEST_ASSERT(two_worker_build.program.exec_kernel.physical_tile_count == 6u);
  TEST_ASSERT(two_worker_build.program.exec_kernel.tile_count == 6u);
  TEST_ASSERT(two_worker_build.program.schedule.partition_count == 2u);

  rund::kernel::Workspace target_workspace{};
  FakePool target_pool = BuildStaticPool(2u);
  rund::kernel::KernelProgramCompileRequest target_request = two_worker_request;
  target_request.worker_backend = MakeFakeBackend(&target_pool);
  target_request.physical_tile_policy =
      rund::kernel::physical_tiles(2u, 262144u, 0u);
  TEST_ASSERT(ReserveProgram(target_workspace, target_request) == 0);
  const rund::kernel::KernelProgramBuild target_build =
      rund::kernel::CompileKernelProgram(target_workspace, target_request);
  TEST_ASSERT(target_build.ok);
  TEST_ASSERT(target_build.program.exec_kernel.physical_tiling_enabled);
  TEST_ASSERT(target_build.program.exec_kernel.physical_tile_units == 4194304u);
  TEST_ASSERT(target_build.program.exec_kernel.physical_tile_count == 4u);
  TEST_ASSERT(target_build.program.exec_kernel.tile_count == 4u);

  const rund::kernel::KernelProgramPhysicalTilePolicy per_worker_policy =
      rund::kernel::physical_tiles_per_worker(5u, 0u);
  TEST_ASSERT(per_worker_policy.enabled);
  TEST_ASSERT(per_worker_policy.target_tiles_per_worker == 5u);
  TEST_ASSERT(per_worker_policy.min_tile_units == 1u);
  TEST_ASSERT(per_worker_policy.max_tile_units == 0u);

  rund::kernel::Workspace per_worker_workspace{};
  FakePool per_worker_pool = BuildStaticPool(2u);
  rund::kernel::KernelProgramCompileRequest per_worker_request =
      two_worker_request;
  per_worker_request.worker_backend = MakeFakeBackend(&per_worker_pool);
  per_worker_request.physical_tile_policy =
      rund::kernel::physical_tiles_per_worker(5u);
  TEST_ASSERT(ReserveProgram(per_worker_workspace, per_worker_request) == 0);
  const rund::kernel::KernelProgramBuild per_worker_build =
      rund::kernel::CompileKernelProgram(per_worker_workspace,
                                         per_worker_request);
  TEST_ASSERT(per_worker_build.ok);
  TEST_ASSERT(per_worker_build.program.exec_kernel.physical_tiling_enabled);
  TEST_ASSERT(per_worker_build.program.exec_kernel.physical_tile_units ==
              1677722u);
  TEST_ASSERT(per_worker_build.program.exec_kernel.physical_tile_count == 10u);
  TEST_ASSERT(per_worker_build.program.exec_kernel.tile_count == 10u);
  TEST_ASSERT(per_worker_build.program.schedule.partition_count == 2u);

  rund::kernel::Workspace per_worker_capped_workspace{};
  FakePool per_worker_capped_pool = BuildStaticPool(2u);
  rund::kernel::KernelProgramCompileRequest per_worker_capped_request =
      two_worker_request;
  per_worker_capped_request.worker_backend =
      MakeFakeBackend(&per_worker_capped_pool);
  per_worker_capped_request.physical_tile_policy =
      rund::kernel::physical_tiles_per_worker(5u, 1048576u);
  TEST_ASSERT(ReserveProgram(per_worker_capped_workspace,
                             per_worker_capped_request) == 0);
  const rund::kernel::KernelProgramBuild per_worker_capped_build =
      rund::kernel::CompileKernelProgram(per_worker_capped_workspace,
                                         per_worker_capped_request);
  TEST_ASSERT(per_worker_capped_build.ok);
  TEST_ASSERT(
      per_worker_capped_build.program.exec_kernel.physical_tiling_enabled);
  TEST_ASSERT(per_worker_capped_build.program.exec_kernel.physical_tile_units ==
              1048576u);
  TEST_ASSERT(per_worker_capped_build.program.exec_kernel.physical_tile_count ==
              16u);
  TEST_ASSERT(per_worker_capped_build.program.exec_kernel.tile_count == 16u);
  TEST_ASSERT(per_worker_capped_build.program.schedule.partition_count == 2u);

  rund::kernel::Workspace capped_workspace{};
  FakePool capped_pool = BuildStaticPool(2u);
  rund::kernel::KernelProgramCompileRequest capped_request = two_worker_request;
  capped_request.worker_backend = MakeFakeBackend(&capped_pool);
  capped_request.physical_tile_policy =
      rund::kernel::physical_tiles(2u, 262144u, 1048576u);
  TEST_ASSERT(ReserveProgram(capped_workspace, capped_request) == 0);
  const rund::kernel::KernelProgramBuild capped_build =
      rund::kernel::CompileKernelProgram(capped_workspace, capped_request);
  TEST_ASSERT(capped_build.ok);
  TEST_ASSERT(capped_build.program.exec_kernel.physical_tiling_enabled);
  TEST_ASSERT(capped_build.program.exec_kernel.physical_tile_units == 1048576u);
  TEST_ASSERT(capped_build.program.exec_kernel.physical_tile_count == 16u);

  rund::kernel::Workspace alignment_workspace{};
  FakePool alignment_pool = BuildStaticPool(2u);
  rund::kernel::KernelProgramCompileRequest alignment_request =
      two_worker_request;
  alignment_request.worker_backend = MakeFakeBackend(&alignment_pool);
  alignment_request.schedule.preferred_alignment_packets = 1048576u;
  alignment_request.schedule.alignment_group_packets = 1048576u;
  alignment_request.physical_tile_policy =
      rund::kernel::physical_tiles(2u, 1u, 262144u);
  TEST_ASSERT(ReserveProgram(alignment_workspace, alignment_request) == 0);
  const rund::kernel::KernelProgramBuild alignment_build =
      rund::kernel::CompileKernelProgram(alignment_workspace,
                                         alignment_request);
  TEST_ASSERT(alignment_build.ok);
  TEST_ASSERT(!alignment_build.program.exec_kernel.physical_tiling_enabled);
  TEST_ASSERT(alignment_build.program.exec_kernel.physical_tile_units == 0u);
  TEST_ASSERT(alignment_build.program.exec_kernel.physical_tile_count == 0u);

  rund::kernel::Workspace disabled_workspace{};
  FakePool disabled_pool = BuildStaticPool(2u);
  rund::kernel::KernelProgramCompileRequest disabled_request =
      two_worker_request;
  disabled_request.worker_backend = MakeFakeBackend(&disabled_pool);
  disabled_request.physical_tile_policy = rund::kernel::no_physical_tiles();
  TEST_ASSERT(ReserveProgram(disabled_workspace, disabled_request) == 0);
  const rund::kernel::KernelProgramBuild disabled_build =
      rund::kernel::CompileKernelProgram(disabled_workspace, disabled_request);
  TEST_ASSERT(disabled_build.ok);
  TEST_ASSERT(!disabled_build.program.exec_kernel.physical_tiling_enabled);
  TEST_ASSERT(disabled_build.program.exec_kernel.physical_tile_units == 0u);
  TEST_ASSERT(disabled_build.program.exec_kernel.physical_tile_count == 0u);
  TEST_ASSERT(disabled_build.program.exec_kernel.tile_count ==
              disabled_build.program.schedule.partition_count);

  rund::kernel::Workspace small_workspace{};
  FakePool small_pool = BuildStaticPool(8u);
  const rund::kernel::KernelProgramCompileRequest small_request =
      TiledRequest(small_pool, 8u, 1u << 20u);
  TEST_ASSERT(ReserveProgram(small_workspace, small_request) == 0);
  const rund::kernel::KernelProgramBuild small_build =
      rund::kernel::CompileKernelProgram(small_workspace, small_request);
  TEST_ASSERT(small_build.ok);
  TEST_ASSERT(small_build.program.exec_kernel.physical_tiling_enabled);
  TEST_ASSERT(small_build.program.exec_kernel.physical_tile_units == 43691u);
  TEST_ASSERT(small_build.program.exec_kernel.physical_tile_count == 24u);
  TEST_ASSERT(small_build.program.exec_kernel.tile_count == 24u);
  return 0;
}

} // namespace program_no_allocation_contract
