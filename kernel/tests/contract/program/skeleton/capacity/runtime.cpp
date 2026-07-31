#include "local.hpp"

namespace program_skeleton_contract {

rund::kernel::ParallelRuntime
AcquireTestParallelRuntime(void *const raw, const rund::kernel::u32 workers) {
  auto *const context = static_cast<TestParallelRuntimeContext *>(raw);
  if (context == nullptr) {
    return rund::kernel::ParallelRuntime{
        .workers = workers,
        .reason = "test_parallel_runtime_missing",
    };
  }
  const rund::kernel::u32 resolved_workers = workers == 0u ? 4u : workers;
  context->pool.workers = resolved_workers;
  return rund::kernel::ParallelRuntime{
      .workspace = &context->workspace,
      .worker_backend = kernel_contract_test::MakeFakeBackend(&context->pool),
      .workers = resolved_workers,
      .valid = true,
      .reason = "pass",
  };
}

int test_skeleton_policy_runtime() {
  using namespace kernel_contract_test;

  std::vector<rund::kernel::i32> policy_storage(8192u);
  auto policy_view = rund::kernel::view<rund::kernel::i32>(
      policy_storage.data(), rund::kernel::Index<1u>{policy_storage.size()});
  TEST_ASSERT(policy_view);
  const rund::kernel::SkeletonResult seq_result = rund::kernel::each(
      rund::kernel::seq(), rund::kernel::space(policy_storage.size()),
      [&](auto index) noexcept {
        policy_view(index) = static_cast<rund::kernel::i32>(index[0] + 1u);
      });
  TEST_ASSERT(seq_result.ok);
  TEST_ASSERT(seq_result.visited_count == policy_storage.size());
  for (std::size_t index = 0u; index < policy_storage.size(); ++index) {
    TEST_ASSERT(policy_storage[index] ==
                static_cast<rund::kernel::i32>(index + 1u));
  }

  TEST_ASSERT(rund::kernel::par(4u).valid);
  TEST_ASSERT(ExpectReason(
                  "parallel_runtime_missing",
                  rund::kernel::each(rund::kernel::par(4u),
                                     rund::kernel::space(policy_storage.size()),
                                     [](auto) noexcept {})) == 0);
  TEST_ASSERT(rund::kernel::par().valid);
  TEST_ASSERT(ExpectReason(
                  "parallel_runtime_missing",
                  rund::kernel::each(rund::kernel::par(),
                                     rund::kernel::space(policy_storage.size()),
                                     [](auto) noexcept {})) == 0);

  TestParallelRuntimeContext runtime_context{
      .pool = BuildStaticPool(4u),
  };
  {
    rund::kernel::executor_detail::ScopedParallelRuntimeProvider provider(
        rund::kernel::ParallelRuntimeProvider{
            .context = &runtime_context,
            .acquire = AcquireTestParallelRuntime,
        });
    TEST_ASSERT(provider);
    const rund::kernel::SkeletonResult default_par_result = rund::kernel::each(
        rund::kernel::par(), rund::kernel::space(policy_storage.size()),
        [&](auto index) noexcept {
          policy_view(index) =
              rund::math32::detail::ScalarAddWrap(policy_view(index), 7);
        });
    TEST_ASSERT(default_par_result.ok);
    TEST_ASSERT(default_par_result.visited_count == policy_storage.size());
    TEST_ASSERT(default_par_result.partition_boundary_checked);
    TEST_ASSERT(default_par_result.partition_boundary_aligned);
    TEST_ASSERT(runtime_context.workspace.program.ok);
    TEST_ASSERT(runtime_context.workspace.program.schedule.execution_width ==
                4u);
  }
  for (std::size_t index = 0u; index < policy_storage.size(); ++index) {
    TEST_ASSERT(policy_storage[index] ==
                static_cast<rund::kernel::i32>(index + 8u));
  }

  bool policy_zero_visited = false;
  {
    rund::kernel::executor_detail::ScopedParallelRuntimeProvider provider(
        rund::kernel::ParallelRuntimeProvider{
            .context = &runtime_context,
            .acquire = AcquireTestParallelRuntime,
        });
    const rund::kernel::SkeletonResult par_zero =
        rund::kernel::each(rund::kernel::par(4u), rund::kernel::space(0u),
                           [&](auto) noexcept { policy_zero_visited = true; });
    TEST_ASSERT(par_zero.ok);
    TEST_ASSERT(par_zero.visited_count == 0u);
    TEST_ASSERT(!policy_zero_visited);
  }
  TEST_ASSERT(ExpectReason(
                  "parallel_runtime_missing",
                  rund::kernel::each(rund::kernel::par(4u),
                                     rund::kernel::space(policy_storage.size()),
                                     [](auto) noexcept {})) == 0);

  return 0;
}

} // namespace program_skeleton_contract
