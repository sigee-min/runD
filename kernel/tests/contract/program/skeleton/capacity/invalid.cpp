#include "local.hpp"

namespace program_skeleton_contract {

int test_skeleton_invalid_executor_and_policy_rejections() {
  rund::kernel::Workspace invalid_executor_workspace{};
  const rund::kernel::Executor invalid_exec = rund::kernel::executor(
      invalid_executor_workspace, rund::kernel::WorkerBackend{}, 4u);
  TEST_ASSERT(!invalid_exec.valid);
  TEST_ASSERT(
      ExpectReason("executor_backend_invalid",
                   rund::kernel::each(invalid_exec, rund::kernel::space(4u),
                                      [](auto) noexcept {})) == 0);
  const rund::kernel::PreparedEach<1u> invalid_prepared =
      invalid_exec.prepare(rund::kernel::space(4u));
  TEST_ASSERT(!invalid_prepared);
  const rund::kernel::KernelExecutionReport invalid_exec_report =
      rund::kernel::execution_report(invalid_exec);
  TEST_ASSERT(!invalid_exec_report.observed);
  TEST_ASSERT(!invalid_exec_report.ok);
  TEST_ASSERT(invalid_exec_report.reason ==
              std::string_view{"executor_backend_invalid"});
  TEST_ASSERT(ExpectReason("executor_backend_invalid",
                           invalid_prepared.run([](auto) noexcept {})) == 0);

  TEST_ASSERT(!rund::kernel::par(0u).valid);
  TEST_ASSERT(ExpectReason("parallel_zero_workers",
                           rund::kernel::each(rund::kernel::par(0u),
                                              rund::kernel::space(4u),
                                              [](auto) noexcept {})) == 0);
  TEST_ASSERT(!rund::kernel::par(4u, rund::kernel::align(0u)).valid);
  TEST_ASSERT(
      ExpectReason("skeleton_alignment_zero",
                   rund::kernel::each(
                       rund::kernel::par(4u, rund::kernel::align(0u)),
                       rund::kernel::space(4u), [](auto) noexcept {})) == 0);

  TEST_ASSERT(rund::kernel::align(4u).valid);
  TEST_ASSERT(rund::kernel::align(4u).units == 4u);
  TEST_ASSERT(!rund::kernel::align(0u).valid);
  TEST_ASSERT(std::string_view{rund::kernel::align(0u).reason} ==
              "skeleton_alignment_zero");
  TEST_ASSERT(!rund::kernel::align(-1).valid);
  TEST_ASSERT(std::string_view{rund::kernel::align(-1).reason} ==
              "skeleton_alignment_negative");
  return 0;
}

} // namespace program_skeleton_contract
