#include "support.hpp"

#include <kernel/dispatch/orchestrator.hpp>
#include <kernel/program/executor.hpp>
#include <kernel/program/executor/policy.hpp>
#include <kernel/program/skeleton.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

int RunRuntimeKernelContract() {
  using namespace rund::node::test_contract;

  rund::Session session{};

  const rund::Session::Status opened = session.open(Options());
  TEST_ASSERT(opened.ok());
  TEST_ASSERT(opened.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(session.resources());
  const rund::Resources resources = session.resources();
  TEST_ASSERT(resources.worker_capacity_milli.size() == 2u);
  TEST_ASSERT(resources.topology.affinity ==
              rund::EvidenceTruth::Verified);
  std::array<rund::kernel::i32, 4u> par_storage{};
  auto par_view = rund::kernel::view<rund::kernel::i32>(
      par_storage.data(), rund::kernel::Index<1u>{par_storage.size()});
  TEST_ASSERT(par_view);
  const rund::kernel::SkeletonResult missing_parallel = rund::kernel::each(
      rund::kernel::par(2u), rund::kernel::space(par_storage.size()),
      [](auto) noexcept {});
  TEST_ASSERT(!missing_parallel.ok);
  TEST_ASSERT(missing_parallel.reason ==
              std::string_view{"parallel_runtime_missing"});

  bool node_parallel_ok = false;
  bool explicit_parallel_ok = false;
  bool width_mismatch_ok = false;
  const rund::Session::Result scope = session.scope([&] {
    const rund::kernel::SkeletonResult node_parallel = rund::kernel::each(
        rund::kernel::par(), rund::kernel::space(par_storage.size()),
        [&](auto index) noexcept {
          par_view(index) = static_cast<rund::kernel::i32>(index[0] + 5u);
        });
    node_parallel_ok =
        node_parallel.ok && node_parallel.visited_count == par_storage.size();
    const rund::kernel::SkeletonResult explicit_parallel = rund::kernel::each(
        rund::kernel::par(2u), rund::kernel::space(par_storage.size()),
        [](auto) noexcept {});
    explicit_parallel_ok = explicit_parallel.ok;
    const rund::kernel::SkeletonResult width_mismatch = rund::kernel::each(
        rund::kernel::par(3u), rund::kernel::space(par_storage.size()),
        [](auto) noexcept {});
    width_mismatch_ok = !width_mismatch.ok &&
                        width_mismatch.reason ==
                            std::string_view{"parallel_runtime_width_mismatch"};
  });
  TEST_ASSERT(scope.ok());
  TEST_ASSERT(scope);
  TEST_ASSERT(scope.error().empty());
  TEST_ASSERT(scope.exit_code() == 0);
  TEST_ASSERT(node_parallel_ok);
  TEST_ASSERT(explicit_parallel_ok);
  TEST_ASSERT(width_mismatch_ok);
  for (std::size_t index = 0u; index < par_storage.size(); ++index) {
    TEST_ASSERT(par_storage[index] ==
                static_cast<rund::kernel::i32>(index + 5u));
  }
  const rund::kernel::SkeletonResult cleared_parallel = rund::kernel::each(
      rund::kernel::par(2u), rund::kernel::space(par_storage.size()),
      [](auto) noexcept {});
  TEST_ASSERT(!cleared_parallel.ok);
  TEST_ASSERT(cleared_parallel.reason ==
              std::string_view{"parallel_runtime_missing"});

  std::array<rund::kernel::i32, 8192u> rund_storage{};
  std::mutex thread_mutex{};
  std::vector<std::thread::id> thread_ids{};
  rund::kernel::SkeletonResult rund_parallel{};
  bool rund_view_ok = false;
  const rund::Session::Result rund_report = rund::run(
      rund::SessionConfig{.id = 77u, .workers = 4u, .trace_capacity = 64u},
      [&] {
        auto rund_view = rund::kernel::view<rund::kernel::i32>(
            rund_storage.data(), rund::kernel::Index<1u>{rund_storage.size()});
        rund_view_ok = static_cast<bool>(rund_view);
        rund_parallel = rund::kernel::each(
            rund::kernel::par(), rund::kernel::space(rund_storage.size()),
            [&](auto index) {
              {
                std::lock_guard<std::mutex> lock(thread_mutex);
                const std::thread::id id = std::this_thread::get_id();
                bool seen = false;
                for (const std::thread::id known : thread_ids) {
                  if (known == id) {
                    seen = true;
                  }
                }
                if (!seen) {
                  thread_ids.push_back(id);
                }
              }
              rund_view(index) = static_cast<rund::kernel::i32>(index[0] * 2u);
            });
      });
  TEST_ASSERT(rund_report.ok());
  TEST_ASSERT(rund_view_ok);
  TEST_ASSERT(rund_parallel.ok);
  TEST_ASSERT(thread_ids.size() > 1u);
  for (std::size_t index = 0u; index < rund_storage.size(); ++index) {
    TEST_ASSERT(rund_storage[index] ==
                static_cast<rund::kernel::i32>(index * 2u));
  }

  constexpr std::size_t run_count = 2u;
  constexpr std::size_t value_count = 256u;
  std::array<std::array<rund::kernel::i32, value_count>, run_count>
      independent_storage{};
  std::array<rund::kernel::SkeletonResult, run_count> independent_parallel{};
  std::array<rund::ReasonCode, run_count> independent_code{
      rund::ReasonCode::SessionResultMissing,
      rund::ReasonCode::SessionResultMissing};
  std::array<std::thread, run_count> independent_runs{};
  for (std::size_t run = 0u; run < run_count; ++run) {
    independent_runs[run] = std::thread{[&, run] {
      const rund::Session::Result report = rund::run(
          rund::SessionConfig{.id = static_cast<std::uint64_t>(78u + run),
                              .workers = 2u},
          [&] {
            auto view = rund::kernel::view<rund::kernel::i32>(
                independent_storage[run].data(),
                rund::kernel::Index<1u>{value_count});
            independent_parallel[run] = rund::kernel::each(
                rund::kernel::par(), rund::kernel::space(value_count),
                [&](auto index) noexcept {
                  view(index) = static_cast<rund::kernel::i32>(
                      index[0] + run * value_count);
                });
          });
      independent_code[run] = report.code();
    }};
  }
  for (std::thread &run : independent_runs) {
    run.join();
  }
  for (std::size_t run = 0u; run < run_count; ++run) {
    TEST_ASSERT(independent_code[run] == rund::ReasonCode::Ok);
    TEST_ASSERT(independent_parallel[run].ok);
    for (std::size_t index = 0u; index < value_count; ++index) {
      TEST_ASSERT(independent_storage[run][index] ==
                  static_cast<rund::kernel::i32>(index + run * value_count));
    }
  }

  return 0;
}
