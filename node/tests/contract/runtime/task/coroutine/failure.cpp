#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include "local.hpp"

namespace rund::node::test_contract::coroutine {

rund::task::Task<void>
ObserveInvalidIo(std::atomic<std::uint32_t> *const after_await,
                 rund::ReasonCode *const code) {
  const rund::task::IoResult ready =
      co_await rund::host::io::readable(rund::host::io::FdView{});
  *code = ready.code();
  after_await->fetch_add(1u, std::memory_order_release);
}

rund::task::Task<void>
ObserveChildException(std::atomic<std::uint32_t> *const after_await,
                      rund::ReasonCode *const code) {
  const rund::task::Status joined =
      co_await rund::task::spawn("coroutine-throwing-child", [] {
        throw std::runtime_error{"coroutine child failure"};
      });
  *code = joined.code();
  after_await->fetch_add(1u, std::memory_order_release);
}

int RunRuntimeTaskCoroutineFailureContract() {
  std::atomic<std::uint32_t> after_invalid_io{0u};
  rund::ReasonCode invalid_io_code = rund::ReasonCode::Ok;
  rund::task::Status invalid_io_join{};
  bool invalid_io_handle_valid = false;
  const rund::Session::Result invalid_io_report = rund::run(
      rund::SessionConfig{
          .id = 795u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
              },
      },
      [&] {
        const rund::task::Handle task = rund::task::spawn(
            "coroutine-invalid-io",
            ObserveInvalidIo(&after_invalid_io, &invalid_io_code));
        invalid_io_handle_valid = static_cast<bool>(task);
        invalid_io_join = rund::task::join(task);
      });
  TEST_ASSERT(invalid_io_report.ok());
  TEST_ASSERT(invalid_io_handle_valid);
  TEST_ASSERT(invalid_io_join.ok());
  TEST_ASSERT(after_invalid_io.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(invalid_io_code == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(invalid_io_report.tasks().completed() == 1u);
  TEST_ASSERT(invalid_io_report.tasks().failed() == 0u);
  TEST_ASSERT(invalid_io_report.tasks().coroutine_failures() == 0u);

  std::atomic<std::uint32_t> after_child_exception{0u};
  rund::ReasonCode child_exception_code = rund::ReasonCode::Ok;
  rund::task::Status child_exception_join{};
  bool child_exception_handle_valid = false;
  const rund::Session::Result child_exception_report = rund::run(
      rund::SessionConfig{
          .id = 799u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
              },
      },
      [&] {
        const rund::task::Handle task =
            rund::task::spawn("coroutine-child-exception",
                              ObserveChildException(&after_child_exception,
                                                    &child_exception_code));
        child_exception_handle_valid = static_cast<bool>(task);
        child_exception_join = rund::task::join(task);
      });
  TEST_ASSERT(child_exception_report.ok());
  TEST_ASSERT(child_exception_handle_valid);
  TEST_ASSERT(child_exception_join.ok());
  TEST_ASSERT(after_child_exception.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(child_exception_code == rund::ReasonCode::TaskFailed);
  TEST_ASSERT(child_exception_report.tasks().spawned() == 2u);
  TEST_ASSERT(child_exception_report.tasks().completed() == 1u);
  TEST_ASSERT(child_exception_report.tasks().failed() == 1u);
  TEST_ASSERT(child_exception_report.tasks().coroutine_failures() == 0u);

  return 0;
}

} // namespace rund::node::test_contract::coroutine
