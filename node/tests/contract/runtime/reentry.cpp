#include "test/assert.hpp"

#include <rund/session.hpp>

#include <cstdint>
#include <string_view>

namespace {

std::uint32_t plain_calls = 0u;
rund::Session *seen_runtime = nullptr;

void PlainCallback() { ++plain_calls; }

void RuntimeCallback(rund::Session &runtime) { seen_runtime = &runtime; }

struct DualCallback final {
  bool &used_runtime;

  void operator()() { used_runtime = false; }
  void operator()(rund::Session &) { used_runtime = true; }
};

struct BorrowedCallback final {
  std::uint32_t &calls;

  explicit BorrowedCallback(std::uint32_t &value) noexcept : calls(value) {}
  BorrowedCallback(const BorrowedCallback &) = delete;
  BorrowedCallback &operator=(const BorrowedCallback &) = delete;
  void operator()() { ++calls; }
};

} // namespace

int RunRuntimeReentryContract() {
  bool nested_session_ok = false;
  const rund::Session::Result nested_report =
      rund::run(rund::SessionConfig{.workers = 2u}, [&] {
        const rund::Session::Result inner = rund::run([] {});
        nested_session_ok = inner.ok();
      });
  TEST_ASSERT(nested_report.ok());
  TEST_ASSERT(nested_session_ok);

  plain_calls = 0u;
  const rund::Session::Result plain = rund::run(PlainCallback);
  TEST_ASSERT(plain);
  TEST_ASSERT(plain_calls == 1u);

  seen_runtime = nullptr;
  const rund::Session::Result runtime = rund::run(&RuntimeCallback);
  TEST_ASSERT(runtime);
  TEST_ASSERT(seen_runtime != nullptr);

  bool used_runtime = false;
  DualCallback dual{used_runtime};
  TEST_ASSERT(rund::run(dual));
  TEST_ASSERT(used_runtime);

  std::uint32_t borrowed_calls = 0u;
  BorrowedCallback borrowed{borrowed_calls};
  TEST_ASSERT(rund::run(borrowed));
  TEST_ASSERT(borrowed_calls == 1u);

  return 0;
}
