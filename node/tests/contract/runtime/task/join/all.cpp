#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <span>

int RunRuntimeTaskJoinAllContract() {
  std::array<bool, 3u> ran{};
  std::array<rund::task::Handle, 3u> handles{};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
        .workers = 2u,
        .scheduler = {
          .task_capacity = 8u,
          .ready_queue_capacity = 8u,
        },
      },
      [&] {
        for (std::size_t index = 0u; index < handles.size(); ++index) {
          handles[index] = rund::task::spawn("join-all-worker",
                                             [&, index] { ran[index] = true; });
        }
        joined =
            rund::task::join_all(std::span<const rund::task::Handle>{handles});
      });
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(ran[0] && ran[1] && ran[2]);
  return 0;
}
