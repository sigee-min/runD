#include "local.hpp"

#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ready_queue_detail {

int CheckFailureOrder() {
  for (const std::uint32_t workers : {1u, 4u}) {
    rund::task::Status failure{};
    const rund::Session::Result failure_report = rund::run(
        rund::SessionConfig{
          .workers = 1u,
          .scheduler = {
            .task_workers = workers,
            .task_capacity = 8u,
            .ready_queue_capacity = 8u,
          },
        },
        [&] {
          std::array<rund::task::Handle, 8u> handles{};
          for (std::size_t index = 0u; index < handles.size(); ++index) {
            handles[index] = rund::task::spawn("ordered-failure", [index] {
              if (index == 2u) {
                (void)rund::task::join(rund::task::Handle{});
              } else if (index == 5u) {
                throw 1;
              }
            });
            TEST_ASSERT(handles[index]);
          }
          failure = rund::task::join_all(handles);
        });
    TEST_ASSERT(failure_report);
    TEST_ASSERT(!failure);
    TEST_ASSERT(failure.code() ==
                rund::ReasonCode::TaskLeafPrimitiveForbidden);
  }
  return 0;
}

}  // namespace ready_queue_detail
