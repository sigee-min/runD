#include <rund/session.hpp>

namespace consumer_fixture {

[[nodiscard]] bool ValidatePrivateSdkLink() {
  const rund::Session::Result result = rund::run(
      rund::SessionConfig{
        .workers = 1u,
        .scheduler = {
          .task_capacity = 1u,
          .ready_queue_capacity = 1u,
        },
      },
      [] {});
  return result.ok();
}

}  // namespace consumer_fixture
