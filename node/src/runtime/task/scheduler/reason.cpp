#include <rund/task/results.hpp>
#include <rund/task/status.hpp>

namespace rund::node {

task::Status FailJoin(const ReasonCode code) noexcept {
  return task::Status::fail(code);
}

task::Status FailScope(const ReasonCode code) noexcept {
  return task::Status::fail(code);
}

::rund::detail::task::AwaitDecision FailYield(const ReasonCode code) noexcept {
  return ::rund::detail::task::AwaitDecision{.status =
                                                 task::Status::fail(code)};
}

::rund::detail::task::AwaitDecision FailSleep(const ReasonCode code) noexcept {
  return ::rund::detail::task::AwaitDecision{.status =
                                                 task::Status::fail(code)};
}

::rund::detail::task::IoDecision FailIo(const ReasonCode code,
                                        const short revents) noexcept {
  return ::rund::detail::task::IoDecision{.status = task::Status::fail(code),
                                          .revents = revents};
}

} // namespace rund::node
