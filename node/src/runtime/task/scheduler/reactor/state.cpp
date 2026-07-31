#include "../state/model/stop.hpp"
#include "../state/storage.hpp"

namespace rund::node {

SchedulerReactorState::SchedulerReactorState() noexcept = default;
SchedulerReactorState::~SchedulerReactorState() = default;

std::uint64_t ReactorHostHandleId(
    const int fd,
    const std::uint64_t explicit_host_handle_id) noexcept {
  if (explicit_host_handle_id != 0u) {
    return explicit_host_handle_id;
  }
  return static_cast<std::uint64_t>(fd) + 1u;
}

}  // namespace rund::node
