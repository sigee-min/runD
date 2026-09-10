#include "local.hpp"

#include <algorithm>

namespace rund::node::reactor_registry_detail {

OrderIterator FindOrder(ReactorRegistry &registry,
                        const std::uint64_t wait_id) noexcept {
  return std::lower_bound(
      registry.order.begin(), registry.order.end(), wait_id,
      [&registry](const std::uint32_t slot, const std::uint64_t value) {
        return registry.slots[slot].wait.wait_id < value;
      });
}

ConstOrderIterator FindOrder(const ReactorRegistry &registry,
                             const std::uint64_t wait_id) noexcept {
  return std::lower_bound(
      registry.order.begin(), registry.order.end(), wait_id,
      [&registry](const std::uint32_t slot, const std::uint64_t value) {
        return registry.slots[slot].wait.wait_id < value;
      });
}

FdIterator FindFd(ReactorRegistry &registry, const ReactorHandle fd) noexcept {
  return std::lower_bound(
      registry.fds.begin(), registry.fds.end(), fd,
      [](const ReactorFdState &state, const ReactorHandle value) {
        return state.fd < value;
      });
}

ConstFdIterator FindFd(const ReactorRegistry &registry,
                       const ReactorHandle fd) noexcept {
  return std::lower_bound(
      registry.fds.begin(), registry.fds.end(), fd,
      [](const ReactorFdState &state, const ReactorHandle value) {
        return state.fd < value;
      });
}

bool FdMatches(const ConstFdIterator found, const ConstFdIterator end,
               const ReactorHandle fd) noexcept {
  return found != end && found->fd == fd;
}

bool FdMatches(const FdIterator found, const FdIterator end,
               const ReactorHandle fd) noexcept {
  return found != end && found->fd == fd;
}

} // namespace rund::node::reactor_registry_detail
