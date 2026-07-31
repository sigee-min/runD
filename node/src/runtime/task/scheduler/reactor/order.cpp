#include "order.hpp"

#include <algorithm>

namespace rund::node {

bool ReactorReadyPrecedes(const ReactorReady& lhs,
                          const ReactorReady& rhs) noexcept {
  if (lhs.wait_id != rhs.wait_id) {
    return lhs.wait_id < rhs.wait_id;
  }
  if (lhs.task_id != rhs.task_id) {
    return lhs.task_id < rhs.task_id;
  }
  if (lhs.fd != rhs.fd) {
    return lhs.fd < rhs.fd;
  }
  return lhs.interest < rhs.interest;
}

}  // namespace rund::node
