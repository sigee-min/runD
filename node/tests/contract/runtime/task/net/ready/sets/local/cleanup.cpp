#include "../local.hpp"

#include <unistd.h>

namespace rund::node::test_contract::ready_sets {

SocketPairCleanup::~SocketPairCleanup() {
  if (left >= 0) {
    static_cast<void>(::close(left));
  }
  if (right >= 0) {
    static_cast<void>(::close(right));
  }
}

}  // namespace rund::node::test_contract::ready_sets
