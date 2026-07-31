#include "../state/storage.hpp"

namespace rund::node {

void Scheduler::WakeReadyReactor() noexcept {
  (void)DrainReadyReactor(0, false);
}

}  // namespace rund::node
