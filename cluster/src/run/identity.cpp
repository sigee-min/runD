#include <cluster/run/identity.hpp>

namespace rund::cluster {

bool RunKey::complete() const noexcept {
  if (!static_cast<bool>(work) || !static_cast<bool>(input) ||
      !static_cast<bool>(program) || !static_cast<bool>(fold) ||
      !static_cast<bool>(numeric) || !static_cast<bool>(capacity) ||
      !static_cast<bool>(output)) {
    return false;
  }
  return !sharded || static_cast<bool>(shard);
}

bool operator==(const RunKey &lhs, const RunKey &rhs) noexcept {
  const bool same_shard =
      lhs.sharded == rhs.sharded && (!lhs.sharded || lhs.shard == rhs.shard);
  return lhs.work == rhs.work && same_shard && lhs.input == rhs.input &&
         lhs.time == rhs.time && lhs.checkpoint == rhs.checkpoint &&
         lhs.program == rhs.program && lhs.fold == rhs.fold &&
         lhs.numeric == rhs.numeric && lhs.capacity == rhs.capacity &&
         lhs.output == rhs.output;
}

bool operator==(const RunAttempt &lhs, const RunAttempt &rhs) noexcept {
  return lhs.run == rhs.run && lhs.retry == rhs.retry;
}

} // namespace rund::cluster
