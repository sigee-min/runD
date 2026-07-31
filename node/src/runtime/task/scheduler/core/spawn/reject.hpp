#pragma once

namespace rund::node {

inline void RejectSpawnCompletion(const CompletionLease lease,
                                  const ReasonCode code) noexcept {
  if (!lease) {
    return;
  }
  (void)CompletionPool::terminate(lease, code);
  CompletionPool::release(lease);
}

} // namespace rund::node
