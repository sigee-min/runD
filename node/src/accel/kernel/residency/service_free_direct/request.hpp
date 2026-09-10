#pragma once

#include "evidence.hpp"
#include "proof.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

// Aggregate-only handoff. Per-iteration service and callback members are
// intentionally absent and guarded by compile-time contract tests.
struct ServiceFreeDirectRequest final {
  std::shared_ptr<const ServiceFreeDirectProof> proof{};
  std::shared_ptr<void> lowering{};
  std::shared_ptr<const void> admission{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t nonce{};
  ServiceFreeDirectFinalCompletion final{};
  void *user{};
};

} // namespace rund::node::accel::detail
