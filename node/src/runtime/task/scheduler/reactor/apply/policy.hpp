#pragma once

#include <cstddef>

namespace rund::node {

struct ReactorRuntime;

struct ReactorApplyBatchScope {
  ReactorRuntime* reactor = nullptr;
  bool active = false;
  bool batch_add_defer = false;

  explicit ReactorApplyBatchScope(ReactorRuntime& runtime) noexcept;
  ReactorApplyBatchScope(const ReactorApplyBatchScope&) = delete;
  ReactorApplyBatchScope& operator=(const ReactorApplyBatchScope&) = delete;
  ~ReactorApplyBatchScope();
};

[[nodiscard]] bool ReactorApplyPolicyShouldDefer(
    ReactorRuntime& reactor,
    std::size_t ready_depth,
    bool force) noexcept;

void ReactorApplyPolicyRecordFlush(ReactorRuntime& reactor,
                                   bool forced) noexcept;

void ReactorApplyPolicyClear(ReactorRuntime& reactor) noexcept;

}  // namespace rund::node
