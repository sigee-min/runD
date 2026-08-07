#pragma once

#include <cstddef>

namespace rund::node {

struct ReactorRuntime;

class ReactorApplyBatchScope final {
public:
  explicit ReactorApplyBatchScope(ReactorRuntime& runtime) noexcept;
  ReactorApplyBatchScope(const ReactorApplyBatchScope&) = delete;
  ReactorApplyBatchScope& operator=(const ReactorApplyBatchScope&) = delete;
  ~ReactorApplyBatchScope();

private:
  ReactorRuntime& reactor_;
};

[[nodiscard]] bool ReactorApplyPolicyShouldDefer(
    ReactorRuntime& reactor,
    std::size_t ready_depth,
    bool force) noexcept;

void ReactorApplyPolicyRecordFlush(ReactorRuntime& reactor,
                                   bool forced) noexcept;

void ReactorApplyPolicyClear(ReactorRuntime& reactor) noexcept;

}  // namespace rund::node
