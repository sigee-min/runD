#include "internal.hpp"

#include "../../../backend/token.hpp"

#include <utility>

namespace rund::node::accel::detail {

BoundRun::BoundRun(BoundRun &&other) noexcept
    : run(other.run), storage(std::move(other.storage)), ok(other.ok),
      reason(other.reason) {
  if (run.execution != nullptr) {
    run.steps = storage.data();
  }
  other.run = {};
  other.ok = false;
}

BoundRun &BoundRun::operator=(BoundRun &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  run = other.run;
  storage = std::move(other.storage);
  ok = other.ok;
  reason = other.reason;
  if (run.execution != nullptr) {
    run.steps = storage.data();
  }
  other.run = {};
  other.ok = false;
  return *this;
}

void BoundRun::bind(const KernelExecution &execution,
                    const std::uint64_t original_dispatch_count,
                    const std::uint64_t final_dispatch_count) noexcept {
  const std::shared_ptr<PickToken> &token = execution.context_admission.pick;
  run = BackendRun{
      .pick = token == nullptr ? nullptr : &token->raw,
      .ops = token == nullptr ? nullptr : token->ops,
      .execution = &execution,
      .steps = storage.data(),
      .step_count = storage.size(),
      .original_dispatch_count = original_dispatch_count,
      .final_dispatch_count = final_dispatch_count,
  };
}

} // namespace rund::node::accel::detail
