#include "internal.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>

namespace rund::node::accel::detail::prepared_residency {

[[nodiscard]] bool SameCheck(const rund::AccelCheck left,
                             const rund::AccelCheck right) noexcept {
  return left.ok == right.ok && left.reason != nullptr &&
         right.reason != nullptr && std::strcmp(left.reason, right.reason) == 0;
}

[[nodiscard]] bool
SameWindowReceipt(const BackendResidencyWindowReceipt &left,
                  const BackendResidencyWindowReceipt &right) noexcept {
  return SameCheck(left.check, right.check) &&
         left.terminal == right.terminal && left.epoch == right.epoch &&
         left.backend_sequence == right.backend_sequence &&
         left.bank == right.bank && left.dispatched == right.dispatched &&
         left.completed == right.completed && left.may_write == right.may_write;
}

[[nodiscard]] std::size_t WindowStates(
    const PreparedResidencyWindowRequest &request,
    std::array<prepared::PipelineState *, ResidencyWindowCapacity> &states,
    const rund::AccelContext &context, const BackendOps *&ops) noexcept {
  states.fill(nullptr);
  ops = nullptr;
  if (request.plan_identity == 0u || request.token == 0u ||
      request.generation == 0u || request.batch_count == 0u ||
      request.batch_count > ResidencyWindowCapacity ||
      request.release == nullptr || request.final == nullptr ||
      request.user == nullptr) {
    return 0u;
  }
  std::size_t state_count = 0u;
  for (std::size_t index = 0u; index < request.batch_count; ++index) {
    const PreparedResidencyWindowBatch &batch = request.batches[index];
    auto *const state =
        static_cast<prepared::PipelineState *>(batch.pipeline.owner.get());
    if (!batch.pipeline.ok || state == nullptr ||
        !prepared::ValidPipeline(context, *state) || state->ops == nullptr ||
        state->ops->submit_prepared_window == nullptr ||
        state->ops->signal_prepared_window == nullptr ||
        state->ops->abort_prepared_window == nullptr ||
        request.first_epoch >
            std::numeric_limits<std::uint64_t>::max() - index ||
        batch.epoch != request.first_epoch + index ||
        batch.bank != batch.epoch % 2u || batch.control_generation == 0u ||
        batch.local_count == 0u ||
        batch.local_count > ResidencyWindowLocalCapacity ||
        (ops != nullptr && ops != state->ops)) {
      return 0u;
    }
    for (std::size_t local = 0u; local < batch.local_count; ++local) {
      for (std::size_t prior = 0u; prior < local; ++prior) {
        if (batch.locals[local] == batch.locals[prior]) {
          return 0u;
        }
      }
    }
    ops = state->ops;
    bool found = false;
    for (std::size_t prior = 0u; prior < state_count; ++prior) {
      found = found || states[prior] == state;
    }
    if (!found) {
      states[state_count++] = state;
    }
  }
  std::sort(states.begin(), states.begin() + state_count,
            std::less<prepared::PipelineState *>{});
  return state_count;
}

} // namespace rund::node::accel::detail::prepared_residency
