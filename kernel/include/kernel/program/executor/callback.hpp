#pragma once

#include <kernel/program/executor/model.hpp>

#include <type_traits>

namespace rund::kernel::skeleton_detail {

template <std::size_t Rank, typename Callback>
struct ScheduledEachContext {
  Space<Rank> index_space{};
  Alignment boundary_alignment{};
  std::remove_reference_t<Callback>* callback = nullptr;
  FailureSignal* failure_signal = nullptr;
};

template <std::size_t Rank, typename Callback>
inline void InvokeScheduledEach(void* const raw, const Partition& partition) {
  auto* const context = static_cast<ScheduledEachContext<Rank, Callback>*>(raw);
  if (context == nullptr || context->callback == nullptr ||
      context->failure_signal == nullptr) {
    return;
  }
  if (HasFailure(*context->failure_signal)) {
    return;
  }
  const SkeletonResult result = each(context->index_space,
                                     partition,
                                     context->boundary_alignment,
                                     *context->callback);
  if (!result.ok) {
    MarkFailure(*context->failure_signal, result.reason);
  }
}

} // namespace rund::kernel::skeleton_detail
