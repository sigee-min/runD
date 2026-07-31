#include <node/runtime/replay/hash.hpp>

#include "../task/stats/access.hpp"

namespace rund::node::replay_detail {

void MixSchedulerTaskSemantics(std::uint64_t &hash,
                               const task::Stats &tasks) noexcept {
#define RUND_SCHEDULER_REPLAY_SEMANTIC(storage_slot)                           \
  Mix(hash, ::rund::detail::task::StatsAccess::Value(                          \
                tasks, ::rund::detail::task::StatSlot::storage_slot));
#include <node/runtime/replay/task/schema/semantic.def>
#undef RUND_SCHEDULER_REPLAY_SEMANTIC
}

} // namespace rund::node::replay_detail
