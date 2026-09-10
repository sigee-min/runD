#include "../registry.hpp"
#include "internal.hpp"
#include "lease_state.hpp"

#include "frame.hpp"
#include "release_check.hpp"

#include <algorithm>
#include <mutex>

namespace rund::compute::detail::residency {

bool Authority::complete(const std::uint64_t token, const bool success,
                         const bool invalidate_all) noexcept {
  std::lock_guard lock{gate_};
  return complete_locked(token, success, invalidate_all);
}

bool Authority::complete_locked(const std::uint64_t token, const bool success,
                                const bool invalidate_all,
                                const bool allow_forecast_quarantine) noexcept {
  if (view_state_.view_quarantine != nullptr ||
      (!allow_forecast_quarantine &&
       graph_forecast_quarantine_active_locked()) ||
      token == 0u || retry_ready(cycle_state_.graph_persists)) {
    return false;
  }
  const auto epoch = std::find_if(
      cycle_state_.epochs.begin(), cycle_state_.epochs.end(),
      [token](const LeaseSlot &slot) { return slot.token == token; });
  if (epoch != cycle_state_.epochs.end()) {
    if (epoch->cpu_key || epoch->cpu_bound) {
      return false;
    }
    return epoch->state != LeaseState::CpuQuarantined && epoch->cycle == 0u &&
           complete_epoch(frames_, *epoch, success, invalidate_all);
  }
  if (cycle_state_.writeback.token != token ||
      cycle_state_.writeback.state != LeaseState::Drain) {
    return false;
  }
  if (!success) {
    rollback(frames_, cycle_state_.writeback, false);
  } else {
    if (std::any_of(cycle_state_.writeback.transitions.begin(),
                    cycle_state_.writeback.transitions.end(),
                    [](const CacheTransition &transition) {
                      return transition.kind != TransitionKind::Writeback;
                    })) {
      return false;
    }
    for (const CacheTransition &transition :
         cycle_state_.writeback.transitions) {
      Frame &frame = frames_[transition.frame];
      if (transition.kind == TransitionKind::Writeback &&
          frame.key == transition.key) {
        frame.dirty = {};
        frame.state = FrameState::Resident;
      }
    }
  }
  clear(cycle_state_.writeback);
  return true;
}

} // namespace rund::compute::detail::residency
