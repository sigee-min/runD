#include "internal.hpp"

namespace rund::compute::detail::residency::execution::authority_close_detail {

Preparation prepare(const std::vector<Authority::Frame> &frames,
                    const registry_model::ExecutionSlot &slot,
                    const Plan &plan, const Evidence &evidence,
                    const bool successful) noexcept {
  Preparation result{};
  if (slot.cache_admitted) {
    result.failure = close_detail::check_close_undo(
        frames, std::span<const std::uint32_t>{slot.undo_frames},
        slot.undo_count, slot.undo.size());
  } else {
    result.failure = close_detail::check_close_indices(
        frames, std::span<const std::uint32_t>{slot.frames}, slot.frame_count);
  }
  if (result.failure == AuthorityFailure::None &&
      evidence.terminal == TerminalKind::UnknownMayWrite) {
    result.failure = close_detail::check_close_indices(
        frames, std::span<const std::uint32_t>{slot.frames}, slot.frame_count);
  }
  if (result.failure == AuthorityFailure::None && !successful &&
      evidence.terminal != TerminalKind::UnknownMayWrite) {
    for (std::size_t index = 0u;
         index < evidence.failure_count &&
         result.failure == AuthorityFailure::None;
         ++index) {
      const FailureEvidence failure = evidence.failures[index];
      for (std::size_t phase = 0u; phase < 3u; ++phase) {
        if ((failure.may_write & (std::uint8_t{1u} << phase)) == 0u) {
          continue;
        }
        result.failure = close_detail::add_close_node(
            plan, frames, result.work,
            NodeId{.epoch = failure.epoch,
                   .phase = static_cast<Phase>(phase)},
            false, true, result.failure_nodes[result.failure_node_count]);
        if (result.failure == AuthorityFailure::None) {
          ++result.failure_node_count;
        } else {
          break;
        }
      }
    }
  }
  if (result.failure == AuthorityFailure::None && !successful &&
      slot.cache_admitted && slot.native_accepted) {
    result.failure = close_detail::add_close_node(
        plan, frames, result.work,
        NodeId{.epoch = 0u, .phase = Phase::Dispatch}, true, false,
        result.cache_dispatch_node);
  }
  if (result.failure == AuthorityFailure::None && successful &&
      slot.cache_admitted) {
    result.failure = close_detail::check_close_bindings(
        frames, std::span<const CacheBinding>{slot.service_bindings[0]},
        slot.service_binding_count[0u]);
    if (result.failure == AuthorityFailure::None) {
      result.failure = close_detail::add_close_node(
          plan, frames, result.work,
          NodeId{.epoch = 0u, .phase = Phase::Dispatch}, true, false,
          result.success_dispatch_node);
    }
    if (result.failure == AuthorityFailure::None) {
      result.failure = close_detail::add_close_node(
          plan, frames, result.work,
          NodeId{.epoch = 0u, .phase = Phase::Output}, true, false,
          result.success_output_node);
    }
  }
  if (result.failure != AuthorityFailure::None) {
    return result;
  }

  const auto valid_node_index = [&result](const std::size_t value) noexcept {
    return value < result.work.count;
  };
  if (result.failure_node_count > result.failure_nodes.size() ||
      (result.cache_dispatch_node != close_detail::CloseNodeCapacity &&
       !valid_node_index(result.cache_dispatch_node)) ||
      (result.success_dispatch_node != close_detail::CloseNodeCapacity &&
       !valid_node_index(result.success_dispatch_node)) ||
      (result.success_output_node != close_detail::CloseNodeCapacity &&
       !valid_node_index(result.success_output_node))) {
    result.failure = AuthorityFailure::Invalid;
    return result;
  }
  for (std::size_t index = 0u; index < result.failure_node_count; ++index) {
    if (!valid_node_index(result.failure_nodes[index])) {
      result.failure = AuthorityFailure::Invalid;
      return result;
    }
  }
  return result;
}

} // namespace rund::compute::detail::residency::execution::authority_close_detail
