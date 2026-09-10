#include "../window.hpp"

#include "../../registry/execution_owner.hpp"

#include "../../registry/frame.hpp"
#include "../close.hpp"

#include <algorithm>
#include <mutex>

namespace rund::compute::detail::residency {

using execution::close_detail::add_close_node;
using execution::close_detail::check_close_bindings;
using execution::close_detail::check_close_indices;
using execution::close_detail::check_close_undo;
using execution::close_detail::CloseNodeCapacity;
using execution::close_detail::CloseWork;

ExecutionClose
ExecutionOwner::close_execution(const std::uint64_t token,
                                const std::uint64_t generation,
                                const execution::Plan &plan) noexcept {
  std::lock_guard lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked() ||
      !authority_.close_rows_clear_locked() || token == 0u ||
      generation == 0u || authority_.execution_state_.slot.token != token ||
      authority_.execution_state_.slot.sliding_admitted ||
      authority_.execution_state_.slot.generation != generation ||
      authority_.execution_state_.slot.plan != plan.identity() ||
      authority_.execution_state_.slot.epochs != plan.epoch_count() ||
      authority_.execution_state_.slot.native_accepted) {
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
  }
  if (authority_.execution_state_.slot.release_count != 0u &&
      !authority_.execution_state_.slot.window_final) {
    return ExecutionClose{.failure = AuthorityFailure::Busy};
  }
  if (authority_.execution_state_.slot.failed) {
    if (authority_.execution_state_.slot.release_count == 0u) {
      for (std::size_t bank = 0u; bank < execution::BankCapacity; ++bank) {
        for (std::size_t phase = 0u; phase < 3u; ++phase) {
          if (authority_.execution_state_.slot.issued[bank][phase] !=
                  NeverUse &&
              authority_.execution_state_.slot.may_write[bank][phase] &&
              authority_.execution_state_.slot.terminals[bank][phase] !=
                  authority_.execution_state_.slot.issued[bank][phase]) {
            return ExecutionClose{.failure = AuthorityFailure::Busy};
          }
        }
      }
    }
  } else if (authority_.execution_state_.slot.native_inflight != 0u ||
             authority_.execution_state_.slot.progress.input_services !=
                 authority_.execution_state_.slot.epochs ||
             authority_.execution_state_.slot.progress.native_dispatches !=
                 authority_.execution_state_.slot.epochs ||
             authority_.execution_state_.slot.progress.native_completions !=
                 authority_.execution_state_.slot.epochs ||
             authority_.execution_state_.slot.progress.output_services !=
                 authority_.execution_state_.slot.epochs) {
    return ExecutionClose{.failure = AuthorityFailure::Busy};
  }

  CloseWork work{};
  std::array<std::size_t, 3u * ExecutionClose::FailureCapacity> failure_nodes{};
  std::size_t failure_node_count = 0u;
  std::array<std::size_t, execution::WindowCapacity> window_nodes{};
  std::size_t window_node_count = 0u;
  std::size_t cache_dispatch_node = CloseNodeCapacity;
  std::size_t success_dispatch_node = CloseNodeCapacity;
  std::size_t success_output_node = CloseNodeCapacity;
  AuthorityFailure prepared = AuthorityFailure::None;
  if (authority_.execution_state_.slot.undo_count > 0u &&
      authority_.execution_state_.slot.cache_admitted) {
    prepared =
        check_close_undo(authority_.frames_,
                         std::span<const std::uint32_t>{
                             authority_.execution_state_.slot.undo_frames},
                         authority_.execution_state_.slot.undo_count,
                         authority_.execution_state_.slot.undo.size());
  } else if (authority_.execution_state_.slot.cache_admitted &&
             authority_.execution_state_.slot.undo_count >
                 authority_.execution_state_.slot.undo_frames.size()) {
    prepared = AuthorityFailure::Invalid;
  }
  if (prepared == AuthorityFailure::None &&
      (!authority_.execution_state_.slot.cache_admitted &&
       !authority_.execution_state_.slot.window_cache_admitted)) {
    prepared = check_close_indices(
        authority_.frames_,
        std::span<const std::uint32_t>{authority_.execution_state_.slot.frames},
        authority_.execution_state_.slot.frame_count);
  }
  if (prepared == AuthorityFailure::None &&
      authority_.execution_state_.slot.failed &&
      authority_.execution_state_.slot.unknown) {
    prepared = check_close_indices(
        authority_.frames_,
        std::span<const std::uint32_t>{authority_.execution_state_.slot.frames},
        authority_.execution_state_.slot.frame_count);
  }
  if (prepared == AuthorityFailure::None &&
      authority_.execution_state_.slot.failure_count >
          authority_.execution_state_.slot.failures.size()) {
    prepared = AuthorityFailure::Invalid;
  }
  if (prepared == AuthorityFailure::None &&
      authority_.execution_state_.slot.failed &&
      !authority_.execution_state_.slot.unknown) {
    for (std::size_t failure = 0u;
         failure < authority_.execution_state_.slot.failure_count &&
         prepared == AuthorityFailure::None;
         ++failure) {
      const ExecutionFailure value =
          authority_.execution_state_.slot.failures[failure];
      for (std::size_t phase = 0u; phase < 3u; ++phase) {
        if ((value.may_write & (std::uint8_t{1u} << phase)) == 0u) {
          continue;
        }
        prepared = add_close_node(
            plan, authority_.frames_, work,
            execution::NodeId{.epoch = value.epoch,
                              .phase = static_cast<execution::Phase>(phase)},
            false, true, failure_nodes[failure_node_count]);
        if (prepared == AuthorityFailure::None) {
          ++failure_node_count;
        }
        if (prepared != AuthorityFailure::None) {
          break;
        }
      }
    }
  }
  if (prepared == AuthorityFailure::None &&
      authority_.execution_state_.slot.failed &&
      !authority_.execution_state_.slot.unknown &&
      authority_.execution_state_.slot.window_cache_admitted) {
    const std::size_t retained = static_cast<std::size_t>(
        std::min<std::uint64_t>(authority_.execution_state_.slot.release_count,
                                execution::WindowCapacity));
    const std::uint64_t first =
        authority_.execution_state_.slot.release_count - retained;
    for (std::uint64_t epoch = first;
         epoch < authority_.execution_state_.slot.release_count &&
         prepared == AuthorityFailure::None;
         ++epoch) {
      const std::size_t slot = epoch % execution::WindowCapacity;
      if (authority_.execution_state_.slot.release_epochs[slot] != epoch ||
          !authority_.execution_state_.slot.releases[slot].may_write) {
        continue;
      }
      prepared =
          add_close_node(plan, authority_.frames_, work,
                         execution::NodeId{.epoch = epoch,
                                           .phase = execution::Phase::Dispatch},
                         false, true, window_nodes[window_node_count]);
      if (prepared == AuthorityFailure::None) {
        ++window_node_count;
      }
    }
  }
  if (prepared == AuthorityFailure::None &&
      authority_.execution_state_.slot.failed &&
      authority_.execution_state_.slot.cache_admitted &&
      authority_.execution_state_.slot.native_accepted) {
    prepared = add_close_node(
        plan, authority_.frames_, work,
        execution::NodeId{.epoch = 0u, .phase = execution::Phase::Dispatch},
        true, false, cache_dispatch_node);
  }
  if (prepared == AuthorityFailure::None &&
      !authority_.execution_state_.slot.failed &&
      authority_.execution_state_.slot.cache_admitted) {
    prepared = check_close_bindings(
        authority_.frames_,
        std::span<const CacheBinding>{
            authority_.execution_state_.slot.service_bindings[0]},
        authority_.execution_state_.slot.service_binding_count[0u]);
    if (prepared == AuthorityFailure::None) {
      prepared = add_close_node(
          plan, authority_.frames_, work,
          execution::NodeId{.epoch = 0u, .phase = execution::Phase::Dispatch},
          true, false, success_dispatch_node);
    }
    if (prepared == AuthorityFailure::None) {
      prepared = add_close_node(
          plan, authority_.frames_, work,
          execution::NodeId{.epoch = 0u, .phase = execution::Phase::Output},
          true, false, success_output_node);
    }
  }
  if (prepared != AuthorityFailure::None) {
    return ExecutionClose{.failure = prepared};
  }

  const auto valid_node_index = [&work](const std::size_t value) noexcept {
    return value < work.count;
  };
  if (failure_node_count > failure_nodes.size() ||
      window_node_count > window_nodes.size() ||
      (cache_dispatch_node != CloseNodeCapacity &&
       !valid_node_index(cache_dispatch_node)) ||
      (success_dispatch_node != CloseNodeCapacity &&
       !valid_node_index(success_dispatch_node)) ||
      (success_output_node != CloseNodeCapacity &&
       !valid_node_index(success_output_node))) {
    return ExecutionClose{.failure = AuthorityFailure::Invalid};
  }
  for (std::size_t index = 0u; index < failure_node_count; ++index) {
    if (!valid_node_index(failure_nodes[index])) {
      return ExecutionClose{.failure = AuthorityFailure::Invalid};
    }
  }
  for (std::size_t index = 0u; index < window_node_count; ++index) {
    if (!valid_node_index(window_nodes[index])) {
      return ExecutionClose{.failure = AuthorityFailure::Invalid};
    }
  }

  ExecutionClose result{
      .failure = AuthorityFailure::None,
      .progress = authority_.execution_state_.slot.progress,
      .failure_count = authority_.execution_state_.slot.failure_count,
      .success = !authority_.execution_state_.slot.failed,
      .quarantined =
          authority_.execution_state_.slot.unknown ||
          std::any_of(authority_.execution_state_.slot.failures.begin(),
                      authority_.execution_state_.slot.failures.begin() +
                          static_cast<std::ptrdiff_t>(
                              authority_.execution_state_.slot.failure_count),
                      [](const ExecutionFailure failure) {
                        return failure.may_write != 0u;
                      })};
  std::copy_n(authority_.execution_state_.slot.failures.begin(),
              authority_.execution_state_.slot.failure_count,
              result.failures.begin());
  std::size_t failure_read = 0u;
  std::size_t window_read = 0u;
  if (authority_.execution_state_.slot.cache_admitted) {
    for (std::size_t index = 0u;
         index < authority_.execution_state_.slot.undo_count; ++index) {
      authority_.frames_[authority_.execution_state_.slot.undo_frames[index]] =
          authority_.execution_state_.slot.undo[index];
    }
  } else if (!authority_.execution_state_.slot.window_cache_admitted) {
    // Opaque recurrent validation has no Q1 HostService cache receipt.
    for (std::size_t index = 0u;
         index < authority_.execution_state_.slot.frame_count; ++index) {
      const std::uint32_t frame =
          authority_.execution_state_.slot.frames[index];
      authority_.frames_[frame] =
          frame_detail::empty(authority_.frames_[frame]);
    }
  } else {
    // The bounded window publishes each Host-service ticket at its exact
    // terminal. Successful Input service leaves the last page for each bank
    // resident, while successful Output service has already retired both
    // output owners. Replaying the begin snapshot or emptying the whole run
    // here would discard that sole cache authority and turn a warm window
    // into a backing/transfer miss.
  }
  const auto invalidate = [this](const FrameRegion region) noexcept {
    for (std::size_t index = region.first;
         index < static_cast<std::size_t>(region.first) + region.count;
         ++index) {
      authority_.frames_[index] =
          frame_detail::empty(authority_.frames_[index]);
    }
  };
  if (authority_.execution_state_.slot.failed &&
      authority_.execution_state_.slot.unknown) {
    for (std::size_t index = 0u;
         index < authority_.execution_state_.slot.frame_count; ++index) {
      const std::uint32_t frame =
          authority_.execution_state_.slot.frames[index];
      authority_.frames_[frame] =
          frame_detail::empty(authority_.frames_[frame]);
    }
  } else if (authority_.execution_state_.slot.failed) {
    for (std::size_t failure = 0u;
         failure < authority_.execution_state_.slot.failure_count; ++failure) {
      for (std::size_t phase = 0u; phase < 3u; ++phase) {
        if ((authority_.execution_state_.slot.failures[failure].may_write &
             (std::uint8_t{1u} << phase)) == 0u) {
          continue;
        }
        const execution::Node &node = work.nodes[failure_nodes[failure_read++]];
        for (std::size_t mutation = 0u; mutation < node.mutation_count;
             ++mutation) {
          invalidate(node.mutations[mutation]);
        }
      }
    }
    if (authority_.execution_state_.slot.window_cache_admitted) {
      // A Host Output-service failure occurs after its native Dispatch may
      // have written the Device source. The Output node itself names the Host
      // target, so also invalidate only the exact Dispatch regions whose
      // authenticated Releases report may-write.
      const std::size_t retained =
          static_cast<std::size_t>(std::min<std::uint64_t>(
              authority_.execution_state_.slot.release_count,
              execution::WindowCapacity));
      const std::uint64_t first =
          authority_.execution_state_.slot.release_count - retained;
      for (std::uint64_t epoch = first;
           epoch < authority_.execution_state_.slot.release_count; ++epoch) {
        const std::size_t slot = epoch % execution::WindowCapacity;
        if (authority_.execution_state_.slot.release_epochs[slot] != epoch ||
            !authority_.execution_state_.slot.releases[slot].may_write) {
          continue;
        }
        const execution::Node &dispatch =
            work.nodes[window_nodes[window_read++]];
        for (std::size_t mutation = 0u; mutation < dispatch.mutation_count;
             ++mutation) {
          invalidate(dispatch.mutations[mutation]);
        }
      }
    }
    if (authority_.execution_state_.slot.cache_admitted &&
        authority_.execution_state_.slot.native_accepted) {
      const execution::Node &dispatch = work.nodes[cache_dispatch_node];
      for (std::size_t local = 0u; local < dispatch.output_count; ++local) {
        const std::uint32_t frame =
            dispatch.route.target.first + static_cast<std::uint32_t>(local);
        authority_.frames_[frame] =
            frame_detail::empty(authority_.frames_[frame]);
      }
    }
  } else if (authority_.execution_state_.slot.cache_admitted) {
    // Focused granular success is allowed only after all service terminals.
    for (std::size_t index = 0u;
         index < authority_.execution_state_.slot.service_binding_count[0u];
         ++index) {
      const CacheBinding &binding =
          authority_.execution_state_.slot.service_bindings[0u][index];
      if (binding.key.backing == 0u) {
        continue;
      }
      ExecutionFrame &frame = authority_.frames_[binding.frame];
      frame.key = binding.key;
      frame.next_use = binding.next_use;
      frame.retain_until = binding.retain_until;
      frame.dirty = {};
      frame.state = FrameState::Resident;
    }
    const execution::Node &dispatch = work.nodes[success_dispatch_node];
    const execution::Node &output = work.nodes[success_output_node];
    for (std::size_t local = 0u; local < dispatch.output_count; ++local) {
      const std::uint32_t device =
          dispatch.route.target.first + static_cast<std::uint32_t>(local);
      const std::uint32_t host =
          output.route.target.first + static_cast<std::uint32_t>(local);
      authority_.frames_[device] =
          frame_detail::empty(authority_.frames_[device]);
      authority_.frames_[host] = frame_detail::empty(authority_.frames_[host]);
    }
  }
  authority_.execution_state_.slot = registry_model::ExecutionSlot{};
  return result;
}

} // namespace rund::compute::detail::residency
