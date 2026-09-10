#include "internal.hpp"

#include "../../../registry/frame.hpp"

namespace rund::compute::detail::residency::execution::authority_close_detail {

ExecutionClose apply(std::vector<Authority::Frame> &frames,
                     registry_model::ExecutionSlot &slot,
                     const Evidence &evidence,
                     const Validation &validation,
                     const Preparation &prepared) noexcept {
  ExecutionClose result{.failure = AuthorityFailure::None,
                        .progress = validation.progress,
                        .failure_count = evidence.failure_count,
                        .success = validation.successful,
                        .quarantined = !validation.successful};
  for (std::size_t index = 0u; index < evidence.failure_count; ++index) {
    result.failures[index] =
        ExecutionFailure{.epoch = evidence.failures[index].epoch,
                         .phases = evidence.failures[index].phases,
                         .may_write = evidence.failures[index].may_write,
                         .terminal = evidence.failures[index].terminal};
  }
  std::size_t failure_read = 0u;
  if (slot.cache_admitted) {
    for (std::size_t index = 0u; index < slot.undo_count; ++index) {
      frames[slot.undo_frames[index]] = slot.undo[index];
    }
  } else {
    // Opaque recurrent execution without exact HostService receipts may have
    // overwritten any reserved physical owner.
    for (std::size_t index = 0u; index < slot.frame_count; ++index) {
      const std::uint32_t frame = slot.frames[index];
      frames[frame] = frame_detail::empty(frames[frame]);
    }
  }
  if (!validation.successful) {
    const auto invalidate = [&frames](const FrameRegion region) noexcept {
      for (std::size_t frame = region.first;
           frame < static_cast<std::size_t>(region.first) + region.count;
           ++frame) {
        frames[frame] = frame_detail::empty(frames[frame]);
      }
    };
    if (evidence.terminal == TerminalKind::UnknownMayWrite) {
      for (std::size_t index = 0u; index < slot.frame_count; ++index) {
        const std::uint32_t frame = slot.frames[index];
        frames[frame] = frame_detail::empty(frames[frame]);
      }
    } else {
      for (std::size_t index = 0u; index < evidence.failure_count; ++index) {
        for (std::size_t phase = 0u; phase < 3u; ++phase) {
          if ((evidence.failures[index].may_write &
               (std::uint8_t{1u} << phase)) == 0u) {
            continue;
          }
          const Node &node =
              prepared.work.nodes[prepared.failure_nodes[failure_read++]];
          for (std::size_t mutation = 0u; mutation < node.mutation_count;
               ++mutation) {
            invalidate(node.mutations[mutation]);
          }
        }
      }
    }
    if (slot.cache_admitted && slot.native_accepted) {
      const Node &dispatch =
          prepared.work.nodes[prepared.cache_dispatch_node];
      for (std::size_t local = 0u; local < dispatch.output_count; ++local) {
        const std::uint32_t frame =
            dispatch.route.target.first + static_cast<std::uint32_t>(local);
        frames[frame] = frame_detail::empty(frames[frame]);
      }
    }
  } else if (slot.cache_admitted) {
    // Output terminal means the exact transfer and backing/staging write have
    // completed. Only now publish exact clean input keys. Output owners are
    // retired because backing is the sole post-run publication authority.
    for (std::size_t index = 0u; index < slot.service_binding_count[0u];
         ++index) {
      const CacheBinding &binding = slot.service_bindings[0u][index];
      Authority::Frame &frame = frames[binding.frame];
      frame.key = binding.key;
      frame.next_use = binding.next_use;
      frame.retain_until = binding.retain_until;
      frame.dirty = {};
      frame.state = FrameState::Resident;
    }
    const Node &dispatch =
        prepared.work.nodes[prepared.success_dispatch_node];
    const Node &output = prepared.work.nodes[prepared.success_output_node];
    for (std::size_t local = 0u; local < dispatch.output_count; ++local) {
      const std::uint32_t device =
          dispatch.route.target.first + static_cast<std::uint32_t>(local);
      const std::uint32_t host =
          output.route.target.first + static_cast<std::uint32_t>(local);
      frames[device] = frame_detail::empty(frames[device]);
      frames[host] = frame_detail::empty(frames[host]);
    }
  }
  slot = registry_model::ExecutionSlot{};
  return result;
}

} // namespace rund::compute::detail::residency::execution::authority_close_detail
