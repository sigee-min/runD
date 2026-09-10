#include "../../registry.hpp"
#include "../internal.hpp"
#include "../lease_state.hpp"
#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <span>

namespace rund::compute::detail::residency::graph_epoch_detail {

AuthorityResult Relocation::run(
    Authority &authority, const std::span<const PageUse> uses,
    const std::span<const GraphPortRequest> ports, const std::uint64_t epoch,
    AdmissionDraft &draft, registry_model::LeaseSlot &slot) noexcept {
  clear(slot);
  const auto fail = [&authority, &slot](const AuthorityFailure failure) noexcept {
    rollback(authority.frames_, slot, false);
    clear(slot);
    return AuthorityResult{.failure = failure};
  };
  try {
    const auto save_once = [&slot, &authority](const std::uint32_t frame) {
      if (std::find(slot.undo_frames.begin(), slot.undo_frames.end(), frame) ==
          slot.undo_frames.end()) {
        save(slot, frame, authority.frames_[frame]);
      }
    };
    const auto remember_relocation_frame = [&slot](const std::uint32_t frame) {
      if (std::find(slot.relocation_frames.begin(),
                    slot.relocation_frames.end(),
                    frame) == slot.relocation_frames.end()) {
        slot.relocation_frames.push_back(frame);
      }
    };
    const auto original_frame = [&slot](const std::uint32_t frame) {
      const auto found =
          std::find(slot.undo_frames.begin(), slot.undo_frames.end(), frame);
      return found == slot.undo_frames.end()
                 ? registry_model::Frame{}
                 : slot.undo[static_cast<std::size_t>(
                       found - slot.undo_frames.begin())];
    };
    struct Move final {
      std::uint32_t source{};
      std::uint32_t target{};
      bool pending{};
    };
    std::array<Move, PipelineLeafCapacity> moves{};
    for (std::size_t port_index = 0u; port_index < ports.size();
         ++port_index) {
      const GraphPortRequest &request = ports[port_index];
      std::size_t move_count = 0u;
      for (std::size_t page = 0u; page < draft.page_count; ++page) {
        const std::uint32_t source =
            draft.existing_frames[request.first_use + page];
        const std::uint32_t target =
            request.region.first + draft.locals[page];
        if (source != NoFrame && source != target) {
          moves[move_count++] =
              Move{.source = source, .target = target, .pending = true};
        }
      }
      std::size_t pending = move_count;
      while (pending != 0u) {
        bool emitted = false;
        for (std::size_t index = 0u; index < move_count; ++index) {
          if (!moves[index].pending) {
            continue;
          }
          const bool target_is_source = std::any_of(
              moves.begin(),
              moves.begin() + static_cast<std::ptrdiff_t>(move_count),
              [index, &moves](const Move &candidate) {
                return candidate.pending &&
                       candidate.source == moves[index].target;
              });
          if (target_is_source) {
            continue;
          }
          slot.relocations.push_back(GraphRelocation{
              .port = static_cast<std::uint16_t>(port_index),
              .source_frame = moves[index].source,
              .target_frame = moves[index].target,
              .bytes = request.materialization.page_bytes,
          });
          remember_relocation_frame(moves[index].source);
          remember_relocation_frame(moves[index].target);
          moves[index].pending = false;
          --pending;
          emitted = true;
        }
        if (emitted) {
          continue;
        }

        // The remaining dependency is one or more permutations. Preserve one
        // source in an evictable frame outside the executable targets, rotate
        // predecessors into their destinations, then restore the saved page.
        std::size_t saved_index = 0u;
        while (saved_index < move_count && !moves[saved_index].pending) {
          ++saved_index;
        }
        std::uint32_t scratch = NoFrame;
        for (std::size_t cache_index = 0u;
             cache_index < request.cache_region_count && scratch == NoFrame;
             ++cache_index) {
          const FrameRegion cache = request.cache_regions[cache_index];
          for (std::size_t frame_index = cache.first;
               frame_index <
               static_cast<std::size_t>(cache.first) + cache.count;
               ++frame_index) {
            const std::uint32_t candidate =
                static_cast<std::uint32_t>(frame_index);
            const bool final_target = std::any_of(
                draft.locals.begin(),
                draft.locals.begin() +
                    static_cast<std::ptrdiff_t>(draft.page_count),
                [request, candidate](const std::uint32_t local) {
                  return request.region.first + local == candidate;
                });
            const bool pending_source = std::any_of(
                moves.begin(),
                moves.begin() + static_cast<std::ptrdiff_t>(move_count),
                [candidate](const Move &move) {
                  return move.pending && move.source == candidate;
                });
            const registry_model::Frame &frame =
                authority.frames_[candidate];
            if (!final_target && !pending_source &&
                (frame.state == FrameState::Empty ||
                 (frame.state == FrameState::Resident &&
                  frame.dirty.empty() &&
                  (frame.retain_until == NeverUse ||
                   epoch > frame.retain_until)))) {
              scratch = candidate;
              break;
            }
          }
        }
        if (saved_index == move_count || scratch == NoFrame) {
          return fail(AuthorityFailure::Capacity);
        }
        save_once(scratch);
        remember_relocation_frame(scratch);
        if (authority.frames_[scratch].state != FrameState::Empty) {
          slot.transitions.push_back(CacheTransition{
              .key = authority.frames_[scratch].key,
              .frame = scratch,
              .kind = TransitionKind::Unmap,
          });
        }
        const std::uint32_t saved_source = moves[saved_index].source;
        slot.relocations.push_back(GraphRelocation{
            .port = static_cast<std::uint16_t>(port_index),
            .source_frame = saved_source,
            .target_frame = scratch,
            .bytes = request.materialization.page_bytes,
        });
        remember_relocation_frame(saved_source);
        std::uint32_t hole = saved_source;
        for (;;) {
          const auto predecessor = std::find_if(
              moves.begin(),
              moves.begin() + static_cast<std::ptrdiff_t>(move_count),
              [hole](const Move &move) {
                return move.pending && move.target == hole;
              });
          if (predecessor ==
              moves.begin() + static_cast<std::ptrdiff_t>(move_count)) {
            return fail(AuthorityFailure::Invalid);
          }
          const std::size_t index =
              static_cast<std::size_t>(predecessor - moves.begin());
          if (index == saved_index) {
            slot.relocations.push_back(GraphRelocation{
                .port = static_cast<std::uint16_t>(port_index),
                .source_frame = scratch,
                .target_frame = moves[index].target,
                .bytes = request.materialization.page_bytes,
            });
            remember_relocation_frame(moves[index].target);
            moves[index].pending = false;
            --pending;
            break;
          }
          slot.relocations.push_back(GraphRelocation{
              .port = static_cast<std::uint16_t>(port_index),
              .source_frame = moves[index].source,
              .target_frame = moves[index].target,
              .bytes = request.materialization.page_bytes,
          });
          remember_relocation_frame(moves[index].source);
          remember_relocation_frame(moves[index].target);
          hole = moves[index].source;
          moves[index].pending = false;
          --pending;
        }
      }
    }
    for (const std::uint32_t frame : slot.relocation_frames) {
      save_once(frame);
      authority.frames_[frame].state = FrameState::Pinned;
    }

    for (std::size_t port_index = 0u; port_index < ports.size();
         ++port_index) {
      const GraphPortRequest &request = ports[port_index];
      const Access access = uses[request.first_use].access;
      const std::size_t first_binding = slot.bindings.size();
      const std::size_t first_remap = slot.remaps.size();
      slot.remaps.insert(slot.remaps.end(), request.remaps.begin(),
                         request.remaps.end());
      for (std::size_t page = 0u; page < draft.page_count; ++page) {
        const CacheUse &use = draft.projected[request.first_use + page];
        const std::size_t frame = request.region.first + draft.locals[page];
        save_once(static_cast<std::uint32_t>(frame));
        const registry_model::Frame prior =
            original_frame(static_cast<std::uint32_t>(frame));
        const std::uint32_t existing =
            draft.existing_frames[request.first_use + page];
        const bool hit = existing == frame;
        const bool relocated = existing != NoFrame && !hit;
        const bool target_is_source = std::any_of(
            draft.existing_frames.begin() +
                static_cast<std::ptrdiff_t>(request.first_use),
            draft.existing_frames.begin() + static_cast<std::ptrdiff_t>(
                request.first_use + draft.page_count),
            [frame](const std::uint32_t candidate) {
              return candidate == frame;
            });
        if (!hit && !relocated) {
          if (prior.state != FrameState::Empty && !target_is_source) {
            slot.transitions.push_back(CacheTransition{
                .key = prior.key,
                .frame = static_cast<std::uint32_t>(frame),
                .kind = TransitionKind::Unmap,
            });
          }
          if (reads(access)) {
            slot.transitions.push_back(CacheTransition{
                .key = use.key,
                .frame = static_cast<std::uint32_t>(frame),
                .kind = TransitionKind::Fetch,
            });
          }
          slot.transitions.push_back(CacheTransition{
              .key = use.key,
              .frame = static_cast<std::uint32_t>(frame),
              .kind = TransitionKind::Map,
          });
          authority.frames_[frame] = registry_model::Frame{
              .key = use.key,
              .next_use = use.next_use,
              .retain_until = use.retain_until,
              .state = FrameState::Mapping,
              .tier = prior.tier,
              .role = prior.role,
              .extent = prior.extent,
              .view = prior.view,
              .assigned = true,
          };
        } else if (relocated) {
          if (prior.state != FrameState::Empty && !target_is_source) {
            slot.transitions.push_back(CacheTransition{
                .key = prior.key,
                .frame = static_cast<std::uint32_t>(frame),
                .kind = TransitionKind::Unmap,
            });
          }
          slot.transitions.push_back(CacheTransition{
              .key = use.key,
              .frame = static_cast<std::uint32_t>(frame),
              .kind = TransitionKind::Map,
          });
          authority.frames_[frame].state = FrameState::Pinned;
        } else {
          authority.frames_[frame].next_use = use.next_use;
          authority.frames_[frame].retain_until = use.retain_until;
          authority.frames_[frame].state = FrameState::Pinned;
        }
        slot.bindings.push_back(CacheBinding{
            .key = use.key,
            .frame = static_cast<std::uint32_t>(frame),
            .access = access,
            .dirty = use.dirty,
            .prior_dirty = relocated ? authority.frames_[existing].dirty
                                      : prior.dirty,
            .next_use = use.next_use,
            .retain_until = use.retain_until,
            .fetch = reads(access) && existing == NoFrame,
            .relocated = relocated,
            .retire_on_success = access == Access::Read &&
                                 use.key.domain == CacheDomain::Transient &&
                                 use.next_use == NeverUse &&
                                 use.retain_until == epoch,
        });
      }
      slot.ports.push_back(GraphLeasePort{
          .program_port = request.program_port,
          .access = access,
          .resource = request.materialization.resource,
          .region = request.region,
          .cache_regions = request.cache_regions,
          .cache_region_count = request.cache_region_count,
          .first_binding = first_binding,
          .binding_count = draft.page_count,
          .first_remap = first_remap,
          .remap_count = request.remaps.size(),
      });
    }
  } catch (const std::bad_alloc &) {
    return fail(AuthorityFailure::Capacity);
  }
  return AuthorityResult{.failure = AuthorityFailure::None};
}

} // namespace rund::compute::detail::residency::graph_epoch_detail
