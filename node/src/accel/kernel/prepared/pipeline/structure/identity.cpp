#include "../structure.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

using ::rund::kernel::checked::add;

void fingerprint_mix(std::uint64_t &hash, const std::uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
}

void fingerprint_pipeline_header(std::uint64_t &hi, std::uint64_t &lo,
                                 const rund::AccelContext &context,
                                 const PreparedKernelPipelineShape shape,
                                 const std::uint64_t route_count) noexcept {
  hi = 0x72756e442e6c696dull;
  lo = 0x69742e70726f6772ull;
  fingerprint_mix(hi, context.id);
  fingerprint_mix(lo, static_cast<std::uint64_t>(context.api));
  fingerprint_mix(hi, shape.publication_count);
  fingerprint_mix(lo, shape.terminal_publication_count);
  fingerprint_mix(hi, shape.backend_publication_command_count);
  fingerprint_mix(hi, shape.window_state_count);
  fingerprint_mix(lo, shape.window_descriptor_state_count);
  fingerprint_mix(hi, shape.publication_fingerprint_hi);
  fingerprint_mix(lo, shape.publication_fingerprint_lo);
  fingerprint_mix(hi, shape.declared_step_count);
  fingerprint_mix(hi, shape.route_copies);
  fingerprint_mix(hi, static_cast<std::uint64_t>(shape.profile_steps));
  fingerprint_mix(lo, route_count);
}

[[nodiscard]] PreparedKernelPipelineShape
runtime_pipeline_shape(const std::span<const BackendPublish> publications,
                       const std::span<const BackendRecurrence> recurrences,
                       const std::uint32_t declared_step_count,
                       const std::uint32_t route_copies,
                       const bool profile_steps) noexcept {
  PreparedKernelPipelineShape shape{
      .publication_count = publications.size(),
      .declared_step_count = declared_step_count,
      .route_copies = route_copies,
      .profile_steps = profile_steps,
  };
  SeedPreparedKernelPublicationFingerprint(shape.publication_fingerprint_hi,
                                           shape.publication_fingerprint_lo);
  for (const BackendPublish &publication : publications) {
    const PreparedKernelPublicationIdentity &identity = publication.identity;
    const bool window = identity.kind == PreparedKernelPublicationKind::Window;
    std::uint64_t publication_commands = 0u;
    if (!PreparedKernelPublicationCommandContribution(
            identity.kind, identity.outer_bound, publication_commands) ||
        !rund::kernel::checked::add(shape.backend_publication_command_count,
                                    publication_commands,
                                    shape.backend_publication_command_count)) {
      shape.backend_publication_command_count =
          std::numeric_limits<std::uint64_t>::max();
    }
    shape.terminal_publication_count += window ? 0u : 1u;
    MixPreparedKernelPublicationFingerprint(shape.publication_fingerprint_hi,
                                            shape.publication_fingerprint_lo,
                                            identity);
  }
  for (std::size_t index = 0u; index < recurrences.size(); ++index) {
    const BackendWindow *const window = recurrences[index].window;
    if (window == nullptr) {
      continue;
    }
    shape.window_state_count =
        std::max(shape.window_state_count,
                 static_cast<std::uint64_t>(window->state) + 1u);
    bool first = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const BackendWindow *const previous = recurrences[prior].window;
      if (previous != nullptr && previous->state == window->state) {
        first = false;
        break;
      }
    }
    shape.window_descriptor_state_count += first ? 1u : 0u;
  }
  return shape;
}

[[nodiscard]] bool publication_view_matches(
    const rund::kernel::ResidentBufferRef &view,
    const PreparedKernelPublicationViewIdentity &identity) noexcept {
  return view.id != 0u && view.id == identity.resident_id &&
         view.bytes == identity.backing_bytes &&
         view.offset_bytes == identity.offset_bytes &&
         view.count == identity.count &&
         view.stride_bytes == identity.stride_bytes &&
         view.element_bytes == identity.element_bytes &&
         view.usage == identity.usage;
}

[[nodiscard]] bool
valid_publication_shape(const PreparedKernelPipelineShape &shape) noexcept {
  std::uint64_t minimum_commands = 0u;
  return shape.terminal_publication_count <= shape.publication_count &&
         rund::kernel::checked::add(shape.publication_count,
                                    shape.terminal_publication_count,
                                    minimum_commands) &&
         shape.backend_publication_command_count >= minimum_commands &&
         ((shape.publication_count == 0u) ==
          (shape.backend_publication_command_count == 0u));
}

namespace {

void fingerprint_layout(std::uint64_t &hi, std::uint64_t &lo,
                        const KernelViewLayout *const views,
                        const KernelScratchLayout *const scratch) noexcept {
  fingerprint_mix(hi, views == nullptr
                          ? std::numeric_limits<std::uint64_t>::max()
                          : views->size());
  if (views != nullptr) {
    for (const KernelViewSlot &view : *views) {
      fingerprint_mix(lo, view.binding);
      fingerprint_mix(hi, view.slot);
      fingerprint_mix(lo, view.backing_bytes);
      fingerprint_mix(hi, view.offset_bytes);
      fingerprint_mix(lo, view.count);
      fingerprint_mix(hi, view.stride_bytes);
      fingerprint_mix(lo, view.element_bytes);
      fingerprint_mix(hi, view.usage);
    }
  }
  fingerprint_mix(lo, scratch == nullptr
                          ? std::numeric_limits<std::uint64_t>::max()
                          : scratch->size());
  if (scratch != nullptr) {
    for (const KernelScratchPage &page : *scratch) {
      fingerprint_mix(hi, page.slot);
      fingerprint_mix(lo, page.bytes);
    }
  }
}

} // namespace

void fingerprint_route(
    std::uint64_t &hi, std::uint64_t &lo, const std::uint64_t kernel_id,
    const std::uint64_t graph_id_hi, const std::uint64_t graph_id_lo,
    const std::uint64_t node_count, const rund::AccelApi api,
    const std::uint64_t tile_count, const KernelViewLayout *const views,
    const KernelScratchLayout *const scratch, const std::uint64_t entry_count,
    const std::uint64_t occurrence_count, const std::uint64_t window_count,
    const std::uint64_t nested_group_count,
    const std::uint64_t map_recurrence_group_count,
    const std::uint64_t map_recurrence_history_group_count,
    const std::uint64_t recurrence_hi, const std::uint64_t recurrence_lo,
    const std::uint32_t route_copies) noexcept {
  fingerprint_mix(hi, kernel_id);
  fingerprint_mix(lo, graph_id_hi);
  fingerprint_mix(hi, graph_id_lo);
  fingerprint_mix(lo, node_count);
  fingerprint_mix(hi, static_cast<std::uint64_t>(api));
  fingerprint_mix(lo, tile_count);
  fingerprint_layout(hi, lo, views, scratch);
  fingerprint_mix(lo, entry_count);
  fingerprint_mix(hi, occurrence_count);
  fingerprint_mix(lo, window_count);
  fingerprint_mix(hi, nested_group_count);
  fingerprint_mix(lo, map_recurrence_group_count);
  fingerprint_mix(hi, map_recurrence_history_group_count);
  fingerprint_mix(lo, recurrence_hi);
  fingerprint_mix(hi, recurrence_lo);
  fingerprint_mix(lo, route_copies);
}

bool PreparedPublicationViewMatchesForContract(
    const rund::kernel::ResidentBufferRef &view,
    const PreparedKernelPublicationViewIdentity &identity) noexcept {
  return publication_view_matches(view, identity);
}

} // namespace rund::node::accel::detail
