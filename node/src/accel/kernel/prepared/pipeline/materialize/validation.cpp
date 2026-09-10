#include "../../pipeline.hpp"
#include "../../run.hpp"
#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

PreparedKernelPipeline
reject_pipeline(const PreparedPipelineFailureContext &failure,
                const char *const reason) noexcept {
  return PreparedKernelPipeline{.failure = failure.failure(reason)};
}

bool validate_pipeline_request(
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const std::uint8_t> barriers,
    const std::span<const std::uint32_t> declared_steps,
    const std::span<const BackendRecurrence> recurrences,
    const std::span<const BackendPublish> publications) noexcept {
  if (runs.empty() || runs.size() > PreparedPipelineStepCapacity ||
      runs.size() != barriers.size() || runs.size() != declared_steps.size() ||
      runs.size() != recurrences.size() || publications.size() > 32u) {
    return false;
  }
  std::uint32_t state_count = 0u;
  for (const BackendRecurrence &recurrence : recurrences) {
    if (recurrence.window != nullptr) {
      state_count = std::max(state_count, recurrence.window->state + 1u);
    }
  }
  for (const BackendPublish &publication : publications) {
    const PreparedKernelPublicationIdentity &identity = publication.identity;
    const auto &target = publication.target.source;
    const bool window = identity.kind == PreparedKernelPublicationKind::Window;
    bool recurrence_shape_matches = !window;
    if (window) {
      for (const BackendRecurrence &recurrence : recurrences) {
        const BackendWindow *const recurrence_window = recurrence.window;
        if (recurrence_window == nullptr ||
            recurrence_window->state != identity.state) {
          continue;
        }
        if (!recurrence_window->nested() ||
            recurrence_window->maximum != identity.maximum ||
            recurrence_window->tile != identity.tile ||
            recurrence_window->outer_bound != identity.outer_bound) {
          recurrence_shape_matches = false;
          break;
        }
        recurrence_shape_matches = true;
      }
    }
    if (!ValidPreparedKernelPublicationKind(identity.kind) ||
        publication.target.handle == nullptr ||
        identity.target.resource_ordinal ==
            std::numeric_limits<std::uint32_t>::max() ||
        !publication_view_matches(target, identity.target) ||
        identity.state >= state_count || !recurrence_shape_matches ||
        (!window && identity.final >= publication.sources.size()) ||
        (window &&
         (identity.maximum == 0u || identity.tile == 0u ||
          identity.tile > identity.maximum || identity.outer_bound == 0u ||
          target.count != identity.maximum ||
          publication.count.handle == nullptr ||
          identity.count.resource_ordinal ==
              std::numeric_limits<std::uint32_t>::max() ||
          !publication_view_matches(publication.count.source, identity.count) ||
          identity.count.count != 1u ||
          identity.count.element_bytes != sizeof(std::uint32_t))) ||
        (!window && (identity.maximum != 0u || identity.tile != 0u ||
                     identity.outer_bound != 0u ||
                     identity.count.resource_ordinal !=
                         std::numeric_limits<std::uint32_t>::max())) ||
        target.stride_bytes < target.element_bytes ||
        target.usage != rund::kernel::kResidentUsageWrite) {
      return false;
    }
    for (std::size_t bank = 0u; bank < publication.sources.size(); ++bank) {
      const BackendRead &read = publication.sources[bank];
      const auto &source = read.source;
      if (read.handle == nullptr ||
          identity.sources[bank].resource_ordinal ==
              std::numeric_limits<std::uint32_t>::max() ||
          !publication_view_matches(source, identity.sources[bank]) ||
          source.count != (window ? identity.tile : target.count) ||
          source.element_bytes != target.element_bytes ||
          (source.element_bytes != 4u && source.element_bytes != 8u) ||
          source.stride_bytes < source.element_bytes ||
          source.usage != rund::kernel::kResidentUsageRead) {
        return false;
      }
    }
  }
  return true;
}

} // namespace rund::node::accel::detail
