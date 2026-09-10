#pragma once

#include "../contract.hpp"
#include "../prepare.hpp"
#include "../../state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::compute::detail {

struct AdmissionDraft final {
  const PipelineBuildState &build;
  const PipelineMemoryPlan &plan;
  const PipelineBuildSnapshot &frozen;
  std::shared_ptr<PipelineState> state;
  PipelineHash hash{};
  std::vector<bool> initialized_windows;
  std::size_t resource_count{};
  std::size_t observed_bindings{};
  std::uint64_t status_entry_count{};
  std::size_t output_count{};
  std::size_t binding_capacity{};

  AdmissionDraft(const PipelineBuildState &build,
                 const PipelineMemoryPlan &plan,
                 const PipelineBuildSnapshot &frozen) noexcept;
};

[[nodiscard]] Status admit_initial(AdmissionDraft &draft);
[[nodiscard]] Status admit_steps(AdmissionDraft &draft);
[[nodiscard]] Status admit_publications(AdmissionDraft &draft);
[[nodiscard]] Status admit_state_pairs(AdmissionDraft &draft);

namespace admit {

[[nodiscard]] bool publication_matches_resolved(
    const PipelinePublicationViewPlan &publication,
    const PipelineResolvedViewPlan &resolved) noexcept;

[[nodiscard]] Result<std::uint32_t>
admit_resolved_view(const PipelineMemoryPlan &plan, const PipelineState &state,
                    const PipelineResolvedViewPlan &view, Type slot_type,
                    std::size_t slot_count, FixedFormat slot_format,
                    ResourceAccess expected_access) noexcept;

[[nodiscard]] PipelineResolvedViewPlan const *
physical_output_view(const PipelineStepResourcePlan &step,
                     std::size_t physical) noexcept;

[[nodiscard]] Result<bool> resolved_views_intersect(
    const PipelineMemoryPlan &plan, const PipelineResolvedViewPlan &left,
    resource::AccessMode left_mode, const PipelineResolvedViewPlan &right,
    resource::AccessMode right_mode) noexcept;

[[nodiscard]] Status
validate_step_aliases(const PipelineMemoryPlan &plan,
                      const PipelineStepResourcePlan &step) noexcept;

} // namespace admit
} // namespace rund::compute::detail
