#pragma once

#include "../state.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace rund::compute::detail {

inline constexpr std::uint32_t PipelineResourceUnassigned =
    std::numeric_limits<std::uint32_t>::max();

struct PipelineResourceUseEvidence final {
  std::uint32_t first_input{resource::NoNode};
  std::uint32_t first_full_write{resource::NoNode};
};

class PipelineScheduleResources final {
public:
  explicit PipelineScheduleResources(const PipelineBuildState &build);

  [[nodiscard]] Result<std::uint32_t>
  admit(const PipelineBinding &binding, Type slot_type,
        FixedFormat slot_format);
  [[nodiscard]] Result<PipelineResolvedViewPlan>
  resolve(const PipelineBinding &binding, Type slot_type,
          FixedFormat slot_format, Location location = {});
  [[nodiscard]] Result<PipelinePublicationViewPlan>
  publication_view(const PipelineBinding &binding, Type slot_type,
                   std::size_t slot_count, FixedFormat slot_format,
                   std::optional<ResourceAccess> expected_access,
                   std::uint32_t usage, Location location = {});
  [[nodiscard]] Result<PipelinePublicationViewPlan>
  publication_view(const PipelineResolvedViewPlan &view,
                   std::uint32_t usage, Location location = {}) const;
  [[nodiscard]] Status complete_internal_resources();
  [[nodiscard]] static bool append(std::vector<resource::Access> &destination,
                                   const PipelineResolvedViewPlan &view,
                                   std::uint32_t node,
                                   resource::AccessMode mode);
  [[nodiscard]] static bool append(
      std::vector<resource::Access> &destination,
      const PipelinePublicationViewIdentity &view, std::uint32_t node);
  [[nodiscard]] static constexpr JobBufferView
  job_view(const PipelineResolvedViewPlan &view) noexcept {
    return JobBufferView{.offset = view.offset,
                         .count = view.count,
                         .stride = view.stride,
                         .element_bytes = view.element_bytes,
                         .alignment = view.alignment};
  }

  const PipelineBuildState &build;
  std::vector<resource::Resource> shapes;
  std::vector<PipelineResolvedResourcePlan> resources;
  std::vector<PipelineResourceUseEvidence> use_evidence;
  std::vector<resource::Access> accesses;
  std::vector<std::uint8_t> external_flags;
  std::vector<resource::Access> publication_accesses;

private:
  std::unordered_map<const BufferState *, std::uint32_t> external_;
  std::vector<std::uint32_t> internals_;
};

} // namespace rund::compute::detail
