#pragma once

#include "../state.hpp"

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace rund::compute::detail {

inline constexpr std::uint32_t PipelineResourceUnassigned =
    std::numeric_limits<std::uint32_t>::max();

class PipelineScheduleResources final {
public:
  explicit PipelineScheduleResources(const PipelineBuildState &build);

  [[nodiscard]] Result<std::uint32_t> admit(const PipelineBinding &binding);
  [[nodiscard]] static bool append(std::vector<resource::Access> &destination,
                                   const PipelineBinding &binding,
                                   std::uint32_t node, std::uint32_t ordinal,
                                   resource::AccessMode mode);

  const PipelineBuildState &build;
  std::vector<resource::Resource> shapes;
  std::vector<resource::Access> accesses;
  std::vector<std::uint8_t> external_flags;
  std::vector<resource::Access> publication_accesses;

private:
  std::unordered_map<const BufferState *, std::uint32_t> external_;
  std::vector<std::uint32_t> internals_;
};

} // namespace rund::compute::detail
