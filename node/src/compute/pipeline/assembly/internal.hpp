#pragma once

#include "../state.hpp"
#include "../state/assembly.hpp"

namespace rund::compute::detail {

[[nodiscard]] PipelineBinding bind(const ResourceView &view,
                                   bool hidden = false) noexcept;

[[nodiscard]] PipelineBinding
bind(std::uint32_t owner, const PipelineInternal &resource,
     ResourceAccess access, std::size_t offset = 0u, std::size_t count = 0u,
     bool hidden = false) noexcept;

[[nodiscard]] Result<bool> intersects(const PipelineBinding &left,
                                      const ResourceView &right) noexcept;

[[nodiscard]] bool same_view(const ResourceView &left,
                             const ResourceView &right) noexcept;

[[nodiscard]] Status route(const PipelineBuildState &build,
                           const ResourceView &view,
                           PipelineBinding &result) noexcept;

void changed(PipelineBuildState &build) noexcept;

[[nodiscard]] bool has_seed(const PipelineBuildState &build) noexcept;

} // namespace rund::compute::detail
