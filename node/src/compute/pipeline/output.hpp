#pragma once

#include "state.hpp"

#include <rund/compute/pipeline/shape.hpp>
#include <rund/compute/status.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace rund::compute::detail {

struct OutputProjection final {
  static constexpr std::uint32_t unassigned =
      std::numeric_limits<std::uint32_t>::max();

  std::array<std::uint32_t, PipelineLeafCapacity> physical_sources{};
  std::array<std::uint32_t, PipelineLeafCapacity> logical_to_physical{};
  std::size_t physical_count{};
};

[[nodiscard]] Result<OutputProjection>
project_outputs(const PipelineBuildStep &step);

[[nodiscard]] Result<OutputProjection>
project_outputs(const ProgramState &program, std::size_t logical_outputs);

// Sole cold-build logical-to-physical-to-canonical output resolver. Both
// authored downstream routing and the sealed publication planner consume this
// projection so an aliased logical leaf cannot become a second source owner.
[[nodiscard]] Result<PipelineBuildOutputProjection>
resolve_build_output(const PipelineBuildState &build,
                     PipelineBuildOutputCoordinate coordinate);

// Sole final-bank and actual-producer-step law for one recurrence state.
[[nodiscard]] Result<PipelineBuildWindowFinal>
resolve_build_window_final(const PipelineBuildState &build,
                           PipelineBuildWindowControlOrdinal control);

[[nodiscard]] Result<PipelineBuildWindowAnchors>
resolve_build_window_anchors(const PipelineBuildState &build,
                             PipelineBuildWindowControlOrdinal control);

[[nodiscard]] Result<PipelineBuildPublicationBase>
resolve_publication_base(const PipelineBuildState &build,
                         const PipelineBuildPublicationEdge &edge);

[[nodiscard]] Result<PipelineBuildOutputCoordinate>
resolve_publication_source(const PipelineBuildState &build,
                           const PipelineBuildTerminalPublication &publication);

[[nodiscard]] Result<PipelineBuildOutputCoordinate>
resolve_publication_source(const PipelineBuildState &build,
                           const PipelineBuildWindowPublication &publication);

} // namespace rund::compute::detail
