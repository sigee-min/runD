#pragma once

#include <rund/compute/pipeline/shape.hpp>
#include <rund/compute/status.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace rund::compute::detail {

struct PipelineBuildStep;

struct OutputProjection final {
  static constexpr std::uint32_t unassigned =
      std::numeric_limits<std::uint32_t>::max();

  std::array<std::uint32_t, PipelineLeafCapacity> physical_sources{};
  std::array<std::uint32_t, PipelineLeafCapacity> logical_to_physical{};
  std::size_t physical_count{};
};

struct PipelinePlanStep final {
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
};

struct PipelineResourceAdmission final {
  static constexpr std::uint32_t none =
      std::numeric_limits<std::uint32_t>::max();

  std::uint32_t first_input_step{none};
  std::uint32_t first_full_output_step{none};
  std::uint32_t partner{none};
};

struct PhysicalOutputProjection final {
  static constexpr std::uint8_t unassigned =
      std::numeric_limits<std::uint8_t>::max();

  std::array<std::uint8_t, PipelineLeafCapacity> sources{};
};

[[nodiscard]] Result<OutputProjection>
project_outputs(const PipelineBuildStep &step);

} // namespace rund::compute::detail
