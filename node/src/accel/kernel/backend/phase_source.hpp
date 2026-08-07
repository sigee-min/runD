#pragma once

#include "phase.hpp"
#include "source_recipe.hpp"

#include <rund/compute/reason.hpp>

#include <string_view>

namespace rund::node::accel::detail {

enum class PipelineNestedPhaseSourceLanguage {
  Metal,
  Vulkan,
};

// Device sources receive named constants from the same checked host codec.
// Source files retain backend-specific algorithms, but no longer assign their
// own numeric meaning to None/Seed/Action/Fold or the Vulkan preflight bit.
template <typename Sink>
[[nodiscard]] bool EmitPipelineNestedPhaseContract(
    Sink &sink,
    const PipelineNestedPhaseSourceLanguage
        language) noexcept(noexcept(sink.append(std::string_view{}))) {
  std::string_view qualifier{};
  switch (language) {
  case PipelineNestedPhaseSourceLanguage::Metal:
    qualifier = "constant uint ";
    break;
  case PipelineNestedPhaseSourceLanguage::Vulkan:
    qualifier = "const uint ";
    break;
  default:
    return false;
  }
  const auto emit =
      [&](const std::string_view name, const std::uint32_t code) noexcept(
          noexcept(sink.append(std::string_view{}))) {
        return sink.append(qualifier) && sink.append(name) &&
               sink.append(" = ") &&
               backend_source_recipe::append_decimal(sink, code) &&
               sink.append("u;\n");
      };
  for (const rund::compute::detail::PipelineNestedPhaseContract &contract :
       rund::compute::detail::PipelineNestedPhaseContracts) {
    std::uint32_t code = 0u;
    if (!EncodePipelineNestedPhase(contract.phase, code) ||
        !sink.append(qualifier) || !sink.append("rund_pipeline_phase_") ||
        !sink.append(contract.source_key) || !sink.append(" = ") ||
        !backend_source_recipe::append_decimal(sink, code) ||
        !sink.append("u;\n")) {
      return false;
    }
  }
  if (!emit(
          "rund_pipeline_reason_invalid",
          static_cast<std::uint32_t>(rund::compute::Reason::PipelineInvalid)) ||
      (language == PipelineNestedPhaseSourceLanguage::Vulkan &&
       !emit("rund_pipeline_phase_preflight_flag",
             BackendWindowPreflightFlag)) ||
      !sink.append("bool rund_pipeline_phase_valid(uint phase) {\n  return ")) {
    return false;
  }
  bool first = true;
  for (const rund::compute::detail::PipelineNestedPhaseContract &contract :
       rund::compute::detail::PipelineNestedPhaseContracts) {
    if ((!first && !sink.append(" ||\n         ")) ||
        !sink.append("phase == rund_pipeline_phase_") ||
        !sink.append(contract.source_key)) {
      return false;
    }
    first = false;
  }
  if (!sink.append(";\n}\n")) {
    return false;
  }
  if (language != PipelineNestedPhaseSourceLanguage::Vulkan) {
    return true;
  }
  rund::compute::PipelineNestedPhase preflight_phase{};
  if (!ProjectBackendWindowPhase(BackendWindowPreflightPhase,
                                 preflight_phase)) {
    return false;
  }
  const rund::compute::detail::PipelineNestedPhaseContract *const contract =
      rund::compute::detail::pipeline_nested_phase_contract(preflight_phase);
  return contract != nullptr &&
         sink.append(
             "bool rund_pipeline_phase_parameter_decode(\n"
             "    uint parameter, out uint phase, out bool preflight) {\n"
             "  preflight =\n"
             "      (parameter & rund_pipeline_phase_preflight_flag) != "
             "0u;\n"
             "  phase =\n"
             "      parameter & ~rund_pipeline_phase_preflight_flag;\n"
             "  return rund_pipeline_phase_valid(phase) &&\n"
             "         (!preflight || phase == rund_pipeline_phase_") &&
         sink.append(contract->source_key) && sink.append(");\n}\n");
}

} // namespace rund::node::accel::detail
