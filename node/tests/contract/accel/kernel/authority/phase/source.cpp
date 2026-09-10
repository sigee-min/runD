#include "src/accel/kernel/backend/phase/source.hpp"
#include "src/accel/kernel/backend/source/sink.hpp"
#include "src/accel/kernel/prepared/template/registry.hpp"
#include "src/accel/kernel/recurrence.hpp"
#include "src/accel/kernel/status.hpp"
#include "src/accel/source/hash.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "src/accel/vulkan/kernel/control.hpp"
#include "src/accel/vulkan/kernel/window/source.hpp"
#endif

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "src/accel/metal/kernel/pipeline/aggregate/source.hpp"
#include "src/accel/metal/kernel/pipeline/source.hpp"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "local.hpp"

namespace node_accel_contract {

[[nodiscard]] bool GeneratedPhaseSourcesConsumeTheCodec() {
  using namespace rund::node::accel::detail;
  const auto count_occurrences = [](const std::string_view source,
                                    const std::string_view token) {
    std::size_t count = 0u;
    std::size_t cursor = 0u;
    while ((cursor = source.find(token, cursor)) != std::string_view::npos) {
      ++count;
      cursor += token.size();
    }
    return count;
  };
  [[maybe_unused]] const auto has_contract = [&](const std::string_view source,
                                                 const std::string_view
                                                     qualifier,
                                                 const bool packed_parameter) {
    for (const rund::compute::detail::PipelineNestedPhaseContract &contract :
         rund::compute::detail::PipelineNestedPhaseContracts) {
      std::uint32_t code = 0u;
      const std::string declaration =
          std::string{qualifier} + "rund_pipeline_phase_" +
          contract.source_key + " = " +
          (EncodePipelineNestedPhase(contract.phase, code)
               ? std::to_string(code)
               : std::string{"invalid"}) +
          "u;";
      if (source.find(declaration) == std::string_view::npos) {
        return false;
      }
    }
    const std::string invalid_reason =
        std::string{qualifier} + "rund_pipeline_reason_invalid = " +
        std::to_string(static_cast<std::uint32_t>(
            rund::compute::Reason::PipelineInvalid)) +
        "u;";
    constexpr std::string_view ValidBegin =
        "bool rund_pipeline_phase_valid(uint phase)";
    const std::size_t valid_begin = source.find(ValidBegin);
    const std::size_t valid_end = source.find("\n}\n", valid_begin);
    if (source.find(invalid_reason) == std::string_view::npos ||
        valid_begin == std::string_view::npos ||
        valid_end == std::string_view::npos) {
      return false;
    }
    const std::string_view valid_body =
        source.substr(valid_begin, valid_end + 3u - valid_begin);
    std::size_t projected_rows = 0u;
    for (const rund::compute::detail::PipelineNestedPhaseContract &contract :
         rund::compute::detail::PipelineNestedPhaseContracts) {
      const std::string predicate =
          std::string{"phase == rund_pipeline_phase_"} + contract.source_key;
      projected_rows +=
          valid_body.find(predicate) != std::string_view::npos ? 1u : 0u;
    }
    if (projected_rows !=
            rund::compute::detail::PipelineNestedPhaseContracts.size() ||
        count_occurrences(valid_body, "phase == rund_pipeline_phase_") !=
            rund::compute::detail::PipelineNestedPhaseContracts.size() ||
        valid_body.find("return true") != std::string_view::npos) {
      return false;
    }
    if (!packed_parameter) {
      return true;
    }
    rund::compute::PipelineNestedPhase preflight_phase{};
    if (!ProjectBackendWindowPhase(BackendWindowPreflightPhase,
                                   preflight_phase)) {
      return false;
    }
    const auto *const preflight_contract =
        rund::compute::detail::pipeline_nested_phase_contract(preflight_phase);
    constexpr std::string_view DecodeBegin =
        "bool rund_pipeline_phase_parameter_decode(";
    const std::size_t decode_begin = source.find(DecodeBegin, valid_end);
    const std::size_t decode_end = source.find("\n}\n", decode_begin);
    if (preflight_contract == nullptr ||
        decode_begin == std::string_view::npos ||
        decode_end == std::string_view::npos) {
      return false;
    }
    const std::string_view decode_body =
        source.substr(decode_begin, decode_end + 3u - decode_begin);
    const std::string decode_result =
        std::string{"return rund_pipeline_phase_valid(phase) &&\n"
                    "         (!preflight || phase == "
                    "rund_pipeline_phase_"} +
        preflight_contract->source_key + ");";
    return decode_body.find("parameter & rund_pipeline_phase_preflight_flag") !=
               std::string_view::npos &&
           decode_body.find(
               "parameter & ~rund_pipeline_phase_preflight_flag") !=
               std::string_view::npos &&
           count_occurrences(decode_body, decode_result) == 1u &&
           decode_body.find("return true") == std::string_view::npos;
  };

  std::array<char, 32u> invalid_language_storage{};
  backend_source_recipe::FixedBufferSink<32u> invalid_language_sink{
      invalid_language_storage};
  if (EmitPipelineNestedPhaseContract(
          invalid_language_sink,
          static_cast<PipelineNestedPhaseSourceLanguage>(0xffu)) ||
      !invalid_language_sink.valid() || !invalid_language_sink.text().empty()) {
    return false;
  }

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const std::string_view metal_status = MetalPipelineStatusSource();
  const std::string_view metal_aggregate = MetalNestedAggregateSource();
  if (!has_contract(metal_status, "constant uint ", false) ||
      !has_contract(metal_aggregate, "constant uint ", false) ||
      metal_status.find("const bool valid_phase = "
                        "rund_pipeline_phase_valid(params.phase);") ==
          std::string_view::npos ||
      metal_status.find("params.phase == 3u") != std::string_view::npos ||
      metal_status.find("params.phase == 2u") != std::string_view::npos ||
      metal_status.find("failed_nested_phase = 1u") != std::string_view::npos ||
      metal_aggregate.find("failed_nested_phase = 0u") !=
          std::string_view::npos ||
      metal_aggregate.find("failed_nested_phase = 1u") !=
          std::string_view::npos ||
      metal_status.find("state.stopped = params.iteration + 1u") ==
          std::string_view::npos) {
    return false;
  }
#endif

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::string_view canonical = VulkanCanonicalStatusSourceText();
  const std::string_view window = VulkanWindowSourceText();
  const std::string_view reduce = VulkanReduceStatusSourceText();
  std::uint64_t window_bytes = 0u;
  const std::size_t invalid = window.find("if (!valid_phase)");
  const std::size_t state_stop =
      window.find("states[p.state].y = p.iteration + 1u", invalid);
  const std::size_t first_barrier = window.find("  barrier();", state_stop);
  const std::size_t zero_x =
      window.find("argument_words[word] = 0u", first_barrier);
  const std::size_t zero_y =
      window.find("argument_words[word + 1u] = 0u", zero_x);
  const std::size_t zero_z =
      window.find("argument_words[word + 2u] = 0u", zero_y);
  const std::size_t second_barrier = window.find("  barrier();", zero_z);
  if (canonical.size() != 1079u ||
      SourceHash(canonical) != 3790166165241538668ull ||
      reduce.size() != 3563u || SourceHash(reduce) != 17738773106439199512ull ||
      reduce.find("if (p.phase == 2u) {\n"
                  "    if (lane == 0u) {\n"
                  "      control[1] = 0u;") == std::string_view::npos ||
      count_occurrences(reduce, "control[23]") != 1u ||
      reduce.find("control[23] = 0u;") == std::string_view::npos ||
      reduce.find("} else if (control[2] == 0xffffffffu) {\n"
                  "        control[3] = 0u;\n"
                  "      } else {\n"
                  "        control[3] = control[2];") ==
          std::string_view::npos ||
      !has_contract(window, "const uint ", true) ||
      !has_contract(reduce, "const uint ", true) ||
      window.find("const uint rund_pipeline_phase_preflight_flag = " +
                  std::to_string(BackendWindowPreflightFlag) + "u;") ==
          std::string_view::npos ||
      window.find("const bool valid_phase = "
                  "rund_pipeline_phase_parameter_decode(") ==
          std::string_view::npos ||
      window.find("raw_phase, phase, preflight);") == std::string_view::npos ||
      reduce.find("rund_pipeline_phase_valid(p.failed_nested_phase)") ==
          std::string_view::npos ||
      !VulkanWindowSourceBytes(window_bytes) || window_bytes != window.size() ||
      invalid == std::string_view::npos ||
      !(invalid < state_stop && state_stop < first_barrier &&
        first_barrier < zero_x && zero_x < zero_y && zero_y < zero_z &&
        zero_z < second_barrier) ||
      count_occurrences(window, "  barrier();") != 2u ||
      window.find("p.phase & 3u") != std::string_view::npos ||
      window.find("0x80000000u") != std::string_view::npos ||
      window.find("phase == 1u") != std::string_view::npos ||
      window.find("control[22] = 1u") != std::string_view::npos ||
      window.find("control[1] = rund_pipeline_reason_invalid") ==
          std::string_view::npos ||
      reduce.find("control[22] = 0u") != std::string_view::npos ||
      reduce.find("control[22] = rund_pipeline_phase_none") ==
          std::string_view::npos) {
    return false;
  }
#endif
  return true;
}

} // namespace node_accel_contract
