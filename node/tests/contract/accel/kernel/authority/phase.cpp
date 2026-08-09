#include "src/accel/kernel/backend/phase_source.hpp"
#include "src/accel/kernel/backend/source_recipe.hpp"
#include "src/accel/kernel/prepared/template_registry.hpp"
#include "src/accel/kernel/recurrence.hpp"
#include "src/accel/kernel/status.hpp"
#include "src/accel/source/hash.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "src/accel/vulkan/kernel/control.hpp"
#include "src/accel/vulkan/kernel/window.hpp"
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
#include <type_traits>

#include "phase.hpp"

namespace node_accel_contract {

namespace {

[[nodiscard]] constexpr bool
PreparedPipelineNestedCoordinateShapesAreCanonical() noexcept {
  using rund::compute::PipelineNestedPhase;
  using rund::compute::valid_pipeline_nested_coordinate;

  constexpr std::array<PipelineNestedPhase, 4u> phases{
      PipelineNestedPhase::None,
      PipelineNestedPhase::Seed,
      PipelineNestedPhase::Action,
      PipelineNestedPhase::Fold,
  };
  constexpr std::array<bool, 2u> known_values{false, true};
  for (const PipelineNestedPhase phase : phases) {
    for (const bool outer_known : known_values) {
      for (const bool inner_known : known_values) {
        const bool expected =
            outer_known ==
                rund::compute::pipeline_nested_phase_has_outer(phase) &&
            inner_known ==
                rund::compute::pipeline_nested_phase_has_inner(phase);
        if (valid_pipeline_nested_coordinate(phase, outer_known, inner_known) !=
            expected) {
          return false;
        }
      }
    }
  }

  constexpr PipelineNestedPhase invalid =
      static_cast<PipelineNestedPhase>(0xffu);
  if (rund::compute::pipeline_nested_phase_valid(invalid)) {
    return false;
  }
  for (const bool outer_known : known_values) {
    for (const bool inner_known : known_values) {
      if (valid_pipeline_nested_coordinate(invalid, outer_known, inner_known)) {
        return false;
      }
    }
  }
  return true;
}

static_assert(PreparedPipelineNestedCoordinateShapesAreCanonical());

[[nodiscard]] constexpr bool PipelineNestedPhaseCodecIsExact() noexcept {
  using namespace rund::node::accel::detail;
  using rund::compute::PipelineNestedPhase;
  static_assert(std::is_same_v<std::underlying_type_t<PipelineNestedPhase>,
                               std::uint8_t>);
  static_assert(static_cast<std::uint8_t>(PipelineNestedPhase::None) == 0u);
  static_assert(static_cast<std::uint8_t>(PipelineNestedPhase::Seed) == 1u);
  static_assert(static_cast<std::uint8_t>(PipelineNestedPhase::Action) == 2u);
  static_assert(static_cast<std::uint8_t>(PipelineNestedPhase::Fold) == 3u);

  struct PhaseCase final {
    PipelineNestedPhase public_phase;
    BackendWindowPhase backend_phase;
    std::uint32_t code;
  };
  constexpr std::array<PhaseCase, 4u> cases{{
      {PipelineNestedPhase::None, BackendWindowPhase::Ordinary, 0u},
      {PipelineNestedPhase::Seed, BackendWindowPhase::NestedSeed, 1u},
      {PipelineNestedPhase::Action, BackendWindowPhase::NestedAction, 2u},
      {PipelineNestedPhase::Fold, BackendWindowPhase::NestedFold, 3u},
  }};
  for (const PhaseCase item : cases) {
    std::uint32_t public_code = 99u;
    std::uint32_t backend_code = 99u;
    PipelineNestedPhase public_phase = PipelineNestedPhase::Fold;
    BackendWindowPhase backend_phase = BackendWindowPhase::NestedFold;
    PipelineNestedPhase projected = PipelineNestedPhase::Fold;
    if (!EncodePipelineNestedPhase(item.public_phase, public_code) ||
        public_code != item.code ||
        !DecodePipelineNestedPhase(item.code, public_phase) ||
        public_phase != item.public_phase ||
        !EncodeBackendWindowPhase(item.backend_phase, backend_code) ||
        backend_code != item.code ||
        !DecodeBackendWindowPhase(item.code, backend_phase) ||
        backend_phase != item.backend_phase ||
        !ProjectBackendWindowPhase(item.backend_phase, projected) ||
        projected != item.public_phase) {
      return false;
    }
  }

  std::uint32_t code = 91u;
  PipelineNestedPhase public_phase = PipelineNestedPhase::Action;
  BackendWindowPhase backend_phase = BackendWindowPhase::NestedAction;
  PipelineNestedPhase projected = PipelineNestedPhase::Action;
  if (EncodePipelineNestedPhase(static_cast<PipelineNestedPhase>(0xffu),
                                code) ||
      code != 91u || DecodePipelineNestedPhase(4u, public_phase) ||
      public_phase != PipelineNestedPhase::Action ||
      DecodePipelineNestedPhase(0xffu, public_phase) ||
      public_phase != PipelineNestedPhase::Action ||
      EncodeBackendWindowPhase(static_cast<BackendWindowPhase>(0xffu), code) ||
      code != 91u || DecodeBackendWindowPhase(4u, backend_phase) ||
      backend_phase != BackendWindowPhase::NestedAction ||
      DecodeBackendWindowPhase(0xffu, backend_phase) ||
      backend_phase != BackendWindowPhase::NestedAction ||
      ProjectBackendWindowPhase(static_cast<BackendWindowPhase>(0xffu),
                                projected) ||
      projected != PipelineNestedPhase::Action) {
    return false;
  }

  BackendWindowPhase topology_phase = BackendWindowPhase::Ordinary;
  std::uint32_t topology_code = 0u;
  if (static_cast<std::uint8_t>(NestedTemplatePhase::Seed) != 0u ||
      !ProjectNestedBackendWindowPhase(NestedTemplatePhase::Seed,
                                       topology_phase) ||
      topology_phase != BackendWindowPhase::NestedSeed ||
      !EncodeBackendWindowPhase(topology_phase, topology_code) ||
      topology_code != PipelineNestedPhaseSeedCode) {
    return false;
  }
  topology_phase = BackendWindowPhase::NestedAction;
  if (ProjectNestedBackendWindowPhase(static_cast<NestedTemplatePhase>(0xffu),
                                      topology_phase) ||
      topology_phase != BackendWindowPhase::NestedAction) {
    return false;
  }
  std::uint32_t parameter = 77u;
  bool preflight = false;
  backend_phase = BackendWindowPhase::NestedFold;
  if (!EncodeBackendWindowParameter(BackendWindowPhase::NestedSeed, true,
                                    parameter) ||
      parameter != (PipelineNestedPhaseSeedCode | BackendWindowPreflightFlag) ||
      !DecodeBackendWindowParameter(parameter, backend_phase, preflight) ||
      backend_phase != BackendWindowPhase::NestedSeed || !preflight) {
    return false;
  }
  constexpr std::array<std::uint32_t, 4u> invalid_parameters{
      PipelineNestedPhaseActionCode | BackendWindowPreflightFlag,
      0x40000000u | PipelineNestedPhaseSeedCode,
      4u,
      0xffu,
  };
  for (const std::uint32_t invalid : invalid_parameters) {
    backend_phase = BackendWindowPhase::NestedFold;
    preflight = true;
    if (DecodeBackendWindowParameter(invalid, backend_phase, preflight) ||
        backend_phase != BackendWindowPhase::NestedFold || !preflight) {
      return false;
    }
  }
  parameter = 77u;
  return !EncodeBackendWindowParameter(BackendWindowPhase::NestedAction, true,
                                       parameter) &&
         parameter == 77u;
}

static_assert(PipelineNestedPhaseCodecIsExact());

[[nodiscard]] bool PreparedRecurrencePhaseIdentityPreservesItsABI() {
  using namespace rund::node::accel::detail;
  static_assert(
      std::is_same_v<decltype(PreparedKernelRecurrenceIdentity::phase),
                     BackendWindowPhase>);
  static_assert(std::is_standard_layout_v<PreparedKernelRecurrenceIdentity>);
  static_assert(std::is_trivially_copyable_v<PreparedKernelRecurrenceIdentity>);

  BackendWindow invalid_window{};
  invalid_window.phase = static_cast<BackendWindowPhase>(0xffu);
  rund::compute::PipelineNestedPhase projected =
      rund::compute::PipelineNestedPhase::Action;
  if (invalid_window.nested_phase(projected) ||
      projected != rund::compute::PipelineNestedPhase::Action) {
    return false;
  }

  struct FingerprintCase final {
    BackendWindowPhase phase;
    std::uint64_t hi;
  };
  constexpr std::array<FingerprintCase, 4u> cases{{
      {BackendWindowPhase::Ordinary, 0xb0f91e21aa126112ull},
      {BackendWindowPhase::NestedSeed, 0xb0f91e21aa126111ull},
      {BackendWindowPhase::NestedAction, 0xb0f91e21aa126110ull},
      {BackendWindowPhase::NestedFold, 0xb0f91e21aa126117ull},
  }};
  for (const FingerprintCase item : cases) {
    const PreparedKernelRecurrenceIdentity identity{
        .logical_step = 7u,
        .iteration = 2u,
        .bound = 5u,
        .maximum = 17u,
        .tile = 4u,
        .expected = 9u,
        .outer_iteration = 3u,
        .outer_bound = 5u,
        .inner_iteration = 1u,
        .inner_bound = 2u,
        .route = 2u,
        .state = 6u,
        .phase = item.phase,
        .writes_each_iteration = true,
        .has_window = true,
        .has_terminal = true,
    };
    std::uint64_t hi = 0u;
    std::uint64_t lo = 0u;
    SeedPreparedKernelRecurrenceFingerprint(hi, lo);
    if (!MixPreparedKernelRecurrenceFingerprint(hi, lo, identity) ||
        hi != item.hi || lo != 0x6ddd041cd8f83448ull) {
      return false;
    }
  }

  PreparedKernelRecurrenceIdentity invalid{
      .phase = static_cast<BackendWindowPhase>(0xffu),
      .has_window = true,
  };
  std::uint64_t hi = 0x1234u;
  std::uint64_t lo = 0x5678u;
  if (MixPreparedKernelRecurrenceFingerprint(hi, lo, invalid) ||
      hi != 0x1234u || lo != 0x5678u) {
    return false;
  }
  invalid.has_window = false;
  hi = 0x9abcu;
  lo = 0xdef0u;
  return !MixPreparedKernelRecurrenceFingerprint(hi, lo, invalid) &&
         hi == 0x9abcu && lo == 0xdef0u;
}

[[nodiscard]] constexpr bool PreparedControlPhaseCodesAreChecked() noexcept {
  using namespace rund::node::accel::detail;
  PreparedPipelineStatusLayout layout{};
  layout.slices[0] = PreparedProgramStatusSlice{.first = 0u, .count = 1u};
  layout.declared_steps[0] = 0u;
  layout.active_step_count = 1u;
  layout.command_count = 1u;
  layout.declared_step_count = 1u;
  layout.status_entry_count = 1u;

  PreparedPipelineControl control{};
  control.reason = static_cast<std::uint32_t>(rund::compute::Reason::Ok);
  control.verified_prefix = 1u;
  if (!ValidPreparedPipelineControl(control, layout)) {
    return false;
  }
  control.failed_nested_phase = PipelineNestedPhaseSeedCode;
  control.failed_outer_window = 0u;
  if (ValidPreparedPipelineControl(control, layout)) {
    return false;
  }
  control.reason =
      static_cast<std::uint32_t>(rund::compute::Reason::PipelineInvalid);
  control.failed_step = 0u;
  control.verified_prefix = 0u;
  struct CoordinateCase final {
    std::uint32_t phase;
    bool outer;
    bool inner;
  };
  constexpr std::array<CoordinateCase, 4u> cases{{
      {PipelineNestedPhaseNoneCode, false, false},
      {PipelineNestedPhaseSeedCode, true, false},
      {PipelineNestedPhaseActionCode, true, true},
      {PipelineNestedPhaseFoldCode, true, false},
  }};
  for (const CoordinateCase item : cases) {
    control.failed_nested_phase = item.phase;
    control.failed_outer_window = item.outer ? 3u : PreparedPipelineNoStep;
    control.failed_inner_iteration = item.inner ? 2u : PreparedPipelineNoStep;
    if (!ValidPreparedPipelineControl(control, layout)) {
      return false;
    }
    control.failed_inner_iteration = item.inner ? PreparedPipelineNoStep : 2u;
    if (ValidPreparedPipelineControl(control, layout)) {
      return false;
    }
  }
  control.failed_outer_window = PreparedPipelineNoStep;
  control.failed_inner_iteration = PreparedPipelineNoStep;
  control.failed_nested_phase = 4u;
  if (ValidPreparedPipelineControl(control, layout)) {
    return false;
  }
  control.failed_nested_phase = 0xffu;
  return !ValidPreparedPipelineControl(control, layout);
}

static_assert(PreparedControlPhaseCodesAreChecked());

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
      reduce.size() != 3492u || SourceHash(reduce) != 16305988097943134686ull ||
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

} // namespace

bool PhaseSourceIdentityContract() {
  return PreparedPipelineNestedCoordinateShapesAreCanonical() &&
         PipelineNestedPhaseCodecIsExact() &&
         PreparedRecurrencePhaseIdentityPreservesItsABI() &&
         PreparedControlPhaseCodesAreChecked() &&
         GeneratedPhaseSourcesConsumeTheCodec();
}

} // namespace node_accel_contract
