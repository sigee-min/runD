#include "src/accel/kernel/backend/phase/source.hpp"
#include "src/accel/kernel/prepared/template/registry.hpp"
#include "src/accel/kernel/recurrence.hpp"
#include "src/accel/kernel/status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "phase.hpp"
#include "phase/local.hpp"

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

// The GPU admission gate has no declared failing step. Final must not project
// that sentinel as a verified-prefix count, and reserved is never a reason.
[[nodiscard]] constexpr bool AdmissionFailureControlIsCanonical() noexcept {
  using namespace rund::node::accel::detail;
  PreparedPipelineStatusLayout layout{};
  layout.declared_step_count = 3u;
  PreparedPipelineControl control{};
  control.reason =
      static_cast<std::uint32_t>(rund::compute::Reason::PipelineInvalid);
  if (!ValidPreparedPipelineControl(control, layout)) {
    return false;
  }
  control.verified_prefix = PreparedPipelineNoStep;
  if (ValidPreparedPipelineControl(control, layout)) {
    return false;
  }
  control.verified_prefix = 0u;
  control.reserved = control.reason;
  return !ValidPreparedPipelineControl(control, layout);
}

static_assert(AdmissionFailureControlIsCanonical());

} // namespace

bool PhaseSourceIdentityContract() {
  return PreparedPipelineNestedCoordinateShapesAreCanonical() &&
         PipelineNestedPhaseCodecIsExact() &&
         PreparedRecurrencePhaseIdentityPreservesItsABI() &&
         PreparedControlPhaseCodesAreChecked() &&
         GeneratedPhaseSourcesConsumeTheCodec();
}

} // namespace node_accel_contract
