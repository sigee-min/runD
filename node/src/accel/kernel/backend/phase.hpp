#pragma once

#include <rund/compute/pipeline/coordinate.hpp>

#include <cstdint>
#include <type_traits>

namespace rund::node::accel::detail {

// Public diagnostics, backend parameters, prepared identity, and generated
// device source all consume this one numeric contract. The public values are
// installed ABI and therefore remain explicit and stable; the backend type is
// distinct so topology phases cannot be ordinal-cast into a native parameter.
inline constexpr std::uint32_t PipelineNestedPhaseNoneCode =
    static_cast<std::uint8_t>(rund::compute::PipelineNestedPhase::None);
inline constexpr std::uint32_t PipelineNestedPhaseSeedCode =
    static_cast<std::uint8_t>(rund::compute::PipelineNestedPhase::Seed);
inline constexpr std::uint32_t PipelineNestedPhaseActionCode =
    static_cast<std::uint8_t>(rund::compute::PipelineNestedPhase::Action);
inline constexpr std::uint32_t PipelineNestedPhaseFoldCode =
    static_cast<std::uint8_t>(rund::compute::PipelineNestedPhase::Fold);

enum class BackendWindowPhase : std::uint8_t {
  Ordinary = PipelineNestedPhaseNoneCode,
  NestedSeed = PipelineNestedPhaseSeedCode,
  NestedAction = PipelineNestedPhaseActionCode,
  NestedFold = PipelineNestedPhaseFoldCode,
};

static_assert(
    std::is_same_v<std::underlying_type_t<BackendWindowPhase>, std::uint8_t>);

[[nodiscard]] constexpr bool
EncodePipelineNestedPhase(const rund::compute::PipelineNestedPhase phase,
                          std::uint32_t &code) noexcept {
  if (rund::compute::detail::pipeline_nested_phase_contract(phase) == nullptr) {
    return false;
  }
  code = static_cast<std::uint8_t>(phase);
  return true;
}

[[nodiscard]] constexpr bool
DecodePipelineNestedPhase(const std::uint32_t code,
                          rund::compute::PipelineNestedPhase &phase) noexcept {
  for (const rund::compute::detail::PipelineNestedPhaseContract &contract :
       rund::compute::detail::PipelineNestedPhaseContracts) {
    if (code == static_cast<std::uint8_t>(contract.phase)) {
      phase = contract.phase;
      return true;
    }
  }
  return false;
}

[[nodiscard]] constexpr bool
EncodeBackendWindowPhase(const BackendWindowPhase phase,
                         std::uint32_t &code) noexcept {
  const std::uint32_t candidate = static_cast<std::uint8_t>(phase);
  rund::compute::PipelineNestedPhase public_phase{};
  if (!DecodePipelineNestedPhase(candidate, public_phase)) {
    return false;
  }
  code = candidate;
  return true;
}

[[nodiscard]] constexpr bool
DecodeBackendWindowPhase(const std::uint32_t code,
                         BackendWindowPhase &phase) noexcept {
  rund::compute::PipelineNestedPhase public_phase{};
  if (!DecodePipelineNestedPhase(code, public_phase)) {
    return false;
  }
  phase =
      static_cast<BackendWindowPhase>(static_cast<std::uint8_t>(public_phase));
  return true;
}

[[nodiscard]] constexpr bool
ProjectBackendWindowPhase(const BackendWindowPhase backend,
                          rund::compute::PipelineNestedPhase &phase) noexcept {
  std::uint32_t code = 0u;
  if (!EncodeBackendWindowPhase(backend, code)) {
    return false;
  }
  return DecodePipelineNestedPhase(code, phase);
}

[[nodiscard]] constexpr bool
BackendWindowPhaseIsNested(const BackendWindowPhase phase) noexcept {
  rund::compute::PipelineNestedPhase public_phase{};
  return ProjectBackendWindowPhase(phase, public_phase) &&
         rund::compute::pipeline_nested_phase_has_outer(public_phase);
}

// Vulkan carries one preflight discriminator beside the phase in a U32 push
// parameter. No other flag bits are admitted and preflight is meaningful only
// for Seed. This prevents an unknown value from being normalized with `& 3`.
inline constexpr std::uint32_t BackendWindowPreflightFlag = 1u << 31u;
inline constexpr BackendWindowPhase BackendWindowPreflightPhase =
    BackendWindowPhase::NestedSeed;

[[nodiscard]] constexpr bool
BackendWindowPreflightValid(const BackendWindowPhase phase,
                            const bool preflight) noexcept {
  return !preflight || phase == BackendWindowPreflightPhase;
}

[[nodiscard]] constexpr bool
EncodeBackendWindowParameter(const BackendWindowPhase phase,
                             const bool preflight,
                             std::uint32_t &parameter) noexcept {
  std::uint32_t code = 0u;
  if (!EncodeBackendWindowPhase(phase, code) ||
      !BackendWindowPreflightValid(phase, preflight)) {
    return false;
  }
  parameter = code | (preflight ? BackendWindowPreflightFlag : 0u);
  return true;
}

[[nodiscard]] constexpr bool
DecodeBackendWindowParameter(const std::uint32_t parameter,
                             BackendWindowPhase &phase,
                             bool &preflight) noexcept {
  BackendWindowPhase decoded{};
  const bool decoded_preflight = (parameter & BackendWindowPreflightFlag) != 0u;
  const std::uint32_t code = parameter & ~BackendWindowPreflightFlag;
  if (!DecodeBackendWindowPhase(code, decoded) ||
      !BackendWindowPreflightValid(decoded, decoded_preflight)) {
    return false;
  }
  phase = decoded;
  preflight = decoded_preflight;
  return true;
}

} // namespace rund::node::accel::detail
