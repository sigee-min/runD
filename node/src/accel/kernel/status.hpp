#pragma once

#include <rund/compute/pipeline/capacity.hpp>
#include <rund/compute/reason.hpp>
#include <rund/compute/stats.hpp>

#include "profile.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>

namespace rund::node::accel::detail {

inline constexpr std::size_t PreparedPipelineStepCapacity =
    rund::compute::PipelineIterationCapacity;
inline constexpr std::uint32_t PreparedPipelineNoStep =
    std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t PreparedPipelineControlBytes = 80u;

// One immutable slice of canonical U32 Reason entries for one active Program.
struct PreparedProgramStatusSlice final {
  std::uint32_t first{};
  std::uint32_t count{};
};

// The active-to-declared projection is retained beside the packed slices so
// control evidence always names the public Pipeline declaration order, even
// when zero-work Programs do not have a backend command.
struct PreparedPipelineStatusLayout final {
  std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity> slices{};
  std::array<std::uint32_t, PreparedPipelineStepCapacity> declared_steps{};
  std::uint32_t active_step_count{};
  std::uint32_t declared_step_count{};
  std::uint32_t status_entry_count{};
  // One native control belongs to each physical stream. Transactional
  // Pipelines alternate two streams, so each stream advances by two public
  // generations; a single-stream Pipeline advances by one.
  std::uint32_t generation_stride{1u};
};

// This is the sole host/device Pipeline control schema. Device status storage
// contains raw canonical compute::Reason values; it does not define another
// failure enum or assign terminal ownership.
struct PreparedPipelineControl final {
  std::uint32_t generation{};
  std::uint32_t reason{};
  std::uint32_t failed_step{PreparedPipelineNoStep};
  std::uint32_t verified_prefix{};
  std::uint64_t generated_item_count{};
  std::uint64_t generated_capacity{};
  std::uint64_t indirect_dispatch_count{};
  std::uint64_t indirect_work_item_count{};
  std::uint64_t iteration_count{};
  std::uint64_t skipped_iteration_count{};
  std::uint64_t conflict_count{};
  std::uint64_t overflow_ordinal{std::numeric_limits<std::uint64_t>::max()};
};

static_assert(sizeof(PreparedPipelineControl) == PreparedPipelineControlBytes);
static_assert(alignof(PreparedPipelineControl) == alignof(std::uint64_t));
static_assert(std::is_standard_layout_v<PreparedPipelineControl>);
static_assert(std::is_trivially_copyable_v<PreparedPipelineControl>);
static_assert(offsetof(PreparedPipelineControl, generation) == 0u);
static_assert(offsetof(PreparedPipelineControl, reason) == 4u);
static_assert(offsetof(PreparedPipelineControl, failed_step) == 8u);
static_assert(offsetof(PreparedPipelineControl, verified_prefix) == 12u);
static_assert(offsetof(PreparedPipelineControl, generated_item_count) == 16u);
static_assert(offsetof(PreparedPipelineControl, overflow_ordinal) == 72u);

using PreparedPipelineStepControl = rund::compute::ControlStats;
inline constexpr std::uint32_t PreparedPipelineStepControlBytes =
    sizeof(PreparedPipelineStepControl);

static_assert(sizeof(PreparedPipelineStepControl) == 64u);
static_assert(alignof(PreparedPipelineStepControl) == alignof(std::uint64_t));
static_assert(std::is_standard_layout_v<PreparedPipelineStepControl>);
static_assert(std::is_trivially_copyable_v<PreparedPipelineStepControl>);
static_assert(offsetof(PreparedPipelineStepControl, generated_item_count) ==
              0u);
static_assert(offsetof(PreparedPipelineStepControl, overflow_ordinal) == 56u);

// Backend completion metadata remains evidence. The common prepared owner
// validates control and projects the one terminal AccelCheck.
struct PreparedPipelineBackendEvidence final {
  PreparedPipelineControl control{};
  PreparedPipelineProfileEvidence profile{};
  std::uint64_t control_command_count{};
  std::uint64_t control_ns{};
  bool submitted{};
  bool control_observed{};
};

struct PreparedPipelineFailureKey final {
  std::uint32_t status_ordinal{PreparedPipelineNoStep};
  std::uint32_t reason_priority{PreparedPipelineNoStep};
};

// Heterogeneous primitive status values require an explicit descriptor. Raw
// backend enum values must never be reinterpreted as canonical Reason values.
struct PreparedStatusReasonMapping final {
  std::uint32_t raw{};
  std::uint32_t reason{};
};

namespace status_detail {

struct CanonicalReason final {
  rund::compute::Reason reason{};
  std::string_view text{};
};

inline constexpr auto CanonicalReasons = std::to_array<CanonicalReason>({
#define RUND_COMPUTE_REASON(name, value, text)                                 \
  CanonicalReason{rund::compute::Reason::name, text},
#include <rund/compute/reason.def>
#undef RUND_COMPUTE_REASON
});

} // namespace status_detail

[[nodiscard]] constexpr bool
CanonicalReasonStatus(const std::uint32_t raw) noexcept {
  if (raw > std::numeric_limits<std::uint16_t>::max()) {
    return false;
  }
  const auto reason = static_cast<rund::compute::Reason>(raw);
  return reason == rund::compute::Reason::Ok ||
         rund::compute::detail::valid(reason);
}

[[nodiscard]] constexpr std::uint32_t
NextPreparedPipelineGeneration(const std::uint32_t generation) noexcept {
  return generation + std::uint32_t{1u};
}

[[nodiscard]] constexpr bool PreparedPipelineGenerationMatches(
    const PreparedPipelineControl &control,
    const std::uint64_t public_generation) noexcept {
  return control.generation == static_cast<std::uint32_t>(public_generation);
}

[[nodiscard]] constexpr const char *
CanonicalReasonText(const std::uint32_t raw) noexcept {
  if (!CanonicalReasonStatus(raw)) {
    return "compute_reason_invalid";
  }
  const auto reason = static_cast<rund::compute::Reason>(raw);
  for (const status_detail::CanonicalReason &entry :
       status_detail::CanonicalReasons) {
    if (entry.reason == reason) {
      return entry.text.data();
    }
  }
  return "compute_reason_invalid";
}

[[nodiscard]] constexpr std::uint32_t
CanonicalReasonFromText(const std::string_view text,
                        const rund::compute::Reason fallback =
                            rund::compute::Reason::BackendFailed) noexcept {
  for (const status_detail::CanonicalReason &entry :
       status_detail::CanonicalReasons) {
    if (entry.text == text) {
      return static_cast<std::uint32_t>(entry.reason);
    }
  }
  return static_cast<std::uint32_t>(rund::compute::detail::valid(fallback)
                                        ? fallback
                                        : rund::compute::Reason::BackendFailed);
}

// Priority is generated from the canonical table order and therefore does not
// depend on the numeric public error code assigned to a Reason.
[[nodiscard]] constexpr std::uint32_t
CanonicalReasonPriority(const std::uint32_t raw) noexcept {
  if (raw == static_cast<std::uint32_t>(rund::compute::Reason::Ok)) {
    return 0u;
  }
  for (std::size_t index = 0u; index < status_detail::CanonicalReasons.size();
       ++index) {
    if (static_cast<std::uint32_t>(
            status_detail::CanonicalReasons[index].reason) == raw) {
      return static_cast<std::uint32_t>(index + 1u);
    }
  }
  return PreparedPipelineNoStep;
}

[[nodiscard]] constexpr PreparedPipelineFailureKey
CanonicalPipelineFailureKey(const std::uint32_t status_ordinal,
                            const std::uint32_t raw_reason) noexcept {
  return PreparedPipelineFailureKey{
      .status_ordinal = status_ordinal,
      .reason_priority = CanonicalReasonPriority(raw_reason),
  };
}

[[nodiscard]] constexpr std::uint64_t
CanonicalPipelineFailureKeyValue(const std::uint32_t status_ordinal,
                                 const std::uint32_t raw_reason) noexcept {
  const PreparedPipelineFailureKey key =
      CanonicalPipelineFailureKey(status_ordinal, raw_reason);
  return (static_cast<std::uint64_t>(key.status_ordinal) << 32u) |
         key.reason_priority;
}

[[nodiscard]] constexpr bool ValidPreparedStatusReasonMappings(
    const std::span<const PreparedStatusReasonMapping> mappings) noexcept {
  for (std::size_t index = 0u; index < mappings.size(); ++index) {
    if (mappings[index].raw == 0u || mappings[index].reason == 0u ||
        !CanonicalReasonStatus(mappings[index].reason)) {
      return false;
    }
    for (std::size_t previous = 0u; previous < index; ++previous) {
      if (mappings[previous].raw == mappings[index].raw) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] constexpr std::uint32_t CanonicalReasonFromBackendStatus(
    const std::uint32_t raw,
    const std::span<const PreparedStatusReasonMapping> mappings) noexcept {
  if (raw == 0u) {
    return static_cast<std::uint32_t>(rund::compute::Reason::Ok);
  }
  for (const PreparedStatusReasonMapping mapping : mappings) {
    if (mapping.raw == raw && CanonicalReasonStatus(mapping.reason) &&
        mapping.reason != 0u) {
      return mapping.reason;
    }
  }
  return PreparedPipelineNoStep;
}

[[nodiscard]] constexpr bool
PreparePipelineStatusLayout(PreparedPipelineStatusLayout &layout,
                            const std::span<const std::uint32_t> declared_steps,
                            const std::uint32_t declared_step_count,
                            const std::uint32_t generation_stride) noexcept {
  layout = {};
  if (declared_steps.empty() ||
      declared_steps.size() > PreparedPipelineStepCapacity ||
      declared_step_count == 0u ||
      (generation_stride != 1u && generation_stride != 2u)) {
    return false;
  }
  std::uint32_t previous = 0u;
  for (std::size_t index = 0u; index < declared_steps.size(); ++index) {
    const std::uint32_t declared = declared_steps[index];
    if (declared >= declared_step_count ||
        (index != 0u && declared <= previous)) {
      layout = {};
      return false;
    }
    layout.declared_steps[index] = declared;
    previous = declared;
  }
  layout.active_step_count = static_cast<std::uint32_t>(declared_steps.size());
  layout.declared_step_count = declared_step_count;
  layout.generation_stride = generation_stride;
  return true;
}

[[nodiscard]] constexpr bool
SetPreparedProgramStatusSlice(PreparedPipelineStatusLayout &layout,
                              const std::uint32_t active_step,
                              const std::uint32_t entry_count) noexcept {
  if (active_step >= layout.active_step_count ||
      active_step >= PreparedPipelineStepCapacity ||
      layout.slices[active_step].count != 0u ||
      layout.slices[active_step].first != 0u ||
      entry_count > std::numeric_limits<std::uint32_t>::max() -
                        layout.status_entry_count) {
    return false;
  }
  layout.slices[active_step] = PreparedProgramStatusSlice{
      .first = layout.status_entry_count,
      .count = entry_count,
  };
  layout.status_entry_count += entry_count;
  return true;
}

[[nodiscard]] constexpr bool ValidPreparedPipelineStatusLayout(
    const PreparedPipelineStatusLayout &layout,
    const std::span<const std::uint32_t> declared_steps,
    const std::uint32_t declared_step_count,
    const std::uint32_t generation_stride) noexcept {
  if (layout.active_step_count != declared_steps.size() ||
      layout.active_step_count == 0u ||
      layout.active_step_count > PreparedPipelineStepCapacity ||
      layout.declared_step_count != declared_step_count ||
      layout.generation_stride != generation_stride ||
      (generation_stride != 1u && generation_stride != 2u) ||
      declared_step_count == 0u) {
    return false;
  }
  std::uint32_t first = 0u;
  for (std::size_t index = 0u; index < declared_steps.size(); ++index) {
    if (layout.declared_steps[index] != declared_steps[index] ||
        layout.slices[index].first != first ||
        layout.declared_steps[index] >= declared_step_count ||
        (index != 0u &&
         layout.declared_steps[index] <= layout.declared_steps[index - 1u]) ||
        layout.slices[index].count >
            std::numeric_limits<std::uint32_t>::max() - first) {
      return false;
    }
    first += layout.slices[index].count;
  }
  return first == layout.status_entry_count;
}

[[nodiscard]] constexpr bool ValidPreparedPipelineControl(
    const PreparedPipelineControl &control,
    const PreparedPipelineStatusLayout &layout) noexcept {
  if (!CanonicalReasonStatus(control.reason) ||
      control.verified_prefix > layout.declared_step_count) {
    return false;
  }
  const bool success =
      control.reason == static_cast<std::uint32_t>(rund::compute::Reason::Ok);
  if (success) {
    return control.failed_step == PreparedPipelineNoStep &&
           control.verified_prefix == layout.declared_step_count;
  }
  if (control.failed_step == PreparedPipelineNoStep) {
    return true;
  }
  if (control.failed_step >= layout.declared_step_count ||
      control.failed_step != control.verified_prefix) {
    return false;
  }
  for (std::size_t index = 0u; index < layout.active_step_count; ++index) {
    if (layout.declared_steps[index] == control.failed_step) {
      return true;
    }
  }
  return false;
}

} // namespace rund::node::accel::detail
