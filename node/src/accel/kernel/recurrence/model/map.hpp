#pragma once

#include "../../backend/run.hpp"
#include "../../nested.hpp"
#include "../../prepared/template/registry.hpp"

#include <kernel/program/compute/limit.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace rund::node::accel::detail {

enum class MapRecurrenceState : std::uint8_t { Ineligible, Ready, Invalid };

struct MapRecurrenceHistory final {
  static constexpr std::size_t Capacity =
      static_cast<std::size_t>(rund::kernel::kMaxComputeBindingCount);
  std::array<rund::kernel::ResidentBufferRef, Capacity> outputs{};
  std::array<std::shared_ptr<void>, Capacity> handles{};
  std::array<std::uint64_t, Capacity> pitch_bytes{};
  std::uint32_t count{};

  [[nodiscard]] rund::kernel::ResidentBindingRange range() const noexcept {
    if (count == 0u || count > Capacity) {
      return {};
    }
    return rund::kernel::ResidentBindingRange{.refs = outputs.data(),
                                               .handles = handles.data(),
                                               .storage_count = count,
                                               .count = count};
  }
  [[nodiscard]] std::span<const std::uint64_t> pitches() const noexcept {
    return count == 0u || count > Capacity
               ? std::span<const std::uint64_t>{}
               : std::span<const std::uint64_t>{pitch_bytes.data(), count};
  }
};

struct MapRecurrenceSourcePlan final {
  std::uint64_t exact_source_bytes{};
  std::uint64_t source_upper_bytes{};
  std::uint64_t source_storage_upper_bytes{};
  std::uint64_t metadata_storage_upper_bytes{};
  bool history{};
  bool ok{};
  const char *reason{"compute_pipeline_recurrence_source_invalid"};
};

struct MapRecurrencePreparationPlan final {
  static constexpr std::size_t Capacity =
      static_cast<std::size_t>(rund::kernel::kMaxComputeBindingCount);

  const KernelExecutionStep *authority{};
  const rund::kernel::LoweringArtifact *canonical_artifact{};
  rund::kernel::ComputePlan plan{};
  std::array<PreparedKernelProgramBindingIdentity, Capacity> inputs{};
  std::array<PreparedKernelProgramBindingIdentity, Capacity> outputs{};
  MapRecurrenceSourcePlan terminal_source{};
  MapRecurrenceSourcePlan history_source{};
  std::uint64_t window_count{};
  std::uint64_t group_count{};
  std::uint64_t history_group_count{};
  std::uint64_t terminal_template_group_capacity{};
  std::uint64_t history_template_group_capacity{};
  std::uint64_t binding_alignment{};
  std::uint32_t input_count{};
  std::uint32_t output_count{};
  bool ok{};
  const char *reason{"compute_pipeline_recurrence_invalid"};

  [[nodiscard]] constexpr std::uint64_t terminal_group_count() const noexcept {
    return group_count >= history_group_count ? group_count - history_group_count
                                              : 0u;
  }
  [[nodiscard]] constexpr bool eligible() const noexcept {
    return ok && group_count != 0u;
  }
  [[nodiscard]] std::span<const PreparedKernelProgramBindingIdentity>
  input_layouts() const noexcept {
    return input_count > Capacity
               ? std::span<const PreparedKernelProgramBindingIdentity>{}
               : std::span<const PreparedKernelProgramBindingIdentity>{
                     inputs.data(), input_count};
  }
  [[nodiscard]] std::span<const PreparedKernelProgramBindingIdentity>
  output_layouts() const noexcept {
    return output_count > Capacity
               ? std::span<const PreparedKernelProgramBindingIdentity>{}
               : std::span<const PreparedKernelProgramBindingIdentity>{
                     outputs.data(), output_count};
  }
};

struct MapRecurrence final {
  MapRecurrenceState state = MapRecurrenceState::Ineligible;
  const BoundStep *first = nullptr;
  const BoundStep *last = nullptr;
  rund::kernel::BindingSet bindings{};
  const rund::kernel::LoweringArtifact *canonical_artifact{};
  MapRecurrenceSourcePlan source_plan{};
  rund::kernel::ComputePlan plan{};
  const rund::kernel::ComputeDispatchWindow *windows = nullptr;
  std::uint64_t window_count = 0u;
  std::uint64_t iterations = 0u;
  std::shared_ptr<const MapRecurrenceHistory> history{};
  const char *reason = "compute_pipeline_recurrence_ineligible";

  [[nodiscard]] constexpr bool ready() const noexcept {
    return state == MapRecurrenceState::Ready;
  }
  [[nodiscard]] constexpr bool invalid() const noexcept {
    return state == MapRecurrenceState::Invalid;
  }
  [[nodiscard]] bool writes_each_iteration() const noexcept {
    return history != nullptr;
  }
};

inline constexpr std::uint32_t NoTileTransducer =
    std::numeric_limits<std::uint32_t>::max();

struct TileTransducer final {
  MapRecurrence recurrence{};
  std::uint32_t template_first{};
  std::uint32_t template_count{};
};

} // namespace rund::node::accel::detail
