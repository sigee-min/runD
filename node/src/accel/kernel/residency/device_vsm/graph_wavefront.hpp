#pragma once

#include "proof.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

inline constexpr std::uint32_t DeviceVsmGraphTraceOffset = 2166136261u;
inline constexpr std::uint32_t DeviceVsmGraphTracePrime = 16777619u;
inline constexpr std::uint32_t DeviceVsmGraphControllerVersion = 1u;
inline constexpr std::uint32_t DeviceVsmGraphPhysicalDispatchCount = 1u;
inline constexpr std::size_t DeviceVsmGraphTileCapacity = 64u;

struct DeviceVsmGraphTile final {
  std::uint32_t batch{};
  std::uint32_t stage{};
};

struct DeviceVsmGraphTileSchedule final {
  std::array<DeviceVsmGraphTile, DeviceVsmGraphTileCapacity> tiles{};
  std::uint32_t count{};
};

struct DeviceVsmGraphController final {
  std::uint32_t version{DeviceVsmGraphControllerVersion};
  std::uint32_t stage_count{};
  std::uint32_t frame_capacity{};
  std::uint32_t batch_count{};
  std::uint32_t map_stage{};
  std::uint32_t collective_stage{};
  std::array<std::uint32_t, DeviceVsmGraphStageCapacity> same_dispatch{};
  std::array<std::uint32_t, DeviceVsmGraphStageCapacity> same_release{};
  std::array<std::uint32_t, DeviceVsmGraphStageCapacity> prior_dispatch{};
  std::array<std::uint32_t, DeviceVsmGraphStageCapacity> prior_release{};
};

static_assert(sizeof(DeviceVsmGraphController) ==
              (6u + 4u * DeviceVsmGraphStageCapacity) * sizeof(std::uint32_t));

[[nodiscard]] inline DeviceVsmGraphController device_vsm_graph_controller(
    const DeviceVsmGraphWavefrontProof &wavefront) noexcept {
  DeviceVsmGraphController result{
      .version = DeviceVsmGraphControllerVersion,
      .stage_count = wavefront.stage_count,
      .frame_capacity = wavefront.frame_capacity,
      .batch_count = wavefront.batch_count,
      .map_stage = wavefront.map_stage,
      .collective_stage = wavefront.collective_stage,
  };
  for (std::size_t stage = 0u; stage < DeviceVsmGraphStageCapacity; ++stage) {
    result.same_dispatch[stage] = wavefront.same_dispatch[stage];
    result.same_release[stage] = wavefront.same_release[stage];
    result.prior_dispatch[stage] = wavefront.prior_dispatch[stage];
    result.prior_release[stage] = wavefront.prior_release[stage];
  }
  return result;
}

[[nodiscard]] inline bool device_vsm_graph_pointwise_topology_valid(
    const DeviceVsmGraphPointwiseTopology &topology,
    const DeviceVsmGraphWavefrontProof &wavefront) noexcept {
  if (topology.stage_count < 2u ||
      topology.stage_count > DeviceVsmGraphStageCapacity ||
      topology.stage_count != wavefront.stage_count ||
      topology.external_input_count == 0u ||
      topology.external_input_count >= DeviceVsmResidentCapacity) {
    return false;
  }
  std::uint8_t referenced_outputs = 0u;
  for (std::size_t stage = 0u; stage < topology.stage_count; ++stage) {
    const DeviceVsmGraphPointwiseStageTopology &node = topology.stages[stage];
    if (node.input_count == 0u ||
        node.input_count > DeviceVsmGraphStageInputCapacity) {
      return false;
    }
    for (std::size_t input = 0u; input < node.input_count; ++input) {
      const DeviceVsmGraphValueSource source = node.inputs[input];
      if (source.kind == DeviceVsmGraphValueSourceKind::ExternalInput) {
        if (source.index >= topology.external_input_count) {
          return false;
        }
      } else if (source.kind == DeviceVsmGraphValueSourceKind::StageOutput) {
        if (source.index >= stage ||
            ((wavefront.same_dispatch[stage] | wavefront.same_release[stage]) &
             (std::uint8_t{1u} << source.index)) == 0u) {
          return false;
        }
        referenced_outputs = static_cast<std::uint8_t>(
            referenced_outputs | (std::uint8_t{1u} << source.index));
      } else {
        return false;
      }
    }
    for (std::size_t input = node.input_count;
         input < DeviceVsmGraphStageInputCapacity; ++input) {
      if (node.inputs[input].kind != DeviceVsmGraphValueSourceKind::Invalid) {
        return false;
      }
    }
  }
  for (std::size_t stage = topology.stage_count;
       stage < DeviceVsmGraphStageCapacity; ++stage) {
    if (topology.stages[stage].input_count != 0u) {
      return false;
    }
    for (const DeviceVsmGraphValueSource source :
         topology.stages[stage].inputs) {
      if (source.kind != DeviceVsmGraphValueSourceKind::Invalid) {
        return false;
      }
    }
  }
  const std::uint8_t required = static_cast<std::uint8_t>(
      (std::uint8_t{1u} << (topology.stage_count - 1u)) - 1u);
  return (referenced_outputs & required) == required;
}

[[nodiscard]] inline bool
device_vsm_graph_wavefront_walk(const DeviceVsmGraphWavefrontProof &wavefront,
                                DeviceVsmGraphTileSchedule *const schedule,
                                std::uint32_t &steps,
                                std::uint32_t &trace) noexcept {
  steps = 0u;
  trace = DeviceVsmGraphTraceOffset;
  if (schedule != nullptr) {
    *schedule = {};
  }
  if (wavefront.stage_count < 2u ||
      wavefront.stage_count > DeviceVsmGraphStageCapacity ||
      wavefront.map_stage >= wavefront.stage_count ||
      wavefront.collective_stage >= wavefront.stage_count ||
      wavefront.map_stage == wavefront.collective_stage ||
      wavefront.frame_capacity == 0u || wavefront.batch_count == 0u) {
    return false;
  }
  if (wavefront.batch_count >
      std::numeric_limits<std::uint32_t>::max() / wavefront.stage_count) {
    return false;
  }
  const std::uint32_t all = (std::uint32_t{1u} << wavefront.stage_count) - 1u;
  for (std::uint32_t stage = 0u; stage < wavefront.stage_count; ++stage) {
    const std::uint32_t same_dispatch = wavefront.same_dispatch[stage];
    const std::uint32_t same_release = wavefront.same_release[stage];
    const std::uint32_t same = same_dispatch | same_release;
    const std::uint32_t prior_dispatch = wavefront.prior_dispatch[stage];
    const std::uint32_t prior_release = wavefront.prior_release[stage];
    const std::uint32_t earlier =
        stage == 0u ? 0u : (std::uint32_t{1u} << stage) - 1u;
    if ((same & ~earlier) != 0u || (same_dispatch & same_release) != 0u ||
        ((prior_dispatch | prior_release) & ~all) != 0u ||
        (prior_dispatch & prior_release) != 0u) {
      return false;
    }
  }
  for (std::size_t stage = wavefront.stage_count;
       stage < DeviceVsmGraphStageCapacity; ++stage) {
    if (wavefront.same_dispatch[stage] != 0u ||
        wavefront.same_release[stage] != 0u ||
        wavefront.prior_dispatch[stage] != 0u ||
        wavefront.prior_release[stage] != 0u) {
      return false;
    }
  }
  std::uint32_t previous = 0u;
  for (std::uint32_t batch = 0u; batch < wavefront.batch_count; ++batch) {
    std::uint32_t completed = 0u;
    bool map_selected = false;
    bool collective_selected = false;
    for (std::uint32_t ordinal = 0u; ordinal < wavefront.stage_count;
         ++ordinal) {
      std::uint32_t selected = wavefront.stage_count;
      for (std::uint32_t stage = 0u; stage < wavefront.stage_count; ++stage) {
        const std::uint32_t bit = std::uint32_t{1u} << stage;
        const std::uint32_t prior =
            wavefront.prior_dispatch[stage] | wavefront.prior_release[stage];
        if ((completed & bit) == 0u &&
            ((wavefront.same_dispatch[stage] | wavefront.same_release[stage]) &
             ~completed) == 0u &&
            (batch == 0u || (prior & ~previous) == 0u)) {
          selected = stage;
          break;
        }
      }
      if (selected == wavefront.stage_count ||
          steps == std::numeric_limits<std::uint32_t>::max() ||
          (schedule != nullptr &&
           schedule->count >= DeviceVsmGraphTileCapacity)) {
        steps = 0u;
        trace = 0u;
        if (schedule != nullptr) {
          *schedule = {};
        }
        return false;
      }
      if (selected == wavefront.collective_stage && !map_selected) {
        steps = 0u;
        trace = 0u;
        if (schedule != nullptr) {
          *schedule = {};
        }
        return false;
      }
      map_selected = map_selected || selected == wavefront.map_stage;
      collective_selected =
          collective_selected || selected == wavefront.collective_stage;
      completed |= std::uint32_t{1u} << selected;
      if (schedule != nullptr) {
        schedule->tiles[schedule->count++] =
            DeviceVsmGraphTile{.batch = batch, .stage = selected};
      }
      ++steps;
      const std::uint32_t coordinate =
          batch * wavefront.stage_count + selected + 1u;
      trace = (trace ^ coordinate) * DeviceVsmGraphTracePrime;
    }
    if (completed != all || !map_selected || !collective_selected) {
      steps = 0u;
      trace = 0u;
      if (schedule != nullptr) {
        *schedule = {};
      }
      return false;
    }
    previous = completed;
  }
  return true;
}

[[nodiscard]] inline bool device_vsm_graph_wavefront_expected(
    const DeviceVsmGraphWavefrontProof &wavefront, std::uint32_t &steps,
    std::uint32_t &trace) noexcept {
  return device_vsm_graph_wavefront_walk(wavefront, nullptr, steps, trace);
}

[[nodiscard]] inline bool
device_vsm_graph_tile_schedule(const DeviceVsmGraphWavefrontProof &wavefront,
                               DeviceVsmGraphTileSchedule &schedule) noexcept {
  std::uint32_t steps = 0u;
  std::uint32_t trace = 0u;
  return device_vsm_graph_wavefront_walk(wavefront, &schedule, steps, trace) &&
         schedule.count == steps;
}

[[nodiscard]] inline bool
device_vsm_graph_wavefront_valid(const DeviceVsmGraphWavefrontProof &wavefront,
                                 const std::uint64_t page_count) noexcept {
  if (page_count == 0u ||
      page_count > std::numeric_limits<std::uint32_t>::max() ||
      wavefront.frame_capacity > page_count) {
    return false;
  }
  const std::uint64_t expected_batches =
      page_count / wavefront.frame_capacity +
      static_cast<std::uint64_t>(page_count % wavefront.frame_capacity != 0u);
  std::uint32_t steps = 0u;
  std::uint32_t trace = 0u;
  return expected_batches == wavefront.batch_count &&
         device_vsm_graph_wavefront_expected(wavefront, steps, trace) &&
         steps == wavefront.stage_count * wavefront.batch_count && trace != 0u;
}

} // namespace rund::node::accel::detail
