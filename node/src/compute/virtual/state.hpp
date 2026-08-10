#pragma once

#include "../pipeline/residency/model.hpp"
#include "../pipeline/state.hpp"

#include <rund/compute/virtual.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace rund::compute::detail {

struct VirtualBufferState final {
  std::shared_ptr<VirtualBacking> backing;
  Type type{Type::I32};
  FixedFormat format{};
  std::uint64_t count{};
  std::uint64_t element_bytes{};
  std::uint64_t bytes{};
};

enum class VirtualPipelinePhase : std::uint8_t {
  Ready,
  Running,
  Poisoned,
};

struct VirtualPipelineState final {
  std::shared_ptr<VirtualBufferState> input;
  std::shared_ptr<VirtualBufferState> output;
  std::shared_ptr<PipelineState> pipeline;
  Stats stats{};
  mutable std::mutex gate;
  VirtualPipelinePhase phase{VirtualPipelinePhase::Ready};
  enum class SampleState : std::uint8_t { Inactive, Active };
  SampleState samples{SampleState::Inactive};
};

} // namespace rund::compute::detail
