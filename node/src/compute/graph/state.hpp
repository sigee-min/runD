#pragma once

#include "../device/state.hpp"
#include "../expression/state.hpp"
#include "../map/step.hpp"
#include "../scan/step.hpp"
#include "../value/arena.hpp"

#include <rund/compute/ops.hpp>

#include <accel/graph/node.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace rund::compute::detail {

struct GraphValue final {
  Type type{Type::I32};
  FixedFormat fixed_format{};
  std::size_t count{};
  std::uint32_t active{};
  std::uint32_t parent{};
};

struct GraphPrimitive final {
  ValueIdRange inputs{};
  ValueIdRange outputs{};
  std::uint32_t output{};
  Primitive primitive{Primitive::Reduce};
  PrimitiveOptions options{};
  rund::AccelGraphNode node{};
  // FlowControl retains graph value IDs for the CPU runtime; node.control is
  // the lowered binding-ordinal form consumed by accelerator execution.
  FlowControl control{};
};

using GraphStep = std::variant<MapStep, ScanStep, GraphPrimitive>;

struct GraphState final {
  std::shared_ptr<DeviceState> device{};
  std::string name{};
  std::size_t count{};
  std::uint64_t authored_nodes{};
  std::vector<GraphValue> values{};
  std::vector<std::uint32_t> inputs{};
  std::vector<BoundedInputSchema> bounded_inputs{};
  ValueIdArena value_ids{};
  std::vector<GraphStep> steps{};
  std::vector<std::uint32_t> outputs{};
  std::vector<std::uint32_t> identity_outputs{};
  Status status{Status::success()};
};

} // namespace rund::compute::detail
