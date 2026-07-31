#pragma once

#include "../map/step.hpp"
#include "../scan/step.hpp"
#include "../value/arena.hpp"

#include <rund/compute/abi/flow.hpp>
#include <rund/compute/ops.hpp>

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

namespace rund::compute {
template <class> class Program;
}

namespace rund::compute::detail {

struct FlowValue final {
  Type type{Type::I32};
  FixedFormat fixed_format{};
  std::size_t count{};
  std::uint32_t guard{};
  std::uint32_t active{};
  std::uint32_t parent{};
};

struct FlowPrimitive final {
  ValueIdRange inputs{};
  ValueIdRange outputs{};
  Primitive operation{Primitive::Reduce};
  PrimitiveOptions options{};
  FlowControl control{};
};

using FlowStep = std::variant<MapStep, ScanStep, FlowPrimitive>;

struct FlowState final {
  Target target{Target::cpu()};
  std::shared_ptr<DeviceState> device;
  std::shared_ptr<ProgramCacheState> cache;
  std::vector<FlowValue> values;
  std::vector<std::uint32_t> inputs;
  std::vector<BoundedInputSchema> bounded_inputs;
  std::vector<HostView> bindings;
  ValueIdArena value_ids;
  std::vector<FlowStep> steps;
  std::uint32_t output{};
  std::vector<std::uint32_t> outputs;
  std::vector<std::uint32_t> logical_outputs;
  Status status{Status::success()};
};

struct FlowAccess final {
  template <class R, class... A>
  [[nodiscard]] static const std::shared_ptr<ProgramState> &
  state(const Program<R(A...)> &program) noexcept {
    return program.state_;
  }
};

} // namespace rund::compute::detail
