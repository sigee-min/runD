#include "fill.hpp"

#include "../../../type.hpp"
#include "../../state.hpp"
#include "../projection.hpp"

#include <kernel/program/compute/window/model.hpp>

#include <cstdint>
#include <limits>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool element_bytes(const Type type,
                                 std::uint64_t &bytes) noexcept {
  switch (type) {
  case Type::I32:
  case Type::U32:
  case Type::FixedLane32:
    bytes = sizeof(std::uint32_t);
    return true;
  case Type::I64:
  case Type::U64:
  case Type::FixedLane64:
    bytes = sizeof(std::uint64_t);
    return true;
  }
  return false;
}

[[nodiscard]] bool clip_identity(const Type type,
                                 const kernel::WindowOp operation,
                                 std::uint64_t &value) noexcept {
  if (operation == kernel::WindowOp::Sum) {
    value = 0u;
    return true;
  }
  const bool minimum = operation == kernel::WindowOp::Min;
  if (!minimum && operation != kernel::WindowOp::Max) {
    return false;
  }
  switch (type) {
  case Type::I32:
  case Type::FixedLane32:
    value = minimum ? static_cast<std::uint32_t>(
                          std::numeric_limits<std::int32_t>::max())
                    : static_cast<std::uint32_t>(
                          std::numeric_limits<std::int32_t>::lowest());
    return true;
  case Type::U32:
    value = minimum ? std::numeric_limits<std::uint32_t>::max() : 0u;
    return true;
  case Type::I64:
  case Type::FixedLane64:
    value = minimum ? static_cast<std::uint64_t>(
                          std::numeric_limits<std::int64_t>::max())
                    : static_cast<std::uint64_t>(
                          std::numeric_limits<std::int64_t>::lowest());
    return true;
  case Type::U64:
    value = minimum ? std::numeric_limits<std::uint64_t>::max() : 0u;
    return true;
  }
  return false;
}

} // namespace

bool project_virtual_fetch_fill(const VirtualPipelineState &state,
                                const VirtualRunProjection &run,
                                VirtualFetchFill &fill) noexcept {
  using residency::execution::FetchFill;
  fill = {};
  if (state.geometry.route == VirtualRoute::Pointwise) {
    fill.policy = FetchFill::ZeroInactiveTail;
    return true;
  }
  if (state.geometry.route != VirtualRoute::Window ||
      (!run.clamp_window && !run.clip_window) ||
      (run.clamp_window && run.clip_window) ||
      !element_bytes(run.input_type, fill.element_bytes)) {
    return false;
  }
  const auto operation = static_cast<kernel::WindowOp>(run.operation);
  if (operation != kernel::WindowOp::Sum &&
      operation != kernel::WindowOp::Min &&
      operation != kernel::WindowOp::Max) {
    return false;
  }
  if (run.clamp_window) {
    fill.policy = FetchFill::RepeatBoundary;
    return true;
  }
  fill.policy = FetchFill::ConstantBoundary;
  return clip_identity(run.input_type, operation, fill.value);
}

} // namespace rund::compute::detail
