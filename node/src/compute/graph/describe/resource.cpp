#include "model.hpp"

#include "../../fixed/format.hpp"
#include "../../size.hpp"
#include "../../type.hpp"

#include <optional>

namespace rund::compute::detail::graph_detail::describe_detail {
namespace {

[[nodiscard]] std::optional<graph::Value>
public_type(const Type type) noexcept {
  switch (type) {
  case Type::I32:
    return graph::Value::I32;
  case Type::U32:
    return graph::Value::U32;
  case Type::I64:
    return graph::Value::I64;
  case Type::U64:
    return graph::Value::U64;
  case Type::FixedLane32:
  case Type::FixedLane64:
    return graph::Value::Fixed;
  }
  return std::nullopt;
}

[[nodiscard]] bool valid(const graph::Info &info,
                         const std::uint32_t id) noexcept {
  return id != 0u && id <= info.resources.size() &&
         info.resources[id - 1u].id == id;
}

void mark(graph::Info &info, const std::span<const std::uint32_t> ids,
          const graph::Visibility visibility) noexcept {
  for (const std::uint32_t id : ids) {
    if (valid(info, id)) {
      info.resources[id - 1u].visibility = visibility;
    }
  }
}

} // namespace

Status build_resources(const GraphState &state, graph::Info &info) {
  info.resources.reserve(state.values.size());
  for (std::size_t index = 0u; index < state.values.size(); ++index) {
    const GraphValue &value = state.values[index];
    const std::size_t width = type_bytes(value.type);
    std::size_t bytes = 0u;
    if (width == 0u || !size::multiply(value.count, width, bytes)) {
      return Status::fail(Reason::GraphCapacity);
    }
    const auto type = public_type(value.type);
    if (!type) {
      return Status::fail(Reason::TypeUnsupported);
    }
    const std::uint32_t id = static_cast<std::uint32_t>(index + 1u);
    info.resources.push_back(graph::Resource{
        .id = id,
        .type = *type,
        .integer_bits = value.fixed_format.integer_bits,
        .fraction_bits = value.fixed_format.fraction_bits,
        .rounding = value.fixed_format.rounding,
        .overflow = value.fixed_format.overflow,
        .approximation = value.fixed_format.approximation,
        .visibility = graph::Visibility::Internal,
        .elements = value.count,
        .element_bytes = width,
        .bytes = bytes,
        .active = value.active,
        .parent = value.parent,
        .alias_group = id,
        .alias_offset_bytes = 0u,
    });
  }

  info.inputs = state.inputs;
  info.outputs = state.outputs;
  // Output is marked first so a value authored as both input and output keeps
  // the historical Input precedence without a per-value membership search.
  mark(info, info.outputs, graph::Visibility::Output);
  mark(info, info.inputs, graph::Visibility::Input);
  return Status::success();
}

Status
validate_bindings(const graph::Info &info,
                  const std::span<const std::uint32_t> identity_outputs) {
  if (info.resources.empty() || info.inputs.empty() || info.outputs.empty() ||
      identity_outputs.empty()) {
    return Status::fail(Reason::GraphIncomplete);
  }
  for (const std::uint32_t id : info.inputs) {
    if (!valid(info, id)) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
  }
  for (const std::uint32_t id : info.outputs) {
    if (!valid(info, id)) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
  }
  for (const std::uint32_t id : identity_outputs) {
    if (!valid(info, id)) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_detail::describe_detail
