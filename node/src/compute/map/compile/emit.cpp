#include "model.hpp"

#include "../../type.hpp"
#include "../name.hpp"

#include <utility>

namespace rund::compute::detail {
namespace {

template <compute_dsl::detail::ScalarMode Mode> struct DynamicBody final {
  static constexpr compute_dsl::detail::ScalarMode scalar_mode() noexcept {
    return Mode;
  }
  [[nodiscard]] const std::vector<compute_dsl::detail::BindingRuntime> &
  bindings() const noexcept {
    return values;
  }
  [[nodiscard]] kernel::u64 tile_count() const noexcept { return count; }
  [[nodiscard]] constexpr kernel::ComputeFixedFormat
  fixed_format() const noexcept {
    if constexpr (Mode == compute_dsl::detail::ScalarMode::FixedLane64 ||
                  Mode == compute_dsl::detail::ScalarMode::FixedLane32) {
      return format;
    }
    return {};
  }
  [[nodiscard]] bool ok() const noexcept { return valid; }
  [[nodiscard]] const char *reason() const noexcept { return why; }

  std::vector<compute_dsl::detail::BindingRuntime> values;
  kernel::u64 count = 0u;
  kernel::ComputeFixedFormat format{};
  bool valid = true;
  const char *why = "ok";
};

template <compute_dsl::detail::ScalarMode Mode>
[[nodiscard]] compute_dsl::ComputeOp
build(const std::size_t count, const std::span<const Type> outputs,
      const std::span<const Type> inputs,
      const std::span<const ExprRef> expressions,
      const InputFixedFormats &input_formats,
      const std::span<const MapRead> routes) {
  DynamicBody<Mode> body{};
  body.count = static_cast<kernel::u64>(count);
  if constexpr (Mode == compute_dsl::detail::ScalarMode::FixedLane32 ||
                Mode == compute_dsl::detail::ScalarMode::FixedLane64) {
    body.format =
        kernel_format(input_formats.get(0u, expressions.front().fixed_format));
  }
  body.values.reserve(inputs.size() + outputs.size());
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    body.values.push_back(compute_dsl::detail::BindingRuntime{
        .kind = compute_dsl::detail::BindingKind::Read,
        .numeric_mode = numeric_mode(inputs[index]),
        .name = map_input_name(inputs.size(), index),
        .element_bytes = static_cast<kernel::u32>(type_bytes(inputs[index]))});
  }
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    body.values.push_back(compute_dsl::detail::BindingRuntime{
        .kind = compute_dsl::detail::BindingKind::Write,
        .numeric_mode = numeric_mode(outputs[index]),
        .name = map_output_name(outputs.size(), index),
        .element_bytes = static_cast<kernel::u32>(type_bytes(outputs[index]))});
  }

  compute_dsl::detail::BuildContext context{body.bindings(), Mode,
                                            body.fixed_format()};
  const std::size_t source_count =
      routes.empty() ? inputs.size() : routes.size();
  std::vector<KernelExpr> reads;
  reads.reserve(source_count);
  for (std::size_t index = 0u; index < source_count; ++index) {
    const MapRead route = routes.empty() ? MapRead{} : routes[index];
    if constexpr (Mode == compute_dsl::detail::ScalarMode::FixedLane32 ||
                  Mode == compute_dsl::detail::ScalarMode::FixedLane64) {
      const kernel::ComputeFixedFormat format = kernel_format(
          input_formats.get(index, expressions.front().fixed_format));
      reads.push_back(
          route.indexed()
              ? compute_dsl::detail::DynamicReadAt(
                    context, static_cast<kernel::u32>(index), route.index,
                    route.count, format)
              : compute_dsl::detail::DynamicRead(
                    context, static_cast<kernel::u32>(index), format));
    } else {
      reads.push_back(route.indexed()
                          ? compute_dsl::detail::DynamicReadAt(
                                context, static_cast<kernel::u32>(index),
                                route.index, route.count)
                          : compute_dsl::detail::DynamicRead(
                                context, static_cast<kernel::u32>(index)));
    }
  }
  const KernelExpr logical_index = compute_dsl::detail::DynamicIndex(context);
  const KernelExpr anchor = reads.empty() ? logical_index : reads.front();
  std::vector<KernelExpr> replay_values;
  std::vector<std::uint8_t> replay_state;
  const ExprState *replay_owner = nullptr;
  for (std::size_t index = 0u; index < expressions.size(); ++index) {
    const ExprState *const owner = expressions[index].state.get();
    if (owner != replay_owner) {
      replay_values.clear();
      replay_state.clear();
      replay_owner = owner;
    }
    const auto value = replay(expressions[index], reads, anchor, logical_index,
                              replay_values, replay_state);
    if (!value) {
      context.reject("compute_expression_unsupported");
      break;
    }
    const ExprOp root =
        expressions[index].state->nodes[expressions[index].node - 1u].operation;
    const kernel::u32 binding = static_cast<kernel::u32>(inputs.size() + index);
    if (root == ExprOp::CheckedOrdinal) {
      compute_dsl::detail::DynamicCheckedOrdinalWrite(context, binding, *value);
    } else if (root == ExprOp::BoundaryMask) {
      compute_dsl::detail::DynamicBoundaryMaskWrite(
          context, binding, *value,
          kernel_format(expressions[index].fixed_format));
    } else {
      compute_dsl::detail::DynamicWrite(context, binding, *value);
    }
  }
  kernel::ComputeIR ir = compute_dsl::detail::BuildIr("", body, context);
  kernel::ComputeMap map = compute_dsl::detail::BuildMap(ir, body);
  if (!ir.ok) {
    map.op_hash_hi = 0u;
    map.op_hash_lo = 0u;
  }
  return compute_dsl::ComputeOp{std::move(ir), map, std::move(body.values),
                                static_cast<kernel::u64>(count)};
}

} // namespace

compute_dsl::ComputeOp
build_dynamic_op(const std::size_t count, const std::span<const Type> outputs,
                 const std::span<const Type> inputs,
                 const std::span<const ExprRef> expressions,
                 const InputFixedFormats &input_formats,
                 const std::span<const MapRead> reads) {
  const Type mode = inputs.empty() ? outputs.front() : inputs.front();
  switch (mode) {
  case Type::I32:
    return build<compute_dsl::detail::ScalarMode::I32>(
        count, outputs, inputs, expressions, input_formats, reads);
  case Type::U32:
    return build<compute_dsl::detail::ScalarMode::U32>(
        count, outputs, inputs, expressions, input_formats, reads);
  case Type::I64:
    return build<compute_dsl::detail::ScalarMode::I64>(
        count, outputs, inputs, expressions, input_formats, reads);
  case Type::U64:
    return build<compute_dsl::detail::ScalarMode::U64>(
        count, outputs, inputs, expressions, input_formats, reads);
  case Type::FixedLane32:
    return build<compute_dsl::detail::ScalarMode::FixedLane32>(
        count, outputs, inputs, expressions, input_formats, reads);
  case Type::FixedLane64:
    return build<compute_dsl::detail::ScalarMode::FixedLane64>(
        count, outputs, inputs, expressions, input_formats, reads);
  }
  return {};
}

} // namespace rund::compute::detail
