#include "../build.hpp"
#include "model.hpp"

#include "../../status.hpp"
#include "../name.hpp"

#include <memory>
#include <utility>

namespace rund::compute::detail {

Result<compute_dsl::ComputeOp>
build_map_operation_multi(const std::size_t count,
                          const std::span<const Type> outputs,
                          const std::span<const Type> inputs,
                          const std::span<const ExprRef> expressions,
                          const std::span<const MapRead> reads) {
  const std::size_t source_count = reads.empty() ? inputs.size() : reads.size();
  if (source_count > inputs.size() || inputs.size() > MapInputCapacity) {
    return Result<compute_dsl::ComputeOp>::fail(Reason::IrBindingInvalid);
  }
  for (const MapRead read : reads) {
    if ((!read.indexed() && read.count != 0u) ||
        (read.indexed() &&
         (read.index < source_count || read.index >= inputs.size() ||
          inputs[read.index] != Type::U32 || read.count == 0u))) {
      return Result<compute_dsl::ComputeOp>::fail(Reason::IrBindingInvalid);
    }
  }
  if (fixed_output_missing_quantize(outputs, expressions)) {
    return Result<compute_dsl::ComputeOp>::fail(Reason::FixedQuantizeRequired);
  }
  InputFixedFormats input_formats{};
  if (!expressions_ok(outputs, inputs.first(source_count), expressions,
                      input_formats)) {
    return Result<compute_dsl::ComputeOp>::fail(Reason::DomainTypeMismatch);
  }
  try {
    compute_dsl::ComputeOp operation = build_dynamic_op(
        count, outputs, inputs, expressions, input_formats, reads);
    if (!operation.ok()) {
      return Result<compute_dsl::ComputeOp>::fail(
          project_reason(operation.reason(), Reason::IrInvalid));
    }
    return Result<compute_dsl::ComputeOp>::success(std::move(operation));
  } catch (const std::bad_alloc &) {
    return Result<compute_dsl::ComputeOp>::fail(Reason::ProgramCapacity);
  }
}

} // namespace rund::compute::detail
