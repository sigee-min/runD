#include "../../kernel/backend/source_recipe.hpp"
#include "local.hpp"
#include "source/build.hpp"

#include <string>

namespace rund::node::accel::detail {

std::string MetalStencilSource(const rund::kernel::StencilOp op,
                               const StencilGpuShape shape,
                               const RangeAggregatePlan &range) {
  const auto emit = [op, shape, &range](auto &sink) noexcept(noexcept(
                        EmitMetalStencilSource(sink, op, shape, range))) {
    return EmitMetalStencilSource(sink, op, shape, range);
  };
  return backend_source_recipe::materialize(emit);
}

bool MetalStencilSourceUpperBytes(const rund::kernel::StencilOp op,
                                  const RangeAggregatePlan &range,
                                  std::uint64_t &upper) noexcept {
  const StencilGpuShape shape = StencilGpuShapeFromRangeAggregatePlan(range);
  const auto emit = [op, shape,
                     &range](backend_source_recipe::CountSink &sink) noexcept {
    return EmitMetalStencilSource(sink, op, shape, range);
  };
  return backend_source_recipe::bytes(emit, upper);
}

} // namespace rund::node::accel::detail
