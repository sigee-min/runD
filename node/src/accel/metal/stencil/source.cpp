#include "local.hpp"
#include "source/build.hpp"
#include "../../kernel/backend/source_recipe.hpp"

#include <string>

namespace rund::node::accel::detail {

std::string MetalStencilSource(const rund::kernel::StencilOp op) {
  const auto emit = [op](auto &sink) noexcept(noexcept(
      EmitMetalStencilSource(sink, op))) {
    return EmitMetalStencilSource(sink, op);
  };
  return backend_source_recipe::materialize(emit);
}

bool MetalStencilSourceUpperBytes(const rund::kernel::StencilOp op,
                                  std::uint64_t &upper) noexcept {
  const auto emit = [op](backend_source_recipe::CountSink &sink) noexcept {
    return EmitMetalStencilSource(sink, op);
  };
  return backend_source_recipe::bytes(emit, upper);
}

} // namespace rund::node::accel::detail
