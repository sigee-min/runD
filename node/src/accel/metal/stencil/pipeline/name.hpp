#pragma once
#include "../local.hpp"
#include "../../../domain.hpp"
namespace rund::node::accel::detail {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline const char *StencilOpName(const rund::kernel::StencilOp op) noexcept {
  if (op == rund::kernel::StencilOp::Min) {
    return "min";
  }
  if (op == rund::kernel::StencilOp::Max) {
    return "max";
  }
  return "sum";
}
[[nodiscard]] inline bool StencilUsesSignedArithmetic(
    const rund::kernel::StencilOp op,
    const rund::kernel::ComputeDomain domain) noexcept {
  // Sum is intentionally modulo-width for every domain and both emitted
  // signed-name variants use uint/ulong. Min/Max are the only operations whose
  // executable semantics depend on signedness.
  return op != rund::kernel::StencilOp::Sum && IsSignedDomain(domain);
}
#include "name/function.hpp"
[[nodiscard]] inline std::string
StencilPipelineKey(const rund::kernel::StencilOp op,
                   const rund::kernel::StencilElement element,
                   const rund::kernel::ComputeDomain domain) {
  std::string key = "stencil.";
  key += StencilOpName(op);
  key += StencilUsesSignedArithmetic(op, domain) ? ".i" : ".u";
  key += element == rund::kernel::StencilElement::U64 ? "64" : "32";
  return key;
}
#endif
} // namespace rund::node::accel::detail
