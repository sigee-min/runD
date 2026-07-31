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
#include "name/function.hpp"
[[nodiscard]] inline std::string
StencilPipelineKey(const rund::kernel::StencilOp op,
                   const rund::kernel::StencilElement element,
                   const rund::kernel::ComputeDomain domain) {
  std::string key = "stencil.";
  key += StencilOpName(op);
  key += IsSignedDomain(domain) ? ".i" : ".u";
  key += element == rund::kernel::StencilElement::U64 ? "64" : "32";
  return key;
}
#endif
} // namespace rund::node::accel::detail
