#pragma once
#include "../../../domain.hpp"
#include "../local.hpp"
#include "../source/op.hpp"
#include "function.hpp"
namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline std::string
ReducePipelineKey(const rund::kernel::ReducePlan &plan,
                  const rund::kernel::ComputeDomain domain) {
  std::string key = "reduce.";
  key += MetalReduceOpName(plan.op);
  key += ".";
  const bool signed_domain = IsSignedDomain(domain);
  key += signed_domain ? "i" : "u";
  key += plan.element == rund::kernel::ReduceElement::U64 ? "64." : "32.";
  key += std::to_string(plan.block_size);
  return key;
}
#endif
} // namespace rund::node::accel::detail
