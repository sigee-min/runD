#include "local.hpp"
#include "source/build.hpp"

#include <string>

namespace rund::node::accel::detail {

std::string MetalStencilSource(const rund::kernel::StencilOp op) {
  std::string source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct StencilParams {
  ulong element_count;
  ulong radius;
};

)MSL";
  AppendMetalStencilKernel(source, op, "uint", "u32");
  AppendMetalStencilKernel(source, op, "ulong", "u64");
  AppendMetalStencilKernel(
      source, op, op == rund::kernel::StencilOp::Sum ? "uint" : "int", "i32");
  AppendMetalStencilKernel(
      source, op, op == rund::kernel::StencilOp::Sum ? "ulong" : "long", "i64");
  return source;
}

} // namespace rund::node::accel::detail
