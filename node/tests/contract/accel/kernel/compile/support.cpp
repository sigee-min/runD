#include <accel/api.hpp>
#include <accel/context/value.hpp>
#include <accel/kernel/check.hpp>
#include <accel/kernel/value.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

#include <type_traits>

namespace node_accel_contract::kernel_case::compile {

bool SignatureAndSupportRejects() {
  static_assert(std::is_same_v<decltype(&rund::node::accel::CompileAccelKernel),
                               CompileAccelKernelFn>);
  static_assert(std::is_same_v<
                decltype(&rund::node::accel::detail::AdmitKernelForSupport),
                AdmitKernelForSupportFn>);
  static_assert(std::is_same_v<decltype(&rund::node::accel::RunAccelKernel),
                               RunAccelKernelFn>);
  if (!CheckReason(rund::node::accel::detail::AdmitKernelForSupport(
                       rund::AccelContext{}, rund::AccelKernel{})
                       .check,
                   "accel_kernel_graph_invalid")) {
    return false;
  }

  rund::AccelKernel forged{};
  forged.check = rund::AccelKernelCheck{true, "ok"};
  forged.reason = "ok";
  forged.graph_id_hi = 1u;
  forged.graph_id_lo = 2u;
  forged.node_count = 1u;
  forged.api = rund::AccelApi::Metal;
  forged.scalar = rund::kernel::ComputeScalar::Lane32;
  forged.frozen_caps = rund::kernel::ComputeCaps{
      .api = rund::kernel::ComputeApi::Metal,
      .device_bytes = 4096u,
      .staging_bytes = 1024u,
      .max_window_tiles = 8u,
      .subgroup_width = 1u,
      .ok = true,
      .reason = "ok",
  };
  return CheckReason(rund::node::accel::detail::AdmitKernelForSupport(
                         rund::AccelContext{}, forged)
                         .check,
                     "accel_kernel_graph_invalid");
}

} // namespace node_accel_contract::kernel_case::compile
