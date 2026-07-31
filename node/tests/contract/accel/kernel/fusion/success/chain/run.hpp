#pragma once

#include <accel/context/value.hpp>

#include <node/accel/context.hpp>

#include "reference.hpp"

#include <cstdio>

namespace node_accel_contract::fusion {

[[nodiscard]] bool RunFusedChainCase(const rund::AccelContext &context,
                                     const rund::compute_dsl::ComputeOp &op,
                                     const Inputs &inputs) {
  const chain::Resources resources = chain::MakeResources(context, inputs);
  if (!resources.ok) {
    std::fprintf(stderr, "fusion resources failed\n");
    return false;
  }
  if (!chain::RunFused(context, op, inputs, resources)) {
    std::fprintf(stderr, "fusion fused execution failed\n");
    return false;
  }
  if (!chain::RunReference(context, op, inputs, resources)) {
    std::fprintf(stderr, "fusion reference execution failed\n");
    return false;
  }

  std::array<rund::kernel::i32, 8u> fused_download{};
  std::array<rund::kernel::i32, 8u> ref_download{};
  if (!rund::node::accel::DownloadAccelBuffer(context, resources.fused_output,
                                              fused_download.data(),
                                              sizeof(fused_download))
           .ok ||
      !rund::node::accel::DownloadAccelBuffer(context, resources.ref_output,
                                              ref_download.data(),
                                              sizeof(ref_download))
           .ok) {
    return false;
  }
  if (fused_download != inputs.add14 || ref_download != inputs.add14 ||
      fused_download != ref_download) {
    std::fprintf(stderr,
                 "fusion output mismatch fused0=%d ref0=%d expected0=%d\n",
                 fused_download[0], ref_download[0], inputs.add14[0]);
    return false;
  }
  return true;
}

} // namespace node_accel_contract::fusion
