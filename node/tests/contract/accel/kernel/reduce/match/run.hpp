#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "bindings.hpp"
#include <iostream>
#include <span>

namespace node_accel_contract::reduce {

template <typename T>
[[nodiscard]] bool MatchesReference(const rund::AccelDevice &pick,
                                    const rund::kernel::ComputeScalar scalar,
                                    const rund::kernel::ComputeDomain domain,
                                    const rund::kernel::ReduceOp op,
                                    const rund::kernel::ReduceElement element,
                                    const std::span<const T> input,
                                    const rund::kernel::u64 block_size = 4u) {
  if (!pick.check.ok) {
    return false;
  }
  const bool signed_domain = domain == rund::kernel::ComputeDomain::I32 ||
                             domain == rund::kernel::ComputeDomain::I64 ||
                             domain == rund::kernel::ComputeDomain::Fixed;
  const match::Reference<T> ref =
      match::BuildReference(op, element, input, signed_domain);
  match::Resources<T> resources = match::BuildResources(
      pick, scalar, domain, op, element, input, block_size);
  if (!ref.ok) {
    std::cerr << "reduce reference failed op=" << static_cast<unsigned>(op)
              << " block=" << block_size << '\n';
    return false;
  }
  if (!resources.kernel.check.ok) {
    std::cerr << "reduce compile failed op=" << static_cast<unsigned>(op)
              << " block=" << block_size
              << " reason=" << resources.kernel.check.reason << '\n';
    return false;
  }
  const auto bindings = match::Bindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = input.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    std::cerr << "reduce execute failed op=" << static_cast<unsigned>(op)
              << " block=" << block_size << " reason=" << evidence.reason
              << " submits=" << evidence.command_submit_count << '\n';
    return false;
  }

  T downloaded = 0u;
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.output, &downloaded, sizeof(downloaded));
  if (!download.ok || downloaded != ref.expected) {
    std::cerr << "reduce result mismatch op=" << static_cast<unsigned>(op)
              << " block=" << block_size << " reason=" << download.reason
              << " actual=" << downloaded << " expected=" << ref.expected
              << '\n';
    return false;
  }
  return true;
}

} // namespace node_accel_contract::reduce

#include "run/cases.hpp"
