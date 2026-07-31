#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "bindings.hpp"

#include <iostream>

namespace node_accel_contract::stencil {
namespace match_detail {

[[nodiscard]] inline const char *
BackendLastError(const rund::AccelDevice &pick) {
  if (pick.backend.last_error == nullptr) {
    return "backend_last_error_absent";
  }
  const char *const reason = pick.backend.last_error(pick.backend.context);
  return reason == nullptr ? "backend_last_error_null" : reason;
}

[[nodiscard]] inline bool Fail(const char *const stage,
                               const char *const reason) {
  std::cerr << "stencil match stage failed: " << stage
            << " reason: " << (reason == nullptr ? "null" : reason) << '\n';
  return false;
}

} // namespace match_detail

template <typename T, std::size_t Count>
[[nodiscard]] bool MatchesReference(const rund::AccelDevice &pick,
                                    const rund::kernel::ComputeScalar scalar,
                                    const rund::kernel::ComputeDomain domain,
                                    const rund::kernel::StencilOp op,
                                    const rund::kernel::StencilElement element,
                                    const rund::kernel::u64 radius,
                                    const std::array<T, Count> &input) {
  namespace fix = node_accel_contract::primitive;
  namespace run = node_accel_contract::stencil::match;
  if (!pick.check.ok) {
    return match_detail::Fail("pick", pick.check.reason);
  }
  const bool signed_domain = domain == rund::kernel::ComputeDomain::I32 ||
                             domain == rund::kernel::ComputeDomain::I64 ||
                             domain == rund::kernel::ComputeDomain::Fixed;
  const run::Reference<T, Count> ref =
      run::BuildReference(op, element, radius, input, signed_domain);
  if (!ref.ok) {
    return match_detail::Fail("reference", "reference_failed");
  }
  run::Resources<T, Count> resources =
      run::BuildResources(pick, scalar, domain, op, element, radius, input);
  if (!resources.kernel.check.ok) {
    return match_detail::Fail("compile", resources.kernel.check.reason);
  }
  const auto bindings = run::Bindings(resources);
  const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
      resources.context, resources.kernel,
      rund::AccelRun{.bindings = bindings.data(),
                     .binding_count = bindings.size(),
                     .tile_count = input.size(),
                     .fresh_evidence = true,
});
  if (!evidence.ok || evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    return match_detail::Fail("run", match_detail::BackendLastError(pick));
  }
  std::array<T, Count> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.output, downloaded.data(),
      downloaded.size() * sizeof(T));
  if (!download.ok) {
    return match_detail::Fail("download", download.reason);
  }
  if (fix::HashValues(downloaded.data(), downloaded.size()) !=
      fix::HashValues(ref.expected.data(), ref.expected.size())) {
    return match_detail::Fail("compare", "output_mismatch");
  }
  return true;
}

} // namespace node_accel_contract::stencil
