#pragma once

#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "bindings.hpp"

#include <cstdio>

namespace node_accel_contract::collective {

template <typename Key>
[[nodiscard]] bool
SortRejectsRunBufferShapeMismatch(const rund::AccelDevice &pick,
                                  const rund::kernel::ComputeScalar scalar) {
  if (!pick.check.ok) {
    return false;
  }
  sort_reject::Resources<Key> resources =
      sort_reject::BuildResources<Key>(pick, scalar);
  if (!resources.kernel.check.ok) {
    return false;
  }
  const auto bindings = sort_reject::Bindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = 8u,
                                            .fresh_evidence = true,
                                        });
  if (!EvidenceReason(evidence, "accel_kernel_buffer_shape_mismatch")) {
    std::fprintf(stderr,
                 "sort shape rejection mismatch: api=%u ok=%u reason=%s\n",
                 static_cast<unsigned>(pick.api),
                 static_cast<unsigned>(evidence.ok), evidence.reason);
    return false;
  }
  return true;
}

} // namespace node_accel_contract::collective
