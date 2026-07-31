#pragma once

#include <accel/check.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <node/accel/context.hpp>

#include "resources.hpp"

namespace node_accel_contract::cpu_context {

[[nodiscard]] inline MapRun RunMapHash(const MapHashWork &work,
                                       const MapHashResources &resources) {
  const std::array<rund::AccelRunBinding, 2u> bindings{
      rund::AccelRunBinding{.buffer = &resources.read,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &resources.write,
                            .role = rund::kernel::BufferRole::Write},
  };
  const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
      resources.context, resources.kernel,
      rund::AccelRun{.bindings = bindings.data(),
                     .binding_count = bindings.size(),
                     .tile_count = kMapHashCount,
                     .fresh_evidence = true,
});
  if (!evidence.ok) {
    return {};
  }

  std::array<rund::kernel::i32, kMapHashCount> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.write, downloaded.data(),
      downloaded.size() * sizeof(downloaded[0]));
  const std::uint64_t hash = HashValues(downloaded.data(), downloaded.size());
  if (!download.ok ||
      hash != HashValues(work.expected.data(), work.expected.size())) {
    return {};
  }
  return MapRun{.hash = hash, .evidence = evidence, .ok = true};
}

[[nodiscard]] inline MapRun
ContextMapHash(const rund::AccelDevice &pick) {
  MapHashWork work = MakeMapHashWork();
  const rund::compute_dsl::ComputeOp op = MakeMapHashOp(work);
  if (!op.ok()) {
    return {};
  }
  const MapHashResources resources = MakeMapHashResources(pick, work, op);
  if (!resources.ok) {
    return {};
  }
  return RunMapHash(work, resources);
}

} // namespace node_accel_contract::cpu_context
