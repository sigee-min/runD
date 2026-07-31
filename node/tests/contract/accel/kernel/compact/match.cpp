#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <kernel/program/compute/compact/reference.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

namespace node_accel_contract {
namespace {

struct RunHash {
  std::uint64_t output_hash = 0u;
  std::uint64_t output_count = 0u;
  bool valid = false;
};

[[nodiscard]] RunHash
CompactHashMatchesCpuReference(const rund::AccelDevice &pick,
                               const rund::kernel::ComputeScalar scalar,
                               const std::array<rund::kernel::u32, 8u> &flags,
                               const std::uint64_t capacity) {
  namespace p = node_accel_contract::primitive;
  if (!pick.check.ok || capacity == 0u || capacity > flags.size()) {
    return {};
  }
  std::array<rund::kernel::u32, 8u> expected{};
  std::uint64_t expected_count = 0u;
  if (!rund::kernel::ReferenceCompactIdsU32(flags.data(), flags.size(),
                                            capacity, expected.data(),
                                            &expected_count)
           .ok) {
    return {};
  }
  const compact::RunContext ctx =
      compact::Compile(pick, scalar, flags, capacity);
  if (!ctx.valid) {
    return {};
  }
  const rund::AccelEvidence evidence = compact::Run(ctx, flags.size());
  if (!evidence.ok || evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    return {};
  }
  std::array<rund::kernel::u32, 8u> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      ctx.context, ctx.output, downloaded.data(),
      capacity * sizeof(rund::kernel::u32));
  for (std::uint64_t index = 1u; index < expected_count; ++index) {
    if (downloaded[static_cast<std::size_t>(index - 1u)] >=
        downloaded[static_cast<std::size_t>(index)]) {
      return {};
    }
  }
  const std::uint64_t hash = p::HashValues(
      downloaded.data(), static_cast<std::size_t>(expected_count));
  if (!download.ok ||
      hash != p::HashValues(expected.data(),
                            static_cast<std::size_t>(expected_count))) {
    return {};
  }
  return RunHash{
      .output_hash = hash,
      .output_count = expected_count,
      .valid = true,
  };
}

} // namespace

bool CompactMatchesCpuReference(const rund::AccelDevice &pick,
                                const rund::kernel::ComputeScalar scalar,
                                const std::array<rund::kernel::u32, 8u> &flags,
                                const std::uint64_t capacity) {
  return CompactHashMatchesCpuReference(pick, scalar, flags, capacity).valid;
}

bool CompactBackendsAgree(const rund::AccelDevice &metal,
                          const rund::AccelDevice &vulkan,
                          const rund::kernel::ComputeScalar scalar,
                          const std::array<rund::kernel::u32, 8u> &flags,
                          const std::uint64_t capacity) {
  const RunHash metal_hash =
      CompactHashMatchesCpuReference(metal, scalar, flags, capacity);
  const RunHash vulkan_hash =
      CompactHashMatchesCpuReference(vulkan, scalar, flags, capacity);
  return metal_hash.valid && vulkan_hash.valid &&
         metal_hash.output_hash == vulkan_hash.output_hash &&
         metal_hash.output_count == vulkan_hash.output_count;
}

} // namespace node_accel_contract
