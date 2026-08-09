#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>
#include <kernel/program/compute/partition/reference.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

#include <array>
#include <cstdio>

namespace node_accel_contract::partition {
namespace {

template <std::size_t Count>
[[nodiscard]] bool Match(const rund::AccelDevice &pick,
                         const std::array<rund::kernel::u32, Count> &flags,
                         const std::array<rund::kernel::u32, Count> &values,
                         const rund::kernel::u64 physical_dispatches) {
  namespace p = node_accel_contract::primitive;
  if (!pick.check.ok) {
    return false;
  }

  std::array<rund::kernel::u32, Count> expected{};
  rund::kernel::u64 false_count = 0u;
  rund::kernel::u64 true_count = 0u;
  const rund::kernel::PartitionResult reference =
      rund::kernel::ReferenceStablePartitionU32(flags.data(), values.data(),
                                                flags.size(), expected.data(),
                                                &false_count, &true_count);
  if (!reference.ok || false_count + true_count != Count) {
    return false;
  }

  Fixture fixture = Make(pick, Count);
  Bind(fixture);
  if (!fixture.context.check.ok || !fixture.flags.check.ok ||
      !fixture.values.check.ok || !fixture.output.check.ok ||
      !fixture.plan.ok ||
      !rund::node::accel::UploadAccelBuffer(
           fixture.context, fixture.flags, flags.data(),
           flags.size() * sizeof(rund::kernel::u32))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           fixture.context, fixture.values, values.data(),
           values.size() * sizeof(rund::kernel::u32))
           .ok) {
    return false;
  }

  const rund::AccelKernel kernel =
      rund::node::accel::CompileAccelKernel(fixture.context, fixture.graph);
  if (!kernel.check.ok) {
    return false;
  }

  const std::array<rund::AccelRunBinding, 3u> bindings{
      rund::AccelRunBinding{
          .buffer = &fixture.flags,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &fixture.values,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &fixture.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
      fixture.context, kernel,
      rund::AccelRun{
          .bindings = bindings.data(),
          .binding_count = bindings.size(),
          .tile_count = fixture.desc.element_count,
          .fresh_evidence = true,
      });
  if (!evidence.outcome.ok ||
      evidence.run.transfer.host_to_device_bytes != 0u ||
      evidence.run.transfer.device_to_host_bytes != 0u ||
      evidence.run.work.dispatch_count != physical_dispatches ||
      evidence.run.work.original_dispatch_count != fixture.plan.pass_count ||
      evidence.run.work.final_dispatch_count != physical_dispatches) {
    std::fprintf(
        stderr,
        "partition run count=%zu ok=%d reason=%s dispatch=%llu expected=%llu "
        "original=%llu final=%llu plan=%llu upload=%llu download=%llu\n",
        Count, evidence.outcome.ok, evidence.outcome.reason,
        static_cast<unsigned long long>(evidence.run.work.dispatch_count),
        static_cast<unsigned long long>(physical_dispatches),
        static_cast<unsigned long long>(
            evidence.run.work.original_dispatch_count),
        static_cast<unsigned long long>(evidence.run.work.final_dispatch_count),
        static_cast<unsigned long long>(fixture.plan.pass_count),
        static_cast<unsigned long long>(
            evidence.run.transfer.host_to_device_bytes),
        static_cast<unsigned long long>(
            evidence.run.transfer.device_to_host_bytes));
    return false;
  }

  std::array<rund::kernel::u32, Count> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      fixture.context, fixture.output, downloaded.data(),
      downloaded.size() * sizeof(rund::kernel::u32));
  const auto actual_hash = p::HashValues(downloaded.data(), downloaded.size());
  const auto expected_hash = p::HashValues(expected.data(), expected.size());
  if (!download.ok || actual_hash != expected_hash) {
    std::fprintf(stderr,
                 "partition output count=%zu download=%d reason=%s "
                 "actual=%llu expected=%llu\n",
                 Count, download.ok, download.reason,
                 static_cast<unsigned long long>(actual_hash),
                 static_cast<unsigned long long>(expected_hash));
    return false;
  }
  return true;
}

} // namespace

bool MatchesReference(const rund::AccelDevice &pick) {
  constexpr std::array<rund::kernel::u32, 8u> flags{1u, 0u, 2u, 0u,
                                                    0u, 7u, 0u, 3u};
  constexpr std::array<rund::kernel::u32, 8u> values{11u, 12u, 13u, 14u,
                                                     15u, 16u, 17u, 18u};
  if (!Match(pick, flags, values, 3u)) {
    return false;
  }
  if (pick.api != rund::AccelApi::Metal) {
    return true;
  }

  std::array<rund::kernel::u32, 1025u> large_flags{};
  std::array<rund::kernel::u32, 1025u> large_values{};
  for (std::size_t index = 0u; index < large_flags.size(); ++index) {
    large_flags[index] = index % 3u == 0u ? 0u : 1u;
    large_values[index] = static_cast<rund::kernel::u32>(index + 11u);
  }
  return Match(pick, large_flags, large_values, 5u);
}

} // namespace node_accel_contract::partition
