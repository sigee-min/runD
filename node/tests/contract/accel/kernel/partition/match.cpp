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

namespace node_accel_contract::partition {

bool MatchesReference(const rund::AccelDevice &pick) {
  namespace p = node_accel_contract::primitive;
  if (!pick.check.ok) {
    return false;
  }

  constexpr std::array<rund::kernel::u32, 8u> flags{1u, 0u, 2u, 0u,
                                                    0u, 7u, 0u, 3u};
  constexpr std::array<rund::kernel::u32, 8u> values{11u, 12u, 13u, 14u,
                                                     15u, 16u, 17u, 18u};
  std::array<rund::kernel::u32, 8u> expected{};
  rund::kernel::u64 false_count = 0u;
  rund::kernel::u64 true_count = 0u;
  const rund::kernel::PartitionResult reference =
      rund::kernel::ReferenceStablePartitionU32(flags.data(), values.data(),
                                                flags.size(), expected.data(),
                                                &false_count, &true_count);
  if (!reference.ok || false_count != 4u || true_count != 4u) {
    return false;
  }

  Fixture fixture = Make(pick);
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
  if (!evidence.ok || evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u ||
      evidence.dispatch_count != fixture.plan.pass_count ||
      evidence.original_dispatch_count != fixture.plan.pass_count ||
      evidence.final_dispatch_count != fixture.plan.pass_count) {
    return false;
  }

  std::array<rund::kernel::u32, 8u> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      fixture.context, fixture.output, downloaded.data(),
      downloaded.size() * sizeof(rund::kernel::u32));
  return download.ok && p::HashValues(downloaded.data(), downloaded.size()) ==
                            p::HashValues(expected.data(), expected.size());
}

} // namespace node_accel_contract::partition
