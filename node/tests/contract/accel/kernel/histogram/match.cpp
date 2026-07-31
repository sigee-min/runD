#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>

#include <accel/graph/factory/primitive/histogram/descriptor.hpp>
#include <kernel/program/compute/histogram/plan.hpp>

#include "local.hpp"
#include "test/compute/fixed.hpp"
#include <node/accel/context.hpp>

namespace node_accel_contract::histogram {
namespace {

struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer bins{};
  rund::AccelBuffer counts{};
  rund::AccelKernel kernel{};
};

template <std::size_t N>
[[nodiscard]] Resources
BuildResources(const rund::AccelDevice &pick,
               const std::array<rund::kernel::u32, N> &bins,
               const rund::kernel::u64 bin_count) {
  namespace fix = node_accel_contract::primitive;
  Resources out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.bins = rund::node::accel::CreateAccelBuffer(
      out.context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                                   sizeof(rund::kernel::u32), bins.size()));
  out.counts = rund::node::accel::CreateAccelBuffer(
      out.context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                                   sizeof(rund::kernel::u32), bin_count));
  if (!out.bins.check.ok || !out.counts.check.ok ||
      !rund::node::accel::UploadAccelBuffer(out.context, out.bins, bins.data(),
                                            bins.size() *
                                                sizeof(rund::kernel::u32))
           .ok) {
    return out;
  }

  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &out.bins,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.counts,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::kernel::HistogramDesc desc{
      .index = rund::kernel::HistogramIndex::U32,
      .count = rund::kernel::HistogramCount::U32,
      .element_count = bins.size(),
      .bin_count = bin_count,
  };
  const rund::kernel::HistogramPlan plan = rund::kernel::PlanHistogram(desc);
  if (!plan.ok) {
    return out;
  }
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelHistogram(refs.data(), refs.size(), desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = rund::kernel::ComputeScalar::Lane32,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(
                           rund::kernel::ComputeScalar::Lane32),
                   });
  return out;
}

[[nodiscard]] std::array<rund::AccelRunBinding, 2u>
Bindings(Resources &resources) {
  return {rund::AccelRunBinding{
              .buffer = &resources.bins,
              .role = rund::kernel::BufferRole::Read,
          },
          rund::AccelRunBinding{
              .buffer = &resources.counts,
              .role = rund::kernel::BufferRole::Write,
          }};
}

template <std::size_t N, std::size_t B>
[[nodiscard]] bool Matches(const rund::AccelDevice &pick,
                           const std::array<rund::kernel::u32, N> &bins,
                           const std::array<rund::kernel::u32, B> &expected) {
  Resources resources = BuildResources(pick, bins, expected.size());
  if (!resources.kernel.check.ok) {
    return false;
  }
  const auto bindings = Bindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = bins.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u) {
    return false;
  }
  std::array<rund::kernel::u32, B> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      resources.context, resources.counts, downloaded.data(),
      downloaded.size() * sizeof(downloaded[0]));
  return download.ok && downloaded == expected;
}

} // namespace

bool MatchesU32(const rund::AccelDevice &pick) {
  const std::array<rund::kernel::u32, 8u> bins{0u, 1u, 1u, 3u, 2u, 1u, 0u, 3u};
  const std::array<rund::kernel::u32, 4u> expected{2u, 3u, 1u, 2u};
  return Matches(pick, bins, expected);
}

bool MatchesParallelU32(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u32, 129u> bins{};
  for (std::size_t i = 0u; i < bins.size(); ++i) {
    bins[i] = static_cast<rund::kernel::u32>(i % 4u);
  }
  const std::array<rund::kernel::u32, 4u> expected{33u, 32u, 32u, 32u};
  return Matches(pick, bins, expected);
}

bool RejectsOutOfRangeBin(const rund::AccelDevice &pick) {
  const std::array<rund::kernel::u32, 8u> bins{0u, 1u, 4u, 3u, 2u, 1u, 0u, 3u};
  Resources resources = BuildResources(pick, bins, 4u);
  if (!resources.kernel.check.ok) {
    return false;
  }
  const auto bindings = Bindings(resources);
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(resources.context, resources.kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = bins.size(),
                                            .fresh_evidence = true,
                                        });
  return primitive::EvidenceReason(evidence, "compute_histogram_bin_invalid");
}

} // namespace node_accel_contract::histogram
