#pragma once

#include <kernel/program/compute/partition/plan.hpp>

#include <accel/api.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>

#include "../primitive/local.hpp"

#include <array>
#include <string_view>

namespace node_accel_contract::partition {

struct Fixture {
  rund::AccelContext context{};
  rund::AccelBuffer flags{};
  rund::AccelBuffer values{};
  rund::AccelBuffer output{};
  rund::kernel::PartitionDesc desc{};
  rund::kernel::PartitionPlan plan{};
  rund::kernel::PartitionHash hash{};
  std::array<rund::AccelGraphBufferRef, 3u> refs{};
  std::array<rund::AccelGraphNode, 1u> nodes{};
  rund::AccelGraph graph{};
};

void Bind(Fixture &fixture) noexcept;
[[nodiscard]] Fixture Make(const rund::AccelDevice &pick);
[[nodiscard]] bool CompileReason(const rund::AccelContext &context,
                                 const rund::AccelGraph &graph,
                                 std::string_view reason);
[[nodiscard]] bool MatchesReference(const rund::AccelDevice &pick);
[[nodiscard]] bool CompileContract();

} // namespace node_accel_contract::partition
