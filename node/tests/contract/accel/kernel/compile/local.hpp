#pragma once

#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/value.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract::kernel_case::compile {

struct Fixture {
  rund::AccelDevice pick{};
  rund::AccelContext context{};
  rund::AccelBuffer input{};
  rund::AccelBuffer output{};
  rund::compute_dsl::ComputeOp op{};
  std::array<rund::AccelGraphBufferRef, 2u> refs{};
  std::array<rund::AccelGraphNode, 1u> nodes{};
  rund::AccelGraph graph{};
  rund::AccelKernel first{};
  rund::AccelKernel second{};
};

[[nodiscard]] Fixture MakeFixture(const rund::AccelDevice &pick);
[[nodiscard]] bool Prepare(Fixture &fixture);
[[nodiscard]] bool SignatureAndSupportRejects();
[[nodiscard]] bool IdentityIsStable(const Fixture &fixture);
[[nodiscard]] bool InitializationIsGraphOwned(const Fixture &fixture);
[[nodiscard]] bool GraphIdIgnoresBufferIds(const Fixture &fixture);
[[nodiscard]] bool MultiNodeGraphIdMatchesKernel(const Fixture &fixture);
[[nodiscard]] bool TwoReadBindingOrderRejects(const Fixture &fixture);
[[nodiscard]] bool TamperedSupportRejects(const Fixture &fixture);
[[nodiscard]] bool ForeignBufferRejects(const Fixture &fixture);
[[nodiscard]] bool LogicalBufferIdentityRejects(const Fixture &fixture);
[[nodiscard]] bool UnsupportedAndInvalidGraphRejects(const Fixture &fixture);

} // namespace node_accel_contract::kernel_case::compile
