#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/value.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/visibility.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <accel/graph/factory/buffer/read.hpp>
#include <accel/graph/factory/buffer/write.hpp>
#include <accel/graph/factory/map.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

#include <array>

namespace node_accel_contract::kernel_case {

[[nodiscard]] std::array<rund::AccelRunBinding, 2u>
Bindings(const ResidentRunFixture &fixture) noexcept {
  return std::array<rund::AccelRunBinding, 2u>{
      rund::AccelRunBinding{
          .buffer = &fixture.input,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &fixture.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
}

[[nodiscard]] rund::AccelRun
RunRequest(const std::array<rund::AccelRunBinding, 2u> &bindings,
           const std::size_t tile_count, const bool fresh_evidence) noexcept {
  return rund::AccelRun{
      .bindings = bindings.data(),
      .binding_count = bindings.size(),
      .tile_count = tile_count,
      .fresh_evidence = fresh_evidence,
  };
}

[[nodiscard]] ResidentRunFixture
MakeResidentRunFixture(const rund::AccelContext &context) {
  ResidentRunFixture fixture{};
  fixture.context = context;
  fixture.input = rund::node::accel::CreateAccelBuffer(
      context, BufferDesc(rund::BufferUsage::ReadOnly));
  fixture.output = rund::node::accel::CreateAccelBuffer(
      context, BufferDesc(rund::BufferUsage::WriteOnly));
  if (!fixture.input.check.ok || !fixture.output.check.ok) {
    return fixture;
  }

  const rund::compute_dsl::ComputeOp op = BuildFixedLane32Op();
  if (!op.ok()) {
    return fixture;
  }
  std::array<rund::AccelGraphBufferRef, 2u> refs{};
  std::array<rund::AccelGraphNode, 1u> nodes{};
  refs = {
      rund::AccelRead(BufferDesc(rund::BufferUsage::ReadOnly), "input",
                      rund::GraphBufferVisibility::External, 1u),
      rund::AccelWrite(BufferDesc(rund::BufferUsage::WriteOnly), "output",
                       rund::GraphBufferVisibility::External, 2u),
  };
  nodes = {
      rund::AccelMap(op.ir(), refs.data(), refs.size(), fixture.output.count)};
  const rund::AccelGraph graph{
      .nodes = nodes.data(),
      .node_count = nodes.size(),
      .scalar = op.ir().scalar,
      .domain = op.ir().domain,
      .fixed_format = op.ir().fixed_format,
  };
  fixture.kernel = rund::node::accel::CompileAccelKernel(context, graph);
  if (!fixture.kernel.check.ok || fixture.kernel.kernel_id == 0u) {
    return fixture;
  }

  fixture.host_input = {-12, -1, 0, 1, 7, 31, 63, 127};
  std::array<rund::kernel::i32, 8u> expected{};
  for (std::size_t index = 0u; index < fixture.host_input.size(); ++index) {
    expected[index] = fixture.host_input[index] + 7;
  }
  fixture.expected_hash = HashFixedLane32(expected);
  const rund::AccelCheck upload = rund::node::accel::UploadAccelBuffer(
      context, fixture.input, fixture.host_input.data(),
      sizeof(fixture.host_input));
  fixture.ok = upload.ok;
  return fixture;
}

} // namespace node_accel_contract::kernel_case
