#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>
#include <accel/runtime.hpp>

#include <accel/graph/factory/map.hpp>

#include "local.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/context.hpp>

#include <vector>

namespace node_accel_contract::kernel_case {

[[nodiscard]] bool
MultiWriteResidentRunMatches(const rund::AccelContext &context) {
  constexpr std::size_t count = 8u;
  const rund::AccelBuffer input = rund::node::accel::CreateAccelBuffer(
      context, BufferDesc(rund::BufferUsage::ReadOnly));
  const rund::AccelBuffer plus = rund::node::accel::CreateAccelBuffer(
      context, BufferDesc(rund::BufferUsage::WriteOnly));
  const rund::AccelBuffer times = rund::node::accel::CreateAccelBuffer(
      context, BufferDesc(rund::BufferUsage::WriteOnly));
  const rund::compute_dsl::ComputeOp op = BuildMultiWriteFixedLane32Op();
  if (!input.check.ok || !plus.check.ok || !times.check.ok || !op.ok()) {
    return false;
  }
  std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelGraphBufferRef{.buffer = &input,
                                .role = rund::kernel::BufferRole::Read,
                                .binding_name = "input"},
      rund::AccelGraphBufferRef{.buffer = &plus,
                                .role = rund::kernel::BufferRole::Write,
                                .binding_name = "plus"},
      rund::AccelGraphBufferRef{.buffer = &times,
                                .role = rund::kernel::BufferRole::Write,
                                .binding_name = "times"}};
  std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelMap(op.ir(), refs.data(), refs.size(), count)};
  const rund::AccelGraph graph{
      .nodes = nodes.data(),
      .node_count = nodes.size(),
      .scalar = op.ir().scalar,
      .domain = op.ir().domain,
      .fixed_format = op.ir().fixed_format,
  };
  const rund::AccelKernel kernel =
      rund::node::accel::CompileAccelKernel(context, graph);
  if (!kernel.check.ok) {
    return false;
  }

  const std::array<rund::kernel::i32, count> source{-7, -3, -1, 0,
                                                    2,  5,  11, 29};
  if (!rund::node::accel::UploadAccelBuffer(context, input, source.data(),
                                            sizeof(source))
           .ok) {
    return false;
  }
  const std::array<rund::AccelRunBinding, 3u> bindings{
      rund::AccelRunBinding{.buffer = &input,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &plus,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &times,
                            .role = rund::kernel::BufferRole::Write}};
  const rund::AccelEvidence run =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = count,
                                            .fresh_evidence = true,
                                        });
  if (!run.ok || run.device_to_host_bytes != 0u) {
    return false;
  }
  std::array<rund::kernel::i32, count> plus_host{};
  std::array<rund::kernel::i32, count> times_host{};
  if (!rund::node::accel::DownloadAccelBuffer(context, plus, plus_host.data(),
                                              sizeof(plus_host))
           .ok ||
      !rund::node::accel::DownloadAccelBuffer(context, times, times_host.data(),
                                              sizeof(times_host))
           .ok) {
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    if (plus_host[index] != source[index] + 1 ||
        times_host[index] != source[index] * 3) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
WindowedResidentRunMatches(const rund::AccelContext &context) {
  constexpr std::uint64_t param_bytes = sizeof(rund::kernel::i32);
  constexpr std::uint64_t tile_bytes = 2u * sizeof(rund::kernel::i32);
  if (context.caps.max_window_tiles == 0u ||
      context.caps.staging_bytes < param_bytes + tile_bytes) {
    return false;
  }

  const std::uint64_t staging_tiles =
      (context.caps.staging_bytes - param_bytes) / tile_bytes;
  const std::uint64_t window_tiles =
      context.caps.max_window_tiles < staging_tiles
          ? context.caps.max_window_tiles
          : staging_tiles;
  const std::size_t tile_count = static_cast<std::size_t>(window_tiles + 3u);
  const rund::AccelBuffer input = rund::node::accel::CreateAccelBuffer(
      context, BufferDesc(rund::BufferUsage::ReadOnly, tile_count));
  const rund::AccelBuffer output = rund::node::accel::CreateAccelBuffer(
      context, BufferDesc(rund::BufferUsage::WriteOnly, tile_count));
  if (!input.check.ok || !output.check.ok) {
    return false;
  }

  const rund::compute_dsl::ComputeOp op =
      BuildFixedLane32OpForTiles(tile_count);
  if (!op.ok()) {
    return false;
  }
  std::array<rund::AccelGraphBufferRef, 2u> refs{};
  std::array<rund::AccelGraphNode, 1u> nodes{};
  const rund::AccelGraph graph = GraphFor(op.ir(), input, output, refs, nodes);
  const rund::AccelKernel kernel =
      rund::node::accel::CompileAccelKernel(context, graph);
  if (!kernel.check.ok || kernel.kernel_id == 0u) {
    return false;
  }

  std::vector<rund::kernel::i32> host(tile_count);
  std::vector<rund::kernel::i32> expected(tile_count);
  for (std::size_t index = 0u; index < tile_count; ++index) {
    host[index] = static_cast<rund::kernel::i32>(static_cast<int>(index) - 17);
    expected[index] = host[index] + 7;
  }
  if (!rund::node::accel::UploadAccelBuffer(context, input, host.data(),
                                            host.size() * sizeof(host[0]))
           .ok) {
    return false;
  }

  const std::array<rund::AccelRunBinding, 2u> bindings{
      rund::AccelRunBinding{
          .buffer = &input,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const std::uint64_t expected_dispatch_count =
      (static_cast<std::uint64_t>(tile_count) + window_tiles - 1u) /
      window_tiles;
  const rund::AccelEvidence run = rund::node::accel::RunAccelKernel(
      context, kernel, RunRequest(bindings, tile_count, true));
  const rund::RuntimeStats after_run =
      rund::node::accel::ReadRuntimeStats(context.pick);
  if (!run.ok || run.dispatch_count == 0u ||
      run.dispatch_count > expected_dispatch_count ||
      run.original_dispatch_count != expected_dispatch_count ||
      run.final_dispatch_count == 0u ||
      run.final_dispatch_count > run.original_dispatch_count ||
      run.dispatch_count != run.final_dispatch_count ||
      run.device_to_host_bytes != 0u || after_run.device_to_host_bytes != 0u) {
    return false;
  }

  std::vector<rund::kernel::i32> downloaded(tile_count);
  return rund::node::accel::DownloadAccelBuffer(
             context, output, downloaded.data(),
             downloaded.size() * sizeof(downloaded[0]))
             .ok &&
         downloaded == expected;
}

} // namespace node_accel_contract::kernel_case
