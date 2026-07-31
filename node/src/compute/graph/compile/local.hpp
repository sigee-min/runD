#pragma once

#include "../../program/state.hpp"
#include "../../status.hpp"
#include "../state.hpp"

#include <accel/graph/node.hpp>
#include <kernel/program/compute/dsl.hpp>
#include <kernel/program/compute/graph/schema.hpp>
#include <rund/compute/graph/info.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace rund::compute::detail::graph_compile {

struct Lowering final {
  std::shared_ptr<GraphState> graph{};
  const graph::Info *layout{};
  std::shared_ptr<ProgramState> program{};
  std::vector<compute_dsl::ComputeOp> operations{};
  std::vector<std::vector<kernel::GraphBufferRef>> cpu_refs{};
  std::vector<kernel::GraphNode> cpu_nodes{};
  std::vector<std::vector<rund::AccelGraphBufferRef>> accel_refs{};
  std::vector<rund::AccelGraphNode> accel_nodes{};
  std::vector<kernel::u64> outputs{};
  std::vector<std::uint8_t> barriers{};
  std::size_t operation{};
  Status runtime{Status::success()};

  [[nodiscard]] bool cpu() const noexcept {
    return program != nullptr && program->cpu_graph != nullptr;
  }
};

[[nodiscard]] Result<std::shared_ptr<ProgramState>>
prepare(const std::shared_ptr<GraphState> &graph, const graph::Info &layout);

[[nodiscard]] Status bind(Lowering &lowering, std::size_t index);
[[nodiscard]] Status map(Lowering &lowering, std::size_t index,
                         const MapStep &step);
[[nodiscard]] Status scan(Lowering &lowering, std::size_t index,
                          const ScanStep &step);
[[nodiscard]] Status primitive(Lowering &lowering, std::size_t index,
                               const GraphPrimitive &step);
[[nodiscard]] Status lower(Lowering &lowering);

[[nodiscard]] Result<std::shared_ptr<ProgramState>> finish(Lowering &&lowering);

[[nodiscard]] std::uint64_t logical(const ProgramState &program,
                                    std::uint32_t value) noexcept;
[[nodiscard]] std::optional<rund::AccelGraphBufferRef>
resident(const Lowering &lowering, std::size_t node, std::uint32_t value,
         kernel::BufferRole role, const char *name);

} // namespace rund::compute::detail::graph_compile
