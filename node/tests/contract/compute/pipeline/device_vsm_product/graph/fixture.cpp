#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund_node_test_device_vsm_product::graph_test {
namespace {

enum class GraphCaseKind : std::uint8_t {
  Product,
  TypedProduct,
  FusedProduct,
  FusedTypedProduct,
  BranchProduct,
  Overflow,
};

bool PrepareGraph(const rund::compute::Backend backend,
                  const std::uint64_t pages, const GraphCaseKind kind,
                  const rund::kernel::ReduceOp operation,
                  PreparedGraph &prepared, bool &unavailable) {
  using namespace rund::compute;
  prepared = {};
  unavailable = false;
  if (pages < 2u) {
    return false;
  }
  const std::uint64_t elements = pages * PageElements - 3u;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  const bool fused = kind == GraphCaseKind::FusedProduct;
  const bool typed = kind == GraphCaseKind::TypedProduct ||
                     kind == GraphCaseKind::FusedTypedProduct;
  const bool fused_typed = kind == GraphCaseKind::FusedTypedProduct;
  const bool overflow = kind == GraphCaseKind::Overflow;
  const bool branch = kind == GraphCaseKind::BranchProduct;
  auto program = [&]() {
    if (branch) {
      return on(*opened)
          .input<std::uint64_t>(PageElements)
          .branch([](auto values) {
            const auto left = values.map("device-vsm-graph-branch-left",
                                         [](auto value) { return value + 1u; });
            const auto right =
                values.map("device-vsm-graph-branch-right", [](auto value) {
                  return value * std::uint64_t{2u};
                });
            return zip(left, right)
                .map("device-vsm-graph-branch-join",
                     [](auto first, auto second) { return first + second; })
                .reduce(Reduce::Sum);
          })
          .compile();
    }
    if (fused) {
      return on(*opened)
          .map<std::uint64_t>("device-vsm-graph-map-reduce", PageElements,
                              [](auto value) { return value + 1u; })
          .map("device-vsm-graph-map-fused",
               [](auto value) { return value + 2u; })
          .reduce(Reduce::Sum)
          .compile();
    }
    if (fused_typed) {
      return on(*opened)
          .map<std::uint64_t>("device-vsm-graph-typed-map-prefix", PageElements,
                              [](auto value) { return value ^ 0x55ull; })
          .map("device-vsm-graph-typed-map-fused",
               [](auto value) { return value * 3ull + 7ull; })
          .reduce(Reduce::Sum)
          .compile();
    }
    if (typed) {
      return on(*opened)
          .map<std::uint64_t>(
              "device-vsm-graph-typed-map-reduce", PageElements,
              [](auto value) { return (value ^ 0x55ull) * 3ull + 7ull; })
          .reduce(Reduce::Sum)
          .compile();
    }
    auto mapped = on(*opened).map<std::uint64_t>(
        "device-vsm-graph-map-reduce", PageElements,
        [](auto value) { return value + 1u; });
    if (operation == rund::kernel::ReduceOp::CountNonzero) {
      return std::move(mapped).count().compile();
    }
    if (operation == rund::kernel::ReduceOp::Min) {
      return std::move(mapped).reduce(Reduce::Min).compile();
    }
    if (operation == rund::kernel::ReduceOp::Max) {
      return std::move(mapped).reduce(Reduce::Max).compile();
    }
    return std::move(mapped).reduce(Reduce::Sum).compile();
  }();
  prepared.input = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      elements * sizeof(std::uint64_t));
  prepared.output = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      sizeof(std::uint64_t));
  std::vector<std::uint64_t> values(static_cast<std::size_t>(elements));
  prepared.operation = operation;
  prepared.expected = operation == rund::kernel::ReduceOp::Min
                          ? std::numeric_limits<std::uint64_t>::max()
                          : 0u;
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = overflow && index < 2u
                        ? std::numeric_limits<std::uint64_t>::max() - 1u
                        : (index * 19u + pages) % 101u;
    if (!overflow) {
      const std::uint64_t mapped = branch ? values[index] * 3u + 1u
                                   : typed
                                       ? (values[index] ^ 0x55ull) * 3ull + 7ull
                                       : values[index] + (fused ? 3u : 1u);
      if (operation == rund::kernel::ReduceOp::Sum) {
        prepared.expected += mapped;
      } else if (operation == rund::kernel::ReduceOp::CountNonzero) {
        prepared.expected += static_cast<std::uint64_t>(mapped != 0u);
      } else if (operation == rund::kernel::ReduceOp::Min) {
        prepared.expected = std::min(prepared.expected, mapped);
      } else {
        prepared.expected = std::max(prepared.expected, mapped);
      }
    }
  }
  prepared.input_bytes = elements * sizeof(std::uint64_t);
  const bool seeded = prepared.input->seed(std::as_bytes(std::span{values}));
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.input);
  auto output = detail::make_virtual_buffer(
      1u, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.output);
  const auto program_state =
      program ? detail::ProgramAccess::state(*program) : nullptr;
  auto state =
      program_state != nullptr && input && output
          ? detail::prepare_virtual_pipeline(
                program_state, std::move(input).value(),
                std::move(output).value(), ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!seeded || !state || state.value() == nullptr) {
    std::fprintf(stderr,
                 "DeviceVsm Graph prepare op=%u program=%u seeded=%u "
                 "state=%u reason=%u\n",
                 static_cast<unsigned>(operation),
                 static_cast<unsigned>(program_state != nullptr),
                 static_cast<unsigned>(seeded),
                 static_cast<unsigned>(static_cast<bool>(state)),
                 static_cast<unsigned>(state.reason()));
    return false;
  }
  prepared.authored_nodes = program_state->graph_info.authored_nodes;
  prepared.lowered_nodes = program_state->graph_info.lowered_nodes;
  prepared.state = std::move(state).value();
  const std::size_t stages =
      prepared.state->pipeline == nullptr ||
              prepared.state->pipeline->residency == nullptr
          ? 0u
          : prepared.state->pipeline->residency->tiled_graph().stages().size();
  prepared.stage_count = static_cast<std::uint32_t>(stages);
  const bool valid =
      prepared.state->geometry.route == detail::VirtualRoute::GraphReduction &&
      prepared.state->pipeline != nullptr &&
      rund::compute::detail::graph_terminal_pipeline(*prepared.state, 0u) !=
          nullptr &&
      stages == (branch ? 4u : 2u) &&
      (branch ? prepared.state->device_vsm_semantic_pipeline != nullptr
              : prepared.state->device_vsm_semantic_pipeline == nullptr);
  if (!valid) {
    std::fprintf(stderr,
                 "DeviceVsm Graph prepare op=%u route=%u stages=%zu "
                 "pipelines=%zu authored=%llu lowered=%llu\n",
                 static_cast<unsigned>(operation),
                 static_cast<unsigned>(prepared.state->geometry.route), stages,
                 prepared.state->graph_pipelines.size(),
                 static_cast<unsigned long long>(prepared.authored_nodes),
                 static_cast<unsigned long long>(prepared.lowered_nodes));
  }
  return valid;
}

} // namespace

bool PrepareGraphProduct(const rund::compute::Backend backend,
                         const std::uint64_t pages, PreparedGraph &prepared,
                         bool &unavailable) {
  return PrepareGraph(backend, pages, GraphCaseKind::Product,
                      rund::kernel::ReduceOp::Sum, prepared, unavailable);
}

bool PrepareFusedGraphProduct(const rund::compute::Backend backend,
                              const std::uint64_t pages,
                              PreparedGraph &prepared, bool &unavailable) {
  return PrepareGraph(backend, pages, GraphCaseKind::FusedProduct,
                      rund::kernel::ReduceOp::Sum, prepared, unavailable);
}

bool PrepareTypedGraphProduct(const rund::compute::Backend backend,
                              const std::uint64_t pages,
                              PreparedGraph &prepared, bool &unavailable) {
  return PrepareGraph(backend, pages, GraphCaseKind::TypedProduct,
                      rund::kernel::ReduceOp::Sum, prepared, unavailable);
}

bool PrepareFusedTypedGraphProduct(const rund::compute::Backend backend,
                                   const std::uint64_t pages,
                                   PreparedGraph &prepared, bool &unavailable) {
  return PrepareGraph(backend, pages, GraphCaseKind::FusedTypedProduct,
                      rund::kernel::ReduceOp::Sum, prepared, unavailable);
}

bool PrepareBranchGraphProduct(const rund::compute::Backend backend,
                               const std::uint64_t pages,
                               PreparedGraph &prepared, bool &unavailable) {
  return PrepareGraph(backend, pages, GraphCaseKind::BranchProduct,
                      rund::kernel::ReduceOp::Sum, prepared, unavailable);
}

bool PrepareGraphOverflow(const rund::compute::Backend backend,
                          const std::uint64_t pages, PreparedGraph &prepared,
                          bool &unavailable) {
  return PrepareGraph(backend, pages, GraphCaseKind::Overflow,
                      rund::kernel::ReduceOp::Sum, prepared, unavailable);
}

bool PrepareGraphCountNonzero(const rund::compute::Backend backend,
                              const std::uint64_t pages,
                              PreparedGraph &prepared, bool &unavailable) {
  return PrepareGraph(backend, pages, GraphCaseKind::Product,
                      rund::kernel::ReduceOp::CountNonzero, prepared,
                      unavailable);
}

bool PrepareGraphMin(const rund::compute::Backend backend,
                     const std::uint64_t pages, PreparedGraph &prepared,
                     bool &unavailable) {
  return PrepareGraph(backend, pages, GraphCaseKind::Product,
                      rund::kernel::ReduceOp::Min, prepared, unavailable);
}

bool PrepareGraphMax(const rund::compute::Backend backend,
                     const std::uint64_t pages, PreparedGraph &prepared,
                     bool &unavailable) {
  return PrepareGraph(backend, pages, GraphCaseKind::Product,
                      rund::kernel::ReduceOp::Max, prepared, unavailable);
}

bool ExactGraphOutput(const PreparedGraph &prepared) noexcept {
  std::array<std::byte, sizeof(std::uint64_t)> bytes{};
  std::uint64_t observed = 0u;
  if (prepared.output == nullptr || !prepared.output->observe(bytes)) {
    return false;
  }
  std::memcpy(&observed, bytes.data(), sizeof(observed));
  return observed == prepared.expected;
}

} // namespace rund_node_test_device_vsm_product::graph_test

#endif
