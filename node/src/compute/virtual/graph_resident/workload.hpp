#pragma once

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include "plan.hpp"
#include "src/compute/expression/state.hpp"
#include "src/compute/flow/state.hpp"
#include "src/compute/graph/compile/slice.hpp"
#include "src/compute/graph/compile/slice/semantic.hpp"
#include "src/compute/graph/state.hpp"
#include "src/hash/fnv.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace rund::compute::graph_resident_workload {
namespace detail {

template <std::uint64_t First, std::size_t Count, class Expression>
[[nodiscard]] constexpr auto add_stage(Expression value) {
  if constexpr (Count == 1u) {
    return value + First;
  } else {
    constexpr std::size_t left = Count / 2u;
    return add_stage<First, left>(value) +
           add_stage<First + left, Count - left>(value);
  }
}

template <class Expression>
[[nodiscard]] constexpr auto max_literal(const Expression &value) {
  using Base = std::remove_cvref_t<Expression>;
  using Literal = rund::compute::detail::StaticExpr<
      std::uint64_t,
      rund::compute::detail::StaticLiteralLike<std::uint64_t, Base>>;
  return Literal{rund::compute::detail::StaticLiteralLike<std::uint64_t, Base>{
      value, std::numeric_limits<std::uint64_t>::max()}};
}

} // namespace detail

struct Spec final {
  enum class Variant : std::uint8_t { Ordinary, AddSatUnsigned };

  using GraphPlan = ::rund::compute::detail::graph_resident_plan::Plan;

  static constexpr std::size_t InputCount = GraphPlan::InputCount;
  static constexpr std::size_t StageCount = GraphPlan::StageCount;
  static constexpr std::size_t FrameElements = 16u;
  static constexpr std::size_t PageCount = GraphPlan::PageCount;
  static constexpr std::size_t TailElements = 7u;
  static constexpr std::size_t ElementCount =
      PageCount * FrameElements - TailElements;
  static constexpr std::size_t LeafCount = 128u;
  static constexpr std::size_t BatchCount = GraphPlan::BatchCount;
  static constexpr std::size_t RunCount = BatchCount + 1u;
  static constexpr std::size_t InternalOwnerCount =
      GraphPlan::InternalOwnerCount;
  static constexpr std::size_t ResourceCount = GraphPlan::ResourceCount;
  static constexpr std::size_t PortCount = GraphPlan::PortCount;
  static constexpr std::uint32_t SchemaVersion = 1u;
  static constexpr std::size_t TotalClassCount = GraphPlan::TotalClassCount;
  static constexpr std::size_t FrameCapacity = GraphPlan::FrameCapacity;
  static constexpr const auto &Edges = GraphPlan::Edges;
  static constexpr const auto &StageInputs = GraphPlan::StageInputs;
  static constexpr const auto &StageOutputs = GraphPlan::StageOutputs;
  static constexpr const auto &StageInputPorts = GraphPlan::ReadCounts;
  static constexpr const auto &StageOutputPorts = GraphPlan::WriteCounts;
  static constexpr const auto &ResourceRoles = GraphPlan::ResourceRoles;
  static constexpr std::array<std::uint64_t, InputCount> Seeds{5u, 11u, 13u};
  static constexpr std::array<std::uint64_t, InputCount> Steps{17u, 29u, 43u};
  using Program = rund::compute::Program<std::uint64_t(
      std::uint64_t, std::uint64_t, std::uint64_t)>;

  [[nodiscard]] static bool valid_variant(const Program &program,
                                          const Variant variant) noexcept {
    const auto state = ::rund::compute::detail::FlowAccess::state(program);
    if (state == nullptr || state->canonical_graph == nullptr) {
      return false;
    }
    const auto &steps = state->canonical_graph->steps;
    std::size_t sat_count = 0u;
    bool sat_root = false;
    for (std::size_t stage = 0u; stage < steps.size(); ++stage) {
      const auto *const map =
          std::get_if<::rund::compute::detail::MapStep>(&steps[stage]);
      if (map == nullptr) {
        continue;
      }
      for (const auto &expression : map->expressions) {
        if (expression.state == nullptr || expression.node == 0u ||
            expression.node > expression.state->nodes.size()) {
          return false;
        }
        const auto &nodes = expression.state->nodes;
        for (std::size_t index = 0u; index < nodes.size(); ++index) {
          const auto &node = nodes[index];
          if (node.operation !=
              ::rund::compute::detail::ExprOp::AddSatUnsigned) {
            continue;
          }
          ++sat_count;
          if (variant != Variant::AddSatUnsigned || stage != 3u ||
              expression.node != index + 1u || node.right == 0u ||
              node.right > nodes.size() ||
              nodes[node.right - 1u].operation !=
                  ::rund::compute::detail::ExprOp::Constant ||
              nodes[node.right - 1u].bits !=
                  std::numeric_limits<std::uint64_t>::max()) {
            return false;
          }
          sat_root = true;
        }
      }
    }
    return variant == Variant::AddSatUnsigned ? sat_count == 1u && sat_root
                                              : sat_count == 0u;
  }

  [[nodiscard]] static bool
  validate(const Program &program,
           const Variant variant = Variant::Ordinary) noexcept {
    using namespace rund::compute;
    const auto slices = ::rund::compute::detail::graph_compile::
        compile_tiled_graph_pointwise_slices(
            ::rund::compute::detail::FlowAccess::state(program));
    const auto fused = ::rund::compute::detail::graph_compile::
        compile_service_free_map_program(
            ::rund::compute::detail::FlowAccess::state(program));
    if (!slices || fused || fused.reason() != Reason::ExpressionCapacity ||
        slices->resources.size() != ResourceCount ||
        slices->stages.size() != StageCount ||
        slices->input_resources.size() != InputCount ||
        slices->output_resource != ResourceCount) {
      return false;
    }
    for (std::size_t index = 0u; index < InputCount; ++index) {
      if (slices->input_resources[index] != index + 1u) {
        return false;
      }
    }
    for (std::size_t index = 0u; index < ResourceCount; ++index) {
      const auto &resource = slices->resources[index];
      const auto expected_kind = ResourceRoles[index] == 0u
                                     ? ::rund::compute::detail::graph_compile::
                                           SliceResourceKind::ExternalInput
                                 : ResourceRoles[index] == 2u
                                     ? ::rund::compute::detail::graph_compile::
                                           SliceResourceKind::ExternalOutput
                                     : ::rund::compute::detail::graph_compile::
                                           SliceResourceKind::Internal;
      if (resource.resource != index + 1u ||
          resource.type != ::rund::compute::detail::Type::U64 ||
          resource.count != FrameElements || resource.kind != expected_kind) {
        return false;
      }
    }
    for (std::size_t stage = 0u; stage < StageCount; ++stage) {
      const auto &slice = slices->stages[stage];
      if (slice.inputs.size() != StageInputPorts[stage] ||
          slice.outputs.size() != StageOutputPorts[stage]) {
        return false;
      }
      const std::size_t edge_start = stage == 0u   ? 0u
                                     : stage == 1u ? 1u
                                     : stage == 2u ? 2u
                                     : stage == 3u ? 4u
                                                   : 5u;
      const std::size_t output_edge = stage == 3u   ? 4u
                                      : stage == 4u ? 5u
                                                    : stage;
      for (std::size_t input = 0u; input < slice.inputs.size(); ++input) {
        if (slice.inputs[input] != StageInputs[stage][input] + 1u ||
            slice.inputs[input] != Edges[edge_start + input][0u] + 1u) {
          return false;
        }
      }
      if (slice.outputs[0u] != StageOutputs[stage] + 1u ||
          slice.outputs[0u] != Edges[output_edge][1u] + 1u) {
        return false;
      }
    }
    return valid_variant(program, variant);
  }

  [[nodiscard]] static constexpr std::uint64_t
  input_value(const std::size_t input, const std::size_t index) noexcept {
    return input < InputCount ? index * Steps[input] + Seeds[input] : 0u;
  }

  [[nodiscard]] static constexpr std::uint64_t
  stage_value(const std::uint64_t value, const std::uint64_t first) noexcept {
    constexpr std::uint64_t literal_sum = LeafCount * (LeafCount + 1u) / 2u;
    return value * LeafCount + literal_sum + (first - 1u) * LeafCount;
  }

  [[nodiscard]] static constexpr std::uint64_t
  expected_value(const std::size_t index,
                 const Variant variant = Variant::Ordinary) noexcept {
    const std::uint64_t x = stage_value(input_value(0u, index), 1u);
    const std::uint64_t y = stage_value(input_value(1u, index), LeafCount + 1u);
    const std::uint64_t z = stage_value(x, 2u * LeafCount + 1u) +
                            stage_value(y, 3u * LeafCount + 1u);
    const std::uint64_t w =
        variant == Variant::AddSatUnsigned
            ? std::numeric_limits<std::uint64_t>::max()
            : stage_value(input_value(2u, index), 4u * LeafCount + 1u);
    return stage_value(z, 5u * LeafCount + 1u) +
           stage_value(w, 6u * LeafCount + 1u);
  }

  [[nodiscard]] static std::uint64_t
  hash(const std::span<const std::uint64_t> values) noexcept {
    return ::rund::node::hash_detail::HashBytes(
        values.data(), values.size() * sizeof(std::uint64_t));
  }

  [[nodiscard]] static std::uint64_t
  expected_hash(const Variant variant = Variant::Ordinary) noexcept {
    std::array<std::uint64_t, ElementCount> values{};
    for (std::size_t index = 0u; index < ElementCount; ++index) {
      values[index] = expected_value(index, variant);
    }
    return hash(std::span{values});
  }

  [[nodiscard]] static constexpr std::uint64_t
  topology_digest(const Variant variant = Variant::Ordinary) noexcept {
    constexpr std::array<std::uint64_t, 8u> shape{
        InputCount,   StageCount, FrameElements, PageCount,
        TailElements, LeafCount,  BatchCount,    InternalOwnerCount};
    std::uint64_t result = 1469598103934665603ull;
    for (const std::uint64_t value : shape) {
      result ^= value;
      result *= 1099511628211ull;
    }
    for (const std::uint64_t value : Seeds) {
      result ^= value;
      result *= 1099511628211ull;
    }
    for (const std::uint64_t value : Steps) {
      result ^= value;
      result *= 1099511628211ull;
    }
    const auto mix = [&](const std::uint64_t value) constexpr {
      result ^= value;
      result *= 1099511628211ull;
    };
    mix(SchemaVersion);
    mix(ResourceCount);
    mix(PortCount);
    for (const auto &edge : Edges) {
      mix(edge[0u]);
      mix(edge[1u]);
    }
    for (const std::uint8_t count : StageInputPorts) {
      mix(count);
    }
    for (const std::uint8_t count : StageOutputPorts) {
      mix(count);
    }
    for (const auto &inputs : StageInputs) {
      mix(inputs[0u]);
      mix(inputs[1u]);
    }
    for (const std::uint8_t output : StageOutputs) {
      mix(output);
    }
    for (const std::uint8_t role : ResourceRoles) {
      mix(role);
    }
    if (variant == Variant::AddSatUnsigned) {
      mix(0xadd5a7u);
    }
    return result == 0u ? 1u : result;
  }

  [[nodiscard]] static constexpr std::uint64_t input_digest() noexcept {
    std::uint64_t result = 1469598103934665603ull;
    for (std::size_t input = 0u; input < InputCount; ++input) {
      for (std::size_t index = 0u; index < ElementCount; ++index) {
        result ^= input_value(input, index);
        result *= 1099511628211ull;
      }
    }
    return result == 0u ? 1u : result;
  }

  [[nodiscard]] static constexpr std::uint64_t
  workload_digest(const Variant variant = Variant::Ordinary) noexcept {
    std::uint64_t result = 1469598103934665603ull;
    const auto mix = [&](const std::uint64_t value) constexpr {
      result ^= value;
      result *= 1099511628211ull;
    };
    mix(ElementCount);
    mix(LeafCount);
    mix(SchemaVersion);
    mix(topology_digest(variant));
    for (const std::uint64_t seed : Seeds) {
      mix(seed);
    }
    for (const std::uint64_t step : Steps) {
      mix(step);
    }
    for (std::size_t index = 0u; index < ElementCount; ++index) {
      mix(expected_value(index, variant));
    }
    return result == 0u ? 1u : result;
  }

  template <Variant V>
  [[nodiscard]] static rund::compute::Result<Program>
  build_variant(const rund::compute::Device &device) {
    return rund::compute::on(device)
        .input<std::uint64_t>(FrameElements)
        .zip_input<std::uint64_t>(FrameElements)
        .zip_input<std::uint64_t>(FrameElements)
        .branch([](auto a, auto b, auto c) {
          const auto x = a.map("graph-resident-x", [](auto value) {
            return detail::add_stage<1u, LeafCount>(value);
          });
          const auto y = b.map("graph-resident-y", [](auto value) {
            return detail::add_stage<LeafCount + 1u, LeafCount>(value);
          });
          const auto z =
              zip(x, y).map("graph-resident-z", [](auto left, auto right) {
                return detail::add_stage<2u * LeafCount + 1u, LeafCount>(left) +
                       detail::add_stage<3u * LeafCount + 1u, LeafCount>(right);
              });
          const auto w = c.map("graph-resident-w", [](auto value) {
            const auto base =
                detail::add_stage<4u * LeafCount + 1u, LeafCount>(value);
            if constexpr (V == Variant::AddSatUnsigned) {
              return rund::compute::add_sat_unsigned(base,
                                                     detail::max_literal(base));
            } else {
              return base;
            }
          });
          return zip(z, w).map(
              "graph-resident-output", [](auto left, auto right) {
                return detail::add_stage<5u * LeafCount + 1u, LeafCount>(left) +
                       detail::add_stage<6u * LeafCount + 1u, LeafCount>(right);
              });
        })
        .compile();
  }

  [[nodiscard]] static rund::compute::Result<Program>
  build(const rund::compute::Device &device,
        const Variant variant = Variant::Ordinary) {
    return variant == Variant::AddSatUnsigned
               ? build_variant<Variant::AddSatUnsigned>(device)
               : build_variant<Variant::Ordinary>(device);
  }
};

static_assert(Spec::ElementCount == 73u);
static_assert(Spec::TotalClassCount == 7u);

} // namespace rund::compute::graph_resident_workload
