#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/abi/graph.hpp>

#include "src/compute/expression/state.hpp"
#include "src/compute/flow/state.hpp"
#include "src/compute/graph/state.hpp"
#include "src/compute/map/build.hpp"
#include "src/compute/program/state.hpp"
#include "tests/contract/target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <vector>

namespace compute_map_contract {
int Run() {
  if (!Canonical()) {
    return 17;
  }
  if (!Replay()) {
    return 19;
  }
  auto device = rund::compute::open(rund::compute::Target::cpu());
  if (!device) {
    return 1;
  }

  auto raw_device = rund::compute::detail::open_cpu(0u);
  if (!raw_device) {
    return 8;
  }
  if (!Liveness(raw_device.value())) {
    return 18;
  }
  using rund::compute::detail::ExprNode;
  using rund::compute::detail::ExprOp;
  using rund::compute::detail::ExprRef;
  using rund::compute::detail::ExprState;
  using rund::compute::detail::Type;
  {
    auto graph = rund::compute::detail::make_graph(raw_device.value(),
                                                   "identity-forward", 4u);
    const std::uint32_t source =
        rund::compute::detail::graph_input(graph, Type::I32);
    const auto state = rund::compute::detail::make_expr();
    const ExprRef expression =
        rund::compute::detail::input(state, Type::I32, 0u);
    const std::array inputs{source};
    const std::uint32_t forwarded = rund::compute::detail::graph_map(
        graph, inputs, expression, "identity-forward");
    if (forwarded != source || graph->values.size() != 1u ||
        !graph->steps.empty()) {
      return 16;
    }
    rund::compute::detail::graph_output(graph, forwarded);
    if (!graph->status || graph->values.size() != 2u ||
        graph->steps.size() != 1u || graph->outputs.size() != 1u ||
        graph->outputs.front() == source) {
      return 16;
    }
  }
  const auto forged_graph = [&](const ExprNode node) {
    auto state = std::make_shared<ExprState>();
    state->nodes.push_back(
        ExprNode{.operation = ExprOp::Input, .type = Type::I32, .left = 0u});
    state->nodes.push_back(node);
    auto graph = rund::compute::detail::make_graph(raw_device.value(),
                                                   "forged-graph", 1u);
    const std::uint32_t source =
        rund::compute::detail::graph_input(graph, Type::I32);
    const std::array inputs{source};
    const std::uint32_t output = rund::compute::detail::graph_map(
        graph, inputs, ExprRef{std::move(state), 2u, Type::I32, {}});
    rund::compute::detail::graph_output(graph, output);
    const std::array types{Type::I32};
    return rund::compute::detail::compile_graph(graph, types, types);
  };
  const Type unknown = static_cast<Type>(0xffu);
  auto invalid_type = forged_graph(ExprNode{
      .operation = ExprOp::Divide, .type = unknown, .left = 1u, .right = 1u});
  auto invalid_operation =
      forged_graph(ExprNode{.operation = static_cast<ExprOp>(0xffu),
                            .type = Type::I32,
                            .left = 1u,
                            .right = 1u});
  if (invalid_type || invalid_operation ||
      invalid_type.error() != "compute_domain_type_mismatch" ||
      invalid_operation.error() != "compute_domain_type_mismatch") {
    return 9;
  }

  auto flow_expression = std::make_shared<ExprState>();
  flow_expression->nodes.push_back(
      ExprNode{.operation = ExprOp::Input, .type = Type::I32, .left = 0u});
  flow_expression->nodes.push_back(
      ExprNode{.operation = static_cast<ExprOp>(0xffu),
               .type = Type::I32,
               .left = 1u,
               .right = 1u});
  auto invalid_flow = rund::compute::detail::make_flow(
      rund::compute::Target::cpu(), Type::I32, 1u);
  rund::compute::detail::flow_map(
      invalid_flow, "forged-flow",
      ExprRef{std::move(flow_expression), 2u, Type::I32, {}});
  auto invalid_flow_program = rund::compute::detail::compile_flow(invalid_flow);
  if (invalid_flow_program ||
      invalid_flow_program.error() != "compute_expression_invalid") {
    return 10;
  }

  using Fixed = rund::compute::Fixed<16u, 16u>;
  auto fixed_source = rund::compute::detail::make_flow(
      rund::compute::Target::cpu(), Type::FixedLane32, 1u,
      rund::compute::detail::storage_format<Fixed>());
  if (rund::compute::detail::flow_retype(fixed_source, 1u, Type::U32) != 0u) {
    return 11;
  }
  auto fixed_source_program = rund::compute::detail::compile_flow(fixed_source);
  if (fixed_source_program ||
      fixed_source_program.error() != "compute_fixed_format_mismatch") {
    return 11;
  }

  auto fixed_output = rund::compute::detail::make_flow(
      rund::compute::Target::cpu(), Type::U32, 1u);
  if (rund::compute::detail::flow_retype(fixed_output, 1u, Type::FixedLane32) !=
      0u) {
    return 12;
  }
  auto fixed_output_program = rund::compute::detail::compile_flow(fixed_output);
  if (fixed_output_program ||
      fixed_output_program.error() != "compute_fixed_format_mismatch") {
    return 12;
  }

  using AlternateFixed = rund::compute::Fixed<17u, 15u>;
  auto forged_fixed_flow = rund::compute::detail::make_flow(
      rund::compute::Target::cpu(), Type::FixedLane32, 1u,
      rund::compute::detail::storage_format<Fixed>());
  auto forged_fixed_expression = std::make_shared<ExprState>();
  forged_fixed_expression->nodes.push_back(ExprNode{
      .operation = ExprOp::Input,
      .type = Type::FixedLane32,
      .fixed_format = rund::compute::detail::storage_format<AlternateFixed>(),
      .left = 0u});
  rund::compute::detail::flow_map(
      forged_fixed_flow, "forged-fixed-format",
      ExprRef{std::move(forged_fixed_expression), 1u, Type::FixedLane32,
              rund::compute::detail::storage_format<AlternateFixed>()});
  auto forged_fixed_program =
      rund::compute::detail::compile_flow(forged_fixed_flow);
  if (forged_fixed_program ||
      forged_fixed_program.error() != "compute_graph_type_mismatch") {
    return 13;
  }

  auto forged_boundary_state = rund::compute::detail::make_expr();
  auto forged_boundary_value =
      rund::compute::detail::input(forged_boundary_state, Type::I32, 0u);
  forged_boundary_value = rund::compute::detail::boundary_mask_expr(
      std::move(forged_boundary_value), Type::U32,
      rund::compute::detail::FixedFormat{.fraction_bits = 1u});
  auto forged_boundary_flow =
      rund::compute::detail::make_flow_on(raw_device.value(), Type::I32, 1u);
  rund::compute::detail::flow_map(forged_boundary_flow,
                                  "forged-boundary-policy",
                                  std::move(forged_boundary_value));
  auto forged_boundary =
      rund::compute::detail::compile_flow(forged_boundary_flow);
  if (forged_boundary ||
      forged_boundary.error() != "compute_expression_type_mismatch") {
    return 14;
  }

  auto twice = rund::compute::on(device.value())
                   .map<std::int32_t>("twice", 4,
                                      [](auto value) { return value * 2 + 5; })
                   .compile();
  if (!twice || twice.value().size() != 4) {
    if (!twice) {
      std::fprintf(stderr, "compute map compile failed: %.*s\n",
                   static_cast<int>(twice.error().size()),
                   twice.error().data());
    }
    return 2;
  }
  const auto twice_backend = twice->backend();
  if (!twice_backend || *twice_backend != rund::compute::Backend::Cpu) {
    return 2;
  }

  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto host = twice.value().run(std::span<const std::int32_t>{input});
  if (!host || host.value() != std::vector<std::int32_t>{7, 9, 11, 13}) {
    if (!host) {
      std::fprintf(stderr, "compute map host failed: %.*s\n",
                   static_cast<int>(host.error().size()), host.error().data());
    } else {
      std::fprintf(stderr, "compute map host output: %d %d %d %d\n",
                   host.value()[0], host.value()[1], host.value()[2],
                   host.value()[3]);
    }
    return 3;
  }

  auto source = device.value().upload(std::span<const std::int32_t>{input});
  auto target = device.value().buffer<std::int32_t>(input.size());
  if (!source || !target) {
    return 4;
  }
  auto run = twice.value().run(source.value(), target.value());
  if (!run) {
    return 5;
  }

  std::array<std::int32_t, 4> output{};
  const auto read =
      run.value().read(target.value(), std::span<std::int32_t>{output});
  if (!read || output != std::array<std::int32_t, 4>{7, 9, 11, 13}) {
    return 6;
  }

  const auto short_input =
      std::span<const std::int32_t>{input.data(), input.size() - 1};
  auto mismatch = twice.value().run(short_input);
  if (mismatch || mismatch.code() != rund::compute::Code::Binding ||
      mismatch.error() != "compute_shape_mismatch") {
    return 7;
  }
  auto moved_program = *twice;
  auto live_program = std::move(moved_program);
  const auto moved_backend = moved_program.backend();
  const auto moved_run = moved_program.run(short_input);
  if (!live_program || moved_program.valid() || moved_backend ||
      moved_backend.reason() != rund::compute::Reason::ProgramInvalid ||
      moved_run ||
      moved_run.reason() != rund::compute::Reason::ProgramInvalid) {
    return 15;
  }
  rund::compute::graph::Fingerprint envelope32{};
  rund::compute::graph::Fingerprint envelope64{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!Carrier(backend, envelope32, envelope64)) {
      return 100 + static_cast<int>(backend);
    }
  }
  return 0;
}

} // namespace compute_map_contract
