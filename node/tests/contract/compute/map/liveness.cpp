#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/abi/graph.hpp>

#include "src/compute/expression/state.hpp"
#include "src/compute/flow/state.hpp"
#include "src/compute/graph/state.hpp"
#include "src/compute/map/build.hpp"
#include "src/compute/program/state.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace compute_map_contract {
[[nodiscard]] bool
Liveness(const std::shared_ptr<rund::compute::detail::DeviceState> &device) {
  using namespace rund::compute;
  using namespace rund::compute::detail;

  const std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};
  const std::array<std::uint32_t, 1u> input_ids{1u};
  auto shared = make_expr();
  const ExprRef input = detail::input(shared, Type::I32, 0u);
  const ExprRef one = constant(shared, Type::I32, 1u);
  const ExprRef two = constant(shared, Type::I32, 2u);
  const ExprRef live = binary(ExprOp::Add, input, one);
  const ExprRef dead = binary(ExprOp::Multiply, input, two);
  const std::array shared_roots{live, dead};

  auto dead_root_flow = make_flow_on(device, Type::I32, input_values.size());
  const ValueIds dead_root_outputs = flow_map_multi(
      dead_root_flow, input_ids, "shared-dead-root", shared_roots);
  if (dead_root_outputs.size() != 2u) {
    return false;
  }
  const std::array selected{dead_root_outputs.front()};
  flow_outputs(dead_root_flow, selected);
  auto dead_root_program = compile_flow(dead_root_flow);
  if (!dead_root_program ||
      (*dead_root_program)->graph_info.nodes.size() != 1u ||
      (*dead_root_program)->graph_info.nodes.front().accesses.size() != 2u ||
      (*dead_root_program)->output_types != std::vector<Type>{Type::I32}) {
    return false;
  }
  auto dead_root_job =
      make_job(*dead_root_program, std::span<const std::int32_t>{input_values});
  if (!dead_root_job || !run_job(*dead_root_job)) {
    return false;
  }
  auto dead_root_values = read_job<std::int32_t>(*dead_root_job);
  if (!dead_root_values ||
      *dead_root_values != std::vector<std::int32_t>{2, 3, 4, 5}) {
    return false;
  }

  auto shared_flow = make_flow_on(device, Type::I32, input_values.size());
  const ValueIds shared_outputs =
      flow_map_multi(shared_flow, input_ids, "shared-live-roots", shared_roots);
  if (shared_outputs.size() != 2u) {
    return false;
  }
  flow_outputs(shared_flow, shared_outputs);
  auto shared_program = compile_flow(shared_flow);
  if (!shared_program ||
      (*shared_program)->output_types !=
          std::vector<Type>{Type::I32, Type::I32} ||
      (*shared_program)->graph_info.nodes.size() != 1u ||
      (*shared_program)->graph_info.nodes.front().accesses.size() != 3u) {
    return false;
  }
  auto shared_job =
      make_job(*shared_program, std::span<const std::int32_t>{input_values});
  if (!shared_job || !run_job(*shared_job)) {
    return false;
  }
  auto shared_first = read_job<std::int32_t>(*shared_job, 0u);
  auto shared_second = read_job<std::int32_t>(*shared_job, 1u);
  if (!shared_first || !shared_second ||
      *shared_first != std::vector<std::int32_t>{2, 3, 4, 5} ||
      *shared_second != std::vector<std::int32_t>{2, 4, 6, 8}) {
    return false;
  }

  auto first_group = make_expr();
  const ExprRef first_input = detail::input(first_group, Type::I32, 0u);
  const ExprRef first_root = binary(ExprOp::Subtract, first_input,
                                    constant(first_group, Type::I32, 1u));
  auto second_group = make_expr();
  const ExprRef second_input = detail::input(second_group, Type::I32, 0u);
  const ExprRef second_root =
      binary(ExprOp::Add, second_input, constant(second_group, Type::I32, 7u));
  const std::array distinct_roots{second_root, first_root};
  auto grouped_flow = make_flow_on(device, Type::I32, input_values.size());
  const ValueIds grouped_outputs = flow_map_multi(
      grouped_flow, input_ids, "distinct-expression-groups", distinct_roots);
  if (grouped_outputs.size() != 2u) {
    return false;
  }
  flow_outputs(grouped_flow, grouped_outputs);
  auto grouped_program = compile_flow(grouped_flow);
  auto grouped_job =
      grouped_program
          ? make_job(*grouped_program,
                     std::span<const std::int32_t>{input_values})
          : Result<std::shared_ptr<JobState>>::fail(grouped_program.reason());
  if (!grouped_job || !run_job(*grouped_job)) {
    return false;
  }
  auto grouped_first = read_job<std::int32_t>(*grouped_job, 0u);
  auto grouped_second = read_job<std::int32_t>(*grouped_job, 1u);
  if (!grouped_first || !grouped_second ||
      *grouped_first != std::vector<std::int32_t>{8, 9, 10, 11} ||
      *grouped_second != std::vector<std::int32_t>{0, 1, 2, 3}) {
    return false;
  }

  auto invalid_input_state = std::make_shared<ExprState>();
  invalid_input_state->nodes.push_back(
      ExprNode{.operation = ExprOp::Input, .type = Type::I32, .left = 1u});
  auto exhausted_state = std::make_shared<ExprState>();
  exhausted_state->nodes.push_back(
      ExprNode{.operation = ExprOp::Input, .type = Type::I32, .left = 0u});
  exhausted_state->status = Status::fail(Reason::ExpressionCapacity);
  const std::array invalid_input_first{
      ExprRef{invalid_input_state, 1u, Type::I32, {}},
      ExprRef{exhausted_state, 1u, Type::I32, {}}};
  auto invalid_input_flow = make_flow_on(device, Type::I32, 1u);
  if (!flow_map_multi(invalid_input_flow, input_ids, "invalid-input-precedence",
                      invalid_input_first)
           .empty() ||
      invalid_input_flow->status.reason() != Reason::GraphTypeMismatch) {
    return false;
  }

  auto valid_state = make_expr();
  const ExprRef valid_input = detail::input(valid_state, Type::I32, 0u);
  const std::array invalid_type_first{
      ExprRef{valid_state, valid_input.node, Type::I64, {}},
      ExprRef{exhausted_state, 1u, Type::I32, {}}};
  auto invalid_type_flow = make_flow_on(device, Type::I32, 1u);
  if (!flow_map_multi(invalid_type_flow, input_ids, "type-precedence",
                      invalid_type_first)
           .empty() ||
      invalid_type_flow->status.reason() != Reason::GraphTypeMismatch) {
    return false;
  }

  auto boundary_state = std::make_shared<ExprState>();
  boundary_state->nodes.reserve(1024u);
  boundary_state->nodes.push_back(
      ExprNode{.operation = ExprOp::Input, .type = Type::I32, .left = 0u});
  for (std::uint32_t index = 1u; index < 1023u; ++index) {
    boundary_state->nodes.push_back(ExprNode{
        .operation = ExprOp::Constant, .type = Type::I32, .bits = index});
  }
  boundary_state->nodes.push_back(ExprNode{
      .operation = ExprOp::Add, .type = Type::I32, .left = 1u, .right = 1u});
  auto boundary_flow = make_flow_on(device, Type::I32, input_values.size());
  flow_map(boundary_flow, "expression-boundary",
           ExprRef{boundary_state, 1024u, Type::I32, {}});
  auto boundary_program = compile_flow(boundary_flow);
  auto boundary_job =
      boundary_program
          ? make_job(*boundary_program,
                     std::span<const std::int32_t>{input_values})
          : Result<std::shared_ptr<JobState>>::fail(boundary_program.reason());
  if (!boundary_job || !run_job(*boundary_job)) {
    return false;
  }
  auto boundary_values = read_job<std::int32_t>(*boundary_job);
  if (!boundary_values ||
      *boundary_values != std::vector<std::int32_t>{2, 4, 6, 8}) {
    return false;
  }

  auto capacity_flow = make_flow_on(device, Type::I32, 1u);
  flow_map(capacity_flow, "expression-capacity",
           ExprRef{exhausted_state, 1u, Type::I32, {}});
  auto capacity_program = compile_flow(capacity_flow);
  if (capacity_program ||
      capacity_program.reason() != Reason::ExpressionCapacity) {
    return false;
  }

  using Fixed = rund::compute::Fixed<16u, 16u>;
  using Alternate = rund::compute::Fixed<17u, 15u>;
  const FixedFormat fixed = storage_format<Fixed>();
  const FixedFormat alternate = storage_format<Alternate>();
  const std::array fixed_inputs{Type::FixedLane32};
  const std::array fixed_outputs{Type::FixedLane32, Type::FixedLane32};
  auto explicit_fixed_state = std::make_shared<ExprState>();
  explicit_fixed_state->nodes.push_back(ExprNode{.operation = ExprOp::Input,
                                                 .type = Type::FixedLane32,
                                                 .fixed_format = fixed,
                                                 .left = 0u});
  const std::array explicit_fixed_roots{
      ExprRef{explicit_fixed_state, 1u, Type::FixedLane32, fixed},
      ExprRef{explicit_fixed_state, 1u, Type::FixedLane32, alternate}};
  if (!build_map_operation_multi(4u, fixed_outputs, fixed_inputs,
                                 explicit_fixed_roots)) {
    return false;
  }

  auto fallback_state = std::make_shared<ExprState>();
  fallback_state->nodes.push_back(ExprNode{.operation = ExprOp::Input,
                                           .type = Type::FixedLane32,
                                           .fixed_format = {},
                                           .left = 0u});
  const std::array inconsistent_fallback_roots{
      ExprRef{fallback_state, 1u, Type::FixedLane32, fixed},
      ExprRef{fallback_state, 1u, Type::FixedLane32, alternate}};
  const auto inconsistent_fallback = build_map_operation_multi(
      4u, fixed_outputs, fixed_inputs, inconsistent_fallback_roots);
  return !inconsistent_fallback &&
         inconsistent_fallback.reason() == Reason::DomainTypeMismatch;
}

} // namespace compute_map_contract
