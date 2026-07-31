#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

namespace compute_shape_contract {
int Run() {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> left{1, 2, 3, 4};
  const std::array<std::int32_t, 4u> right{5, 6, 7, 8};
  auto program = on(Target::cpu(2u))
                     .map<std::int32_t>("static-matrix", left.size(),
                                        [](auto value) { return value; })
                     .matrix<2u, 2u>()
                     .matmul<2u, 2u>()
                     .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(left, right);
  if (!job || !job->run()) {
    return 2;
  }
  auto output = job->read();
  if (!output || *output != std::vector<std::int32_t>{19, 22, 43, 50}) {
    return 3;
  }
  auto dynamic = on(Target::cpu(2u))
                     .map<std::int32_t>("static-matrix", left.size(),
                                        [](auto value) { return value; })
                     .matrix({2u, 2u, 1u})
                     .matmul({2u, 2u, 1u})
                     .compile();
  if (!dynamic) {
    return 4;
  }
  auto dynamic_job = dynamic->resident(left, right);
  if (!dynamic_job || !dynamic_job->run()) {
    return 5;
  }
  auto dynamic_output = dynamic_job->read();
  if (!dynamic_output || *dynamic_output != *output ||
      dynamic_job->stats().graph_hash != job->stats().graph_hash ||
      dynamic_job->stats().output_hash != job->stats().output_hash) {
    return 6;
  }
  auto invalid_matrix = on(Target::cpu(), left)
                            .matrix({1u, 4u, 1u})
                            .matmul(right, {2u, 2u, 1u})
                            .collect();
  if (invalid_matrix ||
      invalid_matrix.error() != "compute_matrix_shape_mismatch") {
    return 20;
  }

  const Fixed<1, 31> zero = Fixed<1, 31>::zero();
  const Fixed<1, 31> half = Fixed<1, 31>::from_raw(1 << 30);
  const std::array<Fixed<1, 31>, 4u> identity{half, zero, zero, half};
  auto invalid_factor =
      on(Target::cpu(), identity).matrix({1u, 4u, 1u}).lu().collect();
  if (invalid_factor ||
      invalid_factor.error() != "compute_factor_shape_square") {
    return 21;
  }
  auto static_solve =
      on(Target::cpu(2u))
          .map<Fixed<1, 31>>(
              "static-solve", identity.size(),
              [](auto value) { return quantize<Fixed<1, 31>>(value); })
          .matrix<2u, 2u>()
          .lu()
          .solve<2u>()
          .values()
          .compile();
  if (!static_solve) {
    return 7;
  }
  auto static_job = static_solve->resident(identity, identity);
  if (!static_job || !static_job->run()) {
    return 8;
  }
  auto static_output = static_job->read();
  if (!static_output) {
    return 9;
  }
  auto dynamic_solve =
      on(Target::cpu(2u))
          .map<Fixed<1, 31>>(
              "static-solve", identity.size(),
              [](auto value) { return quantize<Fixed<1, 31>>(value); })
          .matrix({2u, 2u, 1u})
          .lu()
          .solve(2u)
          .values()
          .compile();
  if (!dynamic_solve) {
    return 10;
  }
  auto solved_job = dynamic_solve->resident(identity, identity);
  if (!solved_job || !solved_job->run()) {
    return 11;
  }
  auto solved_output = solved_job->read();
  if (!solved_output || *solved_output != *static_output ||
      solved_job->stats().graph_hash != static_job->stats().graph_hash ||
      solved_job->stats().output_hash != static_job->stats().output_hash) {
    return 12;
  }

  const Fixed<1, 31> quarter = Fixed<1, 31>::from_raw(1 << 29);
  const Fixed<1, 31> eighth = Fixed<1, 31>::from_raw(1 << 28);
  const std::array<Fixed<1, 31>, 4u> direct_matrix{quarter, zero, zero,
                                                   quarter};
  const std::array<Fixed<1, 31>, 4u> direct_rhs{eighth, zero, zero, eighth};
  const std::vector<Fixed<1, 31>> direct_expected{half, zero, zero, half};
  for (const FactorOp operation :
       {FactorOp::Lu, FactorOp::Qr, FactorOp::Cholesky}) {
    auto direct = on(Target::cpu(2u), direct_matrix)
                      .matrix({2u, 2u, 1u})
                      .solve(direct_rhs, operation, 2u)
                      .collect();
    if (!direct || std::get<0>(*direct) != direct_expected ||
        std::get<1>(*direct) != std::vector<std::uint32_t>{0u}) {
      return 13;
    }
  }

  auto direct_static = on(Target::cpu(2u), direct_matrix)
                           .matrix<2u, 2u>()
                           .solve<FactorOp::Lu, 2u>(direct_rhs)
                           .collect();
  if (!direct_static || std::get<0>(*direct_static) != direct_expected ||
      std::get<1>(*direct_static) != std::vector<std::uint32_t>{0u}) {
    return 14;
  }

  auto deferred_direct =
      on(Target::cpu(2u))
          .map<Fixed<1, 31>>(
              "matrix-input-solve", direct_matrix.size(),
              [](auto value) { return quantize<Fixed<1, 31>>(value); })
          .matrix<2u, 2u>()
          .solve<FactorOp::Qr, 2u>()
          .compile();
  if (!deferred_direct) {
    return 15;
  }
  const auto &direct_graph = deferred_direct->graph();
  std::size_t solve_nodes = 0u;
  std::size_t factor_nodes = 0u;
  for (const graph::Node &node : direct_graph.nodes) {
    solve_nodes += node.operation == graph::Operation::Solve ? 1u : 0u;
    factor_nodes += node.operation == graph::Operation::Factor ? 1u : 0u;
  }
  if (solve_nodes != 1u || factor_nodes != 0u) {
    return 16;
  }
  auto deferred_job = deferred_direct->resident(direct_matrix, direct_rhs);
  if (!deferred_job || !deferred_job->run()) {
    return 17;
  }
  auto deferred_values = deferred_job->read<0u>();
  auto deferred_status = deferred_job->read<1u>();
  if (!deferred_values || !deferred_status ||
      *deferred_values != direct_expected ||
      *deferred_status != std::vector<std::uint32_t>{0u}) {
    return 18;
  }

  auto stage_direct =
      on(Target::cpu(2u))
          .map<Fixed<1, 31>>(
              "matrix-stage-solve", direct_matrix.size(),
              [](auto value) { return quantize<Fixed<1, 31>>(value); })
          .matrix<2u, 2u>()
          .pipe([](auto matrix) {
            return matrix.template solve<FactorOp::Cholesky, 2u>(matrix)
                .status();
          })
          .compile();
  if (!stage_direct) {
    return 19;
  }
  auto stage_job = stage_direct->resident(direct_matrix);
  if (!stage_job || !stage_job->run()) {
    return 22;
  }
  auto stage_status = stage_job->read();
  if (!stage_status || *stage_status != std::vector<std::uint32_t>{0u}) {
    return 23;
  }

  auto invalid_direct = on(Target::cpu(), direct_matrix)
                            .matrix({1u, 4u, 1u})
                            .solve(direct_rhs, FactorOp::Lu, 1u)
                            .collect();
  if (invalid_direct ||
      invalid_direct.error() != "compute_solve_shape_mismatch") {
    return 24;
  }
  const std::array<Fixed<1, 31>, 1u> short_rhs{eighth};
  auto invalid_rhs = on(Target::cpu(), direct_matrix)
                         .matrix({2u, 2u, 1u})
                         .solve(short_rhs, FactorOp::Qr, 1u)
                         .collect();
  if (invalid_rhs || invalid_rhs.error() != "compute_solve_shape_mismatch") {
    return 25;
  }
  const std::array<Fixed<1, 31>, 8u> batched_matrix{
      quarter, zero, zero, quarter, quarter, zero, zero, quarter};
  const std::array<Fixed<1, 31>, 4u> batched_rhs{eighth, zero, eighth, zero};
  auto batched_direct = on(Target::cpu(), batched_matrix)
                            .matrix({2u, 2u, 2u})
                            .solve(batched_rhs, FactorOp::Cholesky, 1u)
                            .collect();
  if (!batched_direct ||
      std::get<0>(*batched_direct) !=
          std::vector<Fixed<1, 31>>{half, zero, half, zero} ||
      std::get<1>(*batched_direct) != std::vector<std::uint32_t>{0u, 0u}) {
    return 26;
  }
  auto invalid_operation =
      on(Target::cpu(), direct_matrix)
          .matrix({2u, 2u, 1u})
          .solve(direct_rhs, static_cast<FactorOp>(255u), 2u)
          .collect();
  if (invalid_operation ||
      invalid_operation.error() != "compute_solve_factor_unsupported") {
    return 27;
  }
  if (!Backend32()) {
    return 28;
  }
  if (!Backend64()) {
    return 29;
  }
  return 0;
}

} // namespace compute_shape_contract
