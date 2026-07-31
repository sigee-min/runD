#include "model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include "tests/contract/target/selection.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace compute_shape_contract {
namespace {
using namespace rund::compute;

template <class T>
[[nodiscard]] bool MatrixInputSolveGraph(const graph::Info &info,
                                         const std::size_t matrix_elements,
                                         const std::size_t rhs_elements,
                                         const std::size_t batches) {
  if (!info.fingerprint || info.inputs.size() != 2u ||
      info.outputs.size() != 2u) {
    return false;
  }
  const auto resource = [&](const std::uint32_t id) -> const graph::Resource * {
    return id == 0u || id > info.resources.size() ? nullptr
                                                  : &info.resources[id - 1u];
  };
  const auto fixed = [](const graph::Resource &value) {
    return value.type == graph::Value::Fixed &&
           value.integer_bits == T::integer_bits &&
           value.fraction_bits == T::fraction_bits &&
           value.element_bytes == sizeof(T);
  };
  const graph::Resource *const matrix_input = resource(info.inputs[0u]);
  const graph::Resource *const rhs_input = resource(info.inputs[1u]);
  const graph::Resource *const values_output = resource(info.outputs[0u]);
  const graph::Resource *const status_output = resource(info.outputs[1u]);
  if (matrix_input == nullptr || rhs_input == nullptr ||
      values_output == nullptr || status_output == nullptr ||
      !fixed(*matrix_input) || !fixed(*rhs_input) || !fixed(*values_output) ||
      matrix_input->visibility != graph::Visibility::Input ||
      rhs_input->visibility != graph::Visibility::Input ||
      values_output->visibility != graph::Visibility::Output ||
      status_output->visibility != graph::Visibility::Output ||
      matrix_input->elements != matrix_elements ||
      rhs_input->elements != rhs_elements ||
      values_output->elements != rhs_elements ||
      status_output->type != graph::Value::U32 ||
      status_output->element_bytes != sizeof(std::uint32_t) ||
      status_output->elements != batches) {
    return false;
  }

  const graph::Node *solve = nullptr;
  std::size_t factors = 0u;
  for (const graph::Node &node : info.nodes) {
    factors += node.operation == graph::Operation::Factor ? 1u : 0u;
    if (node.operation == graph::Operation::Solve) {
      if (solve != nullptr) {
        return false;
      }
      solve = &node;
    }
  }
  if (solve == nullptr || factors != 0u) {
    return false;
  }

  std::size_t reads = 0u;
  std::size_t writes = 0u;
  bool reads_matrix = false;
  bool reads_rhs = false;
  bool writes_values = false;
  bool writes_status = false;
  for (const graph::Access &access : solve->accesses) {
    const graph::Resource *const value = resource(access.resource);
    if (value == nullptr || access.offset_bytes != 0u ||
        access.element_bytes != value->element_bytes ||
        access.element_count != value->elements ||
        access.stride_bytes != value->element_bytes) {
      return false;
    }
    if (access.mode == resource::AccessMode::Read) {
      ++reads;
      if (access.resource == info.inputs[1u] && fixed(*value) &&
          value->elements == rhs_elements) {
        reads_rhs = true;
      } else if (access.resource != info.inputs[1u] && fixed(*value) &&
                 value->elements == matrix_elements) {
        reads_matrix = true;
      }
    } else {
      ++writes;
      writes_values |= access.resource == info.outputs[0u];
      writes_status |= access.resource == info.outputs[1u];
    }
  }
  return reads == 2u && writes == 2u && reads_matrix && reads_rhs &&
         writes_values && writes_status;
}

template <class T> struct MatrixSolveEvidence final {
  std::vector<T> values;
  std::vector<std::uint32_t> status;
  std::uint64_t graph{};
  std::uint64_t output{};
};

template <class T> [[nodiscard]] T Fraction(const unsigned denominator_power) {
  using Raw = typename T::Raw;
  return T::from_raw(
      static_cast<Raw>(Raw{1} << (T::fraction_bits - denominator_power)));
}

template <class T>
[[nodiscard]] bool RunStaticMatrixInputSolve(
    const Backend backend, const FactorOp operation,
    const std::array<T, 4u> &matrix, const std::array<T, 4u> &rhs,
    const std::vector<T> &expected, MatrixSolveEvidence<T> &reference) {
  auto input =
      on(rund::node::test_contract::target_for(backend))
          .template map<T>("public-matrix-input-solve", matrix.size(),
                           [](auto value) { return quantize<T>(value); });
  auto program = [&] {
    if (operation == FactorOp::Lu) {
      return std::move(input)
          .template matrix<2u, 2u>()
          .template solve<FactorOp::Lu, 2u>()
          .compile();
    }
    if (operation == FactorOp::Qr) {
      return std::move(input)
          .template matrix<2u, 2u>()
          .template solve<FactorOp::Qr, 2u>()
          .compile();
    }
    return std::move(input)
        .template matrix<2u, 2u>()
        .template solve<FactorOp::Cholesky, 2u>()
        .compile();
  }();
  if (!program || !MatrixInputSolveGraph<T>(program->graph(), matrix.size(),
                                            rhs.size(), 1u)) {
    std::fprintf(stderr,
                 "matrix-input static graph backend=%u bytes=%zu factor=%u "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<unsigned>(operation),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  auto job = program->resident(matrix, rhs);
  if (!job) {
    std::fprintf(stderr,
                 "matrix-input static run backend=%u bytes=%zu factor=%u "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<unsigned>(operation),
                 static_cast<int>(job.error().size()), job.error().data());
    return false;
  }
  const Status run = job->run();
  if (!run) {
    std::fprintf(stderr,
                 "matrix-input static run backend=%u bytes=%zu factor=%u "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<unsigned>(operation),
                 static_cast<int>(run.error().size()), run.error().data());
    return false;
  }
  auto values = job->template read<0u>();
  auto status = job->template read<1u>();
  if (!values || !status || *values != expected ||
      *status != std::vector<std::uint32_t>{0u}) {
    std::fprintf(stderr,
                 "matrix-input static value backend=%u bytes=%zu factor=%u\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<unsigned>(operation));
    return false;
  }
  const Stats stats = job->stats();
  if (stats.graph_hash == 0u || stats.output_hash == 0u) {
    return false;
  }
  if (backend == Backend::Cpu) {
    reference = MatrixSolveEvidence<T>{.values = std::move(*values),
                                       .status = std::move(*status),
                                       .graph = stats.graph_hash,
                                       .output = stats.output_hash};
    return true;
  }
  return *values == reference.values && *status == reference.status &&
         stats.graph_hash == reference.graph &&
         stats.output_hash == reference.output;
}

template <class T, std::size_t MatrixCount, std::size_t RhsCount>
[[nodiscard]] bool RunDynamicMatrixInputSolve(
    const Backend backend, const FactorOp operation,
    const std::array<T, MatrixCount> &matrix, const MatrixShape shape,
    const std::array<T, RhsCount> &rhs, const std::size_t rhs_cols,
    const std::vector<T> &expected,
    const std::vector<std::uint32_t> &expected_status) {
  auto program =
      on(rund::node::test_contract::target_for(backend))
          .template map<T>("public-dynamic-matrix-input-solve", matrix.size(),
                           [](auto value) { return quantize<T>(value); })
          .matrix(shape)
          .solve(operation, rhs_cols)
          .compile();
  if (!program || !MatrixInputSolveGraph<T>(program->graph(), matrix.size(),
                                            rhs.size(), shape.batches)) {
    const auto reason =
        program ? std::string_view{"graph_mismatch"} : program.error();
    std::fprintf(stderr,
                 "matrix-input dynamic graph backend=%u bytes=%zu factor=%u "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<unsigned>(operation),
                 static_cast<int>(reason.size()), reason.data());
    return false;
  }
  auto job = program->resident(matrix, rhs);
  const Status run = job ? job->run() : Status::fail(job.reason());
  auto result =
      run ? job->read_all() : decltype(job->read_all())::fail(run.reason());
  const Stats stats = job ? job->stats() : Stats{};
  if (!result || std::get<0>(*result) != expected ||
      std::get<1>(*result) != expected_status || stats.graph_hash == 0u ||
      stats.output_hash == 0u) {
    const auto reason =
        result ? std::string_view{"value_mismatch"} : result.error();
    std::fprintf(stderr,
                 "matrix-input dynamic backend=%u bytes=%zu factor=%u "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<unsigned>(operation),
                 static_cast<int>(reason.size()), reason.data());
    if (result) {
      std::fprintf(stderr, "  values");
      for (const T value : std::get<0>(*result)) {
        std::fprintf(stderr, " %lld", static_cast<long long>(value.raw()));
      }
      std::fprintf(stderr, " expected");
      for (const T value : expected) {
        std::fprintf(stderr, " %lld", static_cast<long long>(value.raw()));
      }
      std::fprintf(stderr, " status");
      for (const std::uint32_t value : std::get<1>(*result)) {
        std::fprintf(stderr, " %u", value);
      }
      std::fprintf(stderr, "\n");
    }
    return false;
  }
  return true;
}

template <class T> [[nodiscard]] bool CheckPublicMatrixInputBackendMatrix() {
  const T zero = T::zero();
  const T half = Fraction<T>(1u);
  const T quarter = Fraction<T>(2u);
  const T eighth = Fraction<T>(3u);
  const std::array<T, 4u> matrix{quarter, zero, zero, quarter};
  const std::array<T, 4u> rhs{eighth, zero, zero, eighth};
  const std::vector<T> expected{half, zero, zero, half};
  const std::array<T, 8u> batched_matrix{quarter, zero, zero, quarter,
                                         quarter, zero, zero, quarter};
  const std::array<T, 4u> batched_rhs{eighth, zero, eighth, zero};
  const std::vector<T> batched_expected{half, zero, half, zero};
  for (const FactorOp operation :
       {FactorOp::Lu, FactorOp::Qr, FactorOp::Cholesky}) {
    MatrixSolveEvidence<T> reference{};
    for (const Backend backend :
         rund::node::test_contract::selected_compute_backends()) {
      if (!RunStaticMatrixInputSolve(backend, operation, matrix, rhs, expected,
                                     reference) ||
          !RunDynamicMatrixInputSolve(backend, operation, batched_matrix,
                                      MatrixShape{2u, 2u, 2u}, batched_rhs, 1u,
                                      batched_expected, {0u, 0u})) {
        return false;
      }
    }
  }

  using Raw = typename T::Raw;
  constexpr std::int64_t large_integer =
      sizeof(Raw) == sizeof(std::int32_t) ? 20000 : 400000;
  const Raw scale = static_cast<Raw>(Raw{1} << T::fraction_bits);
  const T large = T::from_raw(static_cast<Raw>(large_integer * scale));
  const std::array<T, 4u> overflow_matrix{half, zero, zero, half};
  const std::array<T, 2u> overflow_rhs{large, zero};
  const std::vector<T> overflow_expected{T::max(), zero};
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!RunDynamicMatrixInputSolve(backend, FactorOp::Lu, overflow_matrix,
                                    MatrixShape{2u, 2u, 1u}, overflow_rhs, 1u,
                                    overflow_expected, {0u})) {
      return false;
    }
    auto invalid_shape =
        on(rund::node::test_contract::target_for(backend), matrix)
            .matrix({1u, 4u, 1u})
            .solve(rhs, FactorOp::Lu, 1u)
            .collect();
    if (invalid_shape ||
        invalid_shape.error() != "compute_solve_shape_mismatch") {
      return false;
    }
  }
  return true;
}

} // namespace

bool Backend32() {
  return CheckPublicMatrixInputBackendMatrix<rund::compute::Fixed<16, 16>>();
}

bool Backend64() {
  return CheckPublicMatrixInputBackendMatrix<rund::compute::Fixed<20, 44>>();
}

} // namespace compute_shape_contract
