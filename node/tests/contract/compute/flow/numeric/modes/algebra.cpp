#include "../local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include "../../../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <tuple>
#include <vector>

namespace rund::node::test_contract::numeric {
namespace {

template <class T> [[nodiscard]] T Half() noexcept {
  using Raw = typename T::Raw;
  return T::from_raw(static_cast<Raw>(Raw{1} << (T::fraction_bits - 1u)));
}

template <class T> [[nodiscard]] T Quarter() noexcept {
  using Raw = typename T::Raw;
  return T::from_raw(static_cast<Raw>(Raw{1} << (T::fraction_bits - 2u)));
}

template <class T> [[nodiscard]] T Eighth() noexcept {
  using Raw = typename T::Raw;
  return T::from_raw(static_cast<Raw>(Raw{1} << (T::fraction_bits - 3u)));
}

template <class Run> [[nodiscard]] bool SameCollect(Run &&run) {
  using rund::compute::Backend;
  auto reference = run(Backend::Cpu);
  if (!reference) {
    std::fprintf(
        stderr, "numeric modes backend=%u unavailable code=%u reason=%.*s\n",
        static_cast<unsigned>(Backend::Cpu),
        static_cast<unsigned>(reference.code()),
        static_cast<int>(reference.error().size()), reference.error().data());
    return false;
  }
  for (const Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    auto output = run(backend);
    if (!output) {
      std::fprintf(
          stderr, "numeric modes backend=%u unavailable code=%u reason=%.*s\n",
          static_cast<unsigned>(backend), static_cast<unsigned>(output.code()),
          static_cast<int>(output.error().size()), output.error().data());
      return false;
    }
    if (*output != *reference) {
      std::fprintf(stderr, "numeric modes backend=%u output-mismatch\n",
                   static_cast<unsigned>(backend));
      return false;
    }
  }
  return true;
}

template <class T>
[[nodiscard]] bool CheckDirectSolveMatrix(const rund::compute::FactorOp factor,
                                          const std::array<T, 4u> &matrix) {
  using namespace rund::compute;
  struct Evidence final {
    std::vector<T> output;
    Stats stats{};
  };
  const auto run = [&](const Backend selected, Evidence &evidence) {
    const auto target = on(rund::node::test_contract::target_for(selected, 2u));
    auto program = [&] {
      auto input =
          target.template map<T>("numeric-direct-solve", matrix.size(),
                                 [](auto value) { return quantize<T>(value); });
      if (factor == FactorOp::Lu) {
        return std::move(input)
            .template matrix<2u, 2u>()
            .lu()
            .template solve<2u>()
            .compile();
      }
      return std::move(input)
          .template matrix<2u, 2u>()
          .cholesky()
          .template solve<2u>()
          .compile();
    }();
    if (!program) {
      std::fprintf(
          stderr,
          "numeric solve factor=%u backend=%u compile code=%u reason=%.*s\n",
          static_cast<unsigned>(factor), static_cast<unsigned>(selected),
          static_cast<unsigned>(program.code()),
          static_cast<int>(program.error().size()), program.error().data());
      return false;
    }
    auto job = program->resident(matrix, matrix);
    if (!job) {
      std::fprintf(
          stderr,
          "numeric solve factor=%u backend=%u resident code=%u reason=%.*s\n",
          static_cast<unsigned>(factor), static_cast<unsigned>(selected),
          static_cast<unsigned>(job.code()),
          static_cast<int>(job.error().size()), job.error().data());
      return false;
    }
    auto executed = job->run();
    if (!executed) {
      std::fprintf(
          stderr,
          "numeric solve factor=%u backend=%u run code=%u reason=%.*s\n",
          static_cast<unsigned>(factor), static_cast<unsigned>(selected),
          static_cast<unsigned>(executed.code()),
          static_cast<int>(executed.error().size()), executed.error().data());
      return false;
    }
    auto outputs = job->read_all();
    if (!outputs) {
      std::fprintf(
          stderr,
          "numeric solve factor=%u backend=%u read code=%u reason=%.*s\n",
          static_cast<unsigned>(factor), static_cast<unsigned>(selected),
          static_cast<unsigned>(outputs.code()),
          static_cast<int>(outputs.error().size()), outputs.error().data());
      return false;
    }
    evidence = {.output = std::move(std::get<0u>(*outputs)),
                .stats = job->stats()};
    if (evidence.stats.graph_hash == 0u || evidence.stats.output_hash == 0u) {
      std::fprintf(stderr,
                   "numeric solve factor=%u backend=%u invalid-stats "
                   "graph=%llu output=%llu\n",
                   static_cast<unsigned>(factor),
                   static_cast<unsigned>(selected),
                   static_cast<unsigned long long>(evidence.stats.graph_hash),
                   static_cast<unsigned long long>(evidence.stats.output_hash));
      return false;
    }
    return true;
  };

  Evidence reference{};
  if (!run(Backend::Cpu, reference)) {
    return false;
  }
  for (const Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    Evidence actual{};
    if (!run(backend, actual) || actual.output != reference.output ||
        actual.stats.graph_hash != reference.stats.graph_hash ||
        actual.stats.output_hash != reference.stats.output_hash) {
      std::fprintf(
          stderr, "numeric solve factor=%u backend=%u parity-mismatch\n",
          static_cast<unsigned>(factor), static_cast<unsigned>(backend));
      return false;
    }
  }
  return true;
}

template <class T> [[nodiscard]] int CheckModes() {
  using namespace rund::compute;
  const T half = Half<T>();
  const T quarter = Quarter<T>();
  const T eighth = Eighth<T>();
  const std::array<T, 4u> matrix{half, eighth, eighth, quarter};

  if (!SameCollect([&](const Backend backend) {
        return on(rund::node::test_contract::target_for(backend), matrix)
            .template matrix<2u, 2u>()
            .qr()
            .collect();
      })) {
    return 1;
  }
  if (!SameCollect([&](const Backend backend) {
        return on(rund::node::test_contract::target_for(backend), matrix)
            .template matrix<2u, 2u>()
            .cholesky()
            .collect();
      })) {
    return 2;
  }
  if (!SameCollect([&](const Backend backend) {
        return on(rund::node::test_contract::target_for(backend), matrix)
            .template matrix<2u, 2u>()
            .qr()
            .template solve<2u>(matrix)
            .collect();
      })) {
    return 3;
  }
  if (!SameCollect([&](const Backend backend) {
        return on(rund::node::test_contract::target_for(backend), matrix)
            .template matrix<2u, 2u>()
            .cholesky()
            .template solve<2u>(matrix)
            .collect();
      })) {
    return 4;
  }
  if (!SameCollect([&](const Backend backend) {
        return on(rund::node::test_contract::target_for(backend), matrix)
            .template matrix<2u, 2u>()
            .template svd<SpectrumVectors::Values>(128u)
            .collect();
      })) {
    return 5;
  }
  if (!SameCollect([&](const Backend backend) {
        return on(rund::node::test_contract::target_for(backend), matrix)
            .template matrix<2u, 2u>()
            .template svd<SpectrumVectors::Full>(128u)
            .collect();
      })) {
    return 6;
  }
  if (!SameCollect([&](const Backend backend) {
        return on(rund::node::test_contract::target_for(backend), matrix)
            .template matrix<2u, 2u>()
            .template eigen<SpectrumVectors::Values>(128u)
            .collect();
      })) {
    return 7;
  }
  if (!SameCollect([&](const Backend backend) {
        return on(rund::node::test_contract::target_for(backend), matrix)
            .template matrix<2u, 2u>()
            .template eigen<SpectrumVectors::Full>(128u)
            .collect();
      })) {
    return 8;
  }
  if (!CheckDirectSolveMatrix(FactorOp::Lu, matrix)) {
    return 9;
  }
  if (!CheckDirectSolveMatrix(FactorOp::Cholesky, matrix)) {
    return 10;
  }

  const std::array<T, 6u> rectangular{half,   eighth, quarter,
                                      eighth, eighth, quarter};
  return SameCollect([&](const Backend backend) {
    return on(rund::node::test_contract::target_for(backend), rectangular)
        .template matrix<3u, 2u>()
        .template svd<SpectrumVectors::Full>(128u)
        .collect();
  })
             ? 0
             : 11;
}

template <class T> [[nodiscard]] int CheckAdditionalAlgebraModes() {
  using namespace rund::compute;
  const T zero = T::zero();
  const T half = Half<T>();
  const std::array<T, 4u> matrix{half, zero, zero, half};
  if (!SameCollect([&](const Backend backend) {
        return on(rund::node::test_contract::target_for(backend), matrix)
            .template matrix<2u, 2u>()
            .lu()
            .collect();
      })) {
    return 1;
  }
  if (!SameCollect([&](const Backend backend) {
        return on(rund::node::test_contract::target_for(backend), matrix)
            .template matrix<2u, 2u>()
            .template svd<SpectrumVectors::Thin>()
            .collect();
      })) {
    return 2;
  }
  return SameCollect([&](const Backend backend) {
    return on(rund::node::test_contract::target_for(backend), matrix)
        .template matrix<2u, 2u>()
        .template eigen<SpectrumVectors::Thin>(64u)
        .collect();
  })
             ? 0
             : 3;
}

} // namespace

int CheckModesLane32() { return CheckModes<rund::compute::Fixed<1, 31>>(); }

int CheckModesLane64() { return CheckModes<rund::compute::Fixed<1, 63>>(); }

int CheckModesFormat32() { return CheckModes<rund::compute::Fixed<16, 16>>(); }

int CheckModesFormat64() { return CheckModes<rund::compute::Fixed<20, 44>>(); }

int CheckAdditional32() {
  return CheckAdditionalAlgebraModes<rund::compute::Fixed<16, 16>>();
}

int CheckAdditional64() {
  return CheckAdditionalAlgebraModes<rund::compute::Fixed<20, 44>>();
}

} // namespace rund::node::test_contract::numeric
