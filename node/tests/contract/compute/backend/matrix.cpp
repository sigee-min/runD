#include "model.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace rund_node_backend_contract {

template <class T> [[nodiscard]] constexpr T MatrixZero() noexcept {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::zero();
  } else {
    return T{0};
  }
}

template <class T>
[[nodiscard]] constexpr T MatrixInputValue(const std::uint32_t numerator) {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    using Raw = typename T::Raw;
    return T::from_raw(static_cast<Raw>(static_cast<Raw>(numerator)
                                        << (T::fraction_bits - 3u)));
  } else {
    return static_cast<T>(numerator);
  }
}

template <class T>
[[nodiscard]] constexpr T MatrixProductValue(const std::uint32_t numerator) {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    using Raw = typename T::Raw;
    return T::from_raw(static_cast<Raw>(static_cast<Raw>(numerator)
                                        << (T::fraction_bits - 6u)));
  } else {
    return static_cast<T>(numerator);
  }
}

template <class T> [[nodiscard]] bool CheckBatchMatrix() {
  using namespace rund::compute;
  const T zero = MatrixZero<T>();
  const std::array<T, 8u> left{
      MatrixInputValue<T>(1u), MatrixInputValue<T>(2u), zero,
      MatrixInputValue<T>(1u), MatrixInputValue<T>(3u), zero,
      MatrixInputValue<T>(1u), MatrixInputValue<T>(2u)};
  const std::array<T, 8u> right{MatrixInputValue<T>(2u),
                                zero,
                                MatrixInputValue<T>(1u),
                                MatrixInputValue<T>(1u),
                                MatrixInputValue<T>(1u),
                                MatrixInputValue<T>(1u),
                                zero,
                                MatrixInputValue<T>(2u)};
  const std::vector<T> expected{
      MatrixProductValue<T>(4u), MatrixProductValue<T>(2u),
      MatrixProductValue<T>(1u), MatrixProductValue<T>(1u),
      MatrixProductValue<T>(3u), MatrixProductValue<T>(3u),
      MatrixProductValue<T>(1u), MatrixProductValue<T>(5u)};
  PrimitiveEvidence reference{};
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    const auto target = on(rund::node::test_contract::target_for(backend, 2u));
    auto program =
        target
            .template map<T>("backend-batch-matrix", left.size(),
                             [](auto value) { return Store<T>(value); })
            .matrix({2u, 2u, 2u})
            .matmul({2u, 2u, 2u})
            .compile();
    if (!program) {
      return false;
    }
    auto job = program->resident(left, right);
    if (!job || !job->run()) {
      return false;
    }
    auto output = job->read();
    const Stats stats = job->stats();
    if (!output || *output != expected || stats.graph_hash == 0u ||
        stats.output_hash == 0u) {
      return false;
    }
    std::vector<std::uint64_t> bits;
    bits.reserve(output->size());
    for (const T value : *output) {
      bits.push_back(DomainBits(value));
    }
    if (backend == Backend::Cpu) {
      reference = {.values = std::move(bits),
                   .graph = stats.graph_hash,
                   .output = stats.output_hash};
    } else if (bits != reference.values ||
               stats.graph_hash != reference.graph ||
               stats.output_hash != reference.output) {
      return false;
    }
  }
  return true;
}

template <class T> [[nodiscard]] bool CheckArgsortDomainMatrix() {
  PrimitiveEvidence reference{};
  const std::array<T, 4> input{DomainValue<T>(3), DomainValue<T>(1),
                               DomainValue<T>(4), DomainValue<T>(2)};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    const auto target =
        rund::compute::on(rund::node::test_contract::target_for(backend, 2u));
    auto program =
        target
            .template map<T>("backend-argsort", input.size(),
                             [](auto value) { return Store<T>(value); })
            .argsort()
            .compile();
    if (!program) {
      return false;
    }
    auto job = program->resident(std::span<const T>{input});
    if (!job || !job->run()) {
      return false;
    }
    const auto result = job->read();
    if (!result) {
      return false;
    }
    std::vector<std::uint64_t> bits(result->begin(), result->end());
    const auto stats = job->stats();
    if (backend == rund::compute::Backend::Cpu) {
      reference = PrimitiveEvidence{.values = std::move(bits),
                                    .graph = stats.graph_hash,
                                    .output = stats.output_hash};
    } else if (bits != reference.values ||
               stats.graph_hash != reference.graph ||
               stats.output_hash != reference.output) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckMatrices() {
  return CheckBatchMatrix<std::int32_t>() &&
         CheckBatchMatrix<std::uint32_t>() &&
         CheckBatchMatrix<std::int64_t>() &&
         CheckBatchMatrix<std::uint64_t>() &&
         CheckBatchMatrix<rund::compute::Fixed<1, 31>>() &&
         CheckBatchMatrix<rund::compute::Fixed<1, 63>>();
}

[[nodiscard]] bool CheckArgsorts() {
  return CheckArgsortDomainMatrix<std::int32_t>() &&
         CheckArgsortDomainMatrix<std::uint32_t>() &&
         CheckArgsortDomainMatrix<std::int64_t>() &&
         CheckArgsortDomainMatrix<std::uint64_t>() &&
         CheckArgsortDomainMatrix<rund::compute::Fixed<1, 31>>() &&
         CheckArgsortDomainMatrix<rund::compute::Fixed<1, 63>>();
}

} // namespace rund_node_backend_contract
