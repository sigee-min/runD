#include "model.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace rund_node_backend_contract {

template <class T, class Build>
[[nodiscard]] bool CheckPrimitive(const rund::compute::Backend backend,
                                  const char *const name, Build &&build,
                                  PrimitiveEvidence &reference) {
  const std::array<T, 4> input{DomainValue<T>(3), DomainValue<T>(1),
                               DomainValue<T>(4), DomainValue<T>(2)};
  const auto target =
      rund::compute::on(rund::node::test_contract::target_for(backend, 2u));
  auto plan = target.template map<T>(
      name, input.size(), [](auto value) { return Store<T>(value); });
  auto staged = build(std::move(plan));
  auto program = std::move(staged).compile();
  if (!program) {
    std::fprintf(
        stderr, "compute primitive=%s backend=%u bytes=%zu compile: %.*s\n",
        name, static_cast<unsigned>(backend), sizeof(T),
        static_cast<int>(program.error().size()), program.error().data());
    return false;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    std::fprintf(stderr,
                 "compute primitive=%s backend=%u bytes=%zu failed: %.*s\n",
                 name, static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(job.error().size()), job.error().data());
    return false;
  }
  auto output = job->read();
  if (!output) {
    std::fprintf(stderr,
                 "compute primitive=%s backend=%u bytes=%zu read failed\n",
                 name, static_cast<unsigned>(backend), sizeof(T));
    return false;
  }
  std::vector<std::uint64_t> bits;
  bits.reserve(output->size());
  for (const T value : *output) {
    bits.push_back(DomainBits(value));
  }
  const rund::compute::Stats stats = job->stats();
  if (backend == rund::compute::Backend::Cpu) {
    reference = PrimitiveEvidence{.values = std::move(bits),
                                  .graph = stats.graph_hash,
                                  .output = stats.output_hash};
    return reference.graph != 0u && reference.output != 0u;
  }
  const bool same = bits == reference.values &&
                    stats.graph_hash == reference.graph &&
                    stats.output_hash == reference.output;
  if (!same) {
    std::fprintf(stderr,
                 "compute primitive parity mismatch: %s backend=%u values=%d "
                 "graph=%llu/%llu output=%llu/%llu\n",
                 name, static_cast<unsigned>(backend), bits == reference.values,
                 static_cast<unsigned long long>(stats.graph_hash),
                 static_cast<unsigned long long>(reference.graph),
                 static_cast<unsigned long long>(stats.output_hash),
                 static_cast<unsigned long long>(reference.output));
  }
  return same;
}

template <class T, class Build>
[[nodiscard]] bool CheckPrimitiveMatrix(const char *const name, Build &&build) {
  PrimitiveEvidence reference{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!CheckPrimitive<T>(backend, name, build, reference)) {
      std::fprintf(stderr,
                   "compute primitive matrix failed: %s backend=%u bytes=%zu\n",
                   name, static_cast<unsigned>(backend), sizeof(T));
      return false;
    }
  }
  return true;
}

template <class T> [[nodiscard]] bool CheckPrimitiveDomainMatrix() {
  return CheckPrimitiveMatrix<T>("backend-scan",
                                 [](auto flow) {
                                   return std::move(flow).scan(
                                       rund::compute::Scan::InclusiveSum);
                                 }) &&
         CheckPrimitiveMatrix<T>("backend-reduce",
                                 [](auto flow) {
                                   return std::move(flow).reduce(
                                       rund::compute::Reduce::Sum);
                                 }) &&
         CheckPrimitiveMatrix<T>(
             "backend-sort",
             [](auto flow) { return std::move(flow).sort(); }) &&
         CheckPrimitiveMatrix<T>(
             "backend-stencil",
             [](auto flow) {
               return std::move(flow).window(
                   {.op = rund::compute::Window::Max, .radius = 1u});
             }) &&
         CheckPrimitiveMatrix<T>("backend-matrix", [](auto flow) {
           return std::move(flow).template matrix<2u, 2u>().transpose();
         });
}

[[nodiscard]] bool CheckPrimitiveDomains() {
  return CheckPrimitiveDomainMatrix<std::int32_t>() &&
         CheckPrimitiveDomainMatrix<std::uint32_t>() &&
         CheckPrimitiveDomainMatrix<std::int64_t>() &&
         CheckPrimitiveDomainMatrix<std::uint64_t>() &&
         CheckPrimitiveDomainMatrix<rund::compute::Fixed<1, 31>>() &&
         CheckPrimitiveDomainMatrix<rund::compute::Fixed<1, 63>>();
}

[[nodiscard]] bool CheckHistogram() {
  return CheckPrimitiveMatrix<std::uint32_t>(
      "backend-histogram",
      [](auto flow) { return std::move(flow).histogram({.bins = 5u}); });
}

} // namespace rund_node_backend_contract
