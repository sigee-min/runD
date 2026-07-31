#include "model.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>

namespace rund_node_backend_contract {

template <class T>
[[nodiscard]] bool CheckBoundary(const rund::compute::Backend backend,
                                 DomainEvidence &reference) {
  const std::array<T, 5> input{DomainMin<T>(), DomainMax<T>(),
                               DomainValue<T>(0), DomainValue<T>(1),
                               DomainValue<T>(-1)};
  const auto target =
      rund::compute::on(rund::node::test_contract::target_for(backend, 2u));
  auto program =
      target
          .template map<T>("backend-domain-boundary", input.size(),
                           [](auto value) { return Store<T>(value + value); })
          .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    return false;
  }
  auto output = job->read();
  if (!output || output->size() != input.size()) {
    return false;
  }
  std::array<std::uint64_t, 5> values{};
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = DomainBits((*output)[index]);
    const std::uint64_t expected = [&] {
      if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>> ||
                    std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
        if (index < 2u) {
          return DomainBits(input[index]);
        }
      }
      return sizeof(T) == sizeof(std::uint32_t)
                 ? static_cast<std::uint64_t>(static_cast<std::uint32_t>(
                       DomainBits(input[index]) * 2u))
                 : DomainBits(input[index]) * 2u;
    }();
    if (values[index] != expected) {
      std::fprintf(stderr,
                   "boundary value mismatch backend=%u bytes=%zu index=%zu "
                   "actual=%llu expected=%llu\n",
                   static_cast<unsigned>(backend), sizeof(T), index,
                   static_cast<unsigned long long>(values[index]),
                   static_cast<unsigned long long>(expected));
      return false;
    }
  }
  const auto stats = job->stats();
  if (backend == rund::compute::Backend::Cpu) {
    reference = {.values = values,
                 .graph = stats.graph_hash,
                 .output = stats.output_hash};
    return reference.graph != 0u && reference.output != 0u;
  }
  return values == reference.values && stats.graph_hash == reference.graph &&
         stats.output_hash == reference.output;
}

template <class T>
[[nodiscard]] bool CheckDomain(const rund::compute::Backend backend,
                               DomainEvidence &reference) {
  const std::array<T, 4> input{DomainValue<T>(1), DomainValue<T>(2),
                               DomainValue<T>(3), DomainValue<T>(4)};
  const std::array<T, 4> expected{DomainValue<T>(2), DomainValue<T>(4),
                                  DomainValue<T>(6), DomainValue<T>(8)};
  const auto target =
      rund::compute::on(rund::node::test_contract::target_for(backend, 2u));
  auto program =
      target
          .template map<T>("backend-domain-parity", input.size(),
                           [](auto value) {
                             const auto zero = value - value;
                             return Store<T>(rund::compute::select(
                                 value > zero, value + value, value));
                           })
          .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(input);
  if (!job || !job->run()) {
    std::fprintf(stderr, "compute domain backend=%u bytes=%zu failed: %.*s\n",
                 static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(job.error().size()), job.error().data());
    return false;
  }
  auto output = job->read();
  if (!output || output->size() != expected.size() ||
      !EqualDomain(std::array<T, 4>{(*output)[0], (*output)[1], (*output)[2],
                                    (*output)[3]},
                   expected)) {
    return false;
  }
  const rund::compute::Stats stats = job->stats();
  if (stats.graph_hash == 0u || stats.output_hash == 0u) {
    return false;
  }
  if (backend == rund::compute::Backend::Cpu) {
    reference =
        DomainEvidence{.graph = stats.graph_hash, .output = stats.output_hash};
    return true;
  }
  return stats.graph_hash == reference.graph &&
         stats.output_hash == reference.output;
}

template <class T> [[nodiscard]] bool CheckDomainMatrix() {
  DomainEvidence domain{};
  DomainEvidence boundary{};
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (!CheckDomain<T>(backend, domain)) {
      std::fprintf(stderr, "domain parity failed backend=%u bytes=%zu\n",
                   static_cast<unsigned>(backend), sizeof(T));
      return false;
    }
    if (!CheckBoundary<T>(backend, boundary)) {
      std::fprintf(stderr, "domain boundary failed backend=%u bytes=%zu\n",
                   static_cast<unsigned>(backend), sizeof(T));
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CheckDomains() {
  return CheckDomainMatrix<std::int32_t>() &&
         CheckDomainMatrix<std::uint32_t>() &&
         CheckDomainMatrix<std::int64_t>() &&
         CheckDomainMatrix<std::uint64_t>() &&
         CheckDomainMatrix<rund::compute::Fixed<1, 31>>() &&
         CheckDomainMatrix<rund::compute::Fixed<1, 63>>();
}

} // namespace rund_node_backend_contract
