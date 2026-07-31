#pragma once

#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include "../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund_node_backend_contract {

template <class Program>
[[nodiscard]] bool UsesBackend(const Program &program,
                               const rund::compute::Backend expected) {
  const auto backend = program.backend();
  return backend && *backend == expected;
}

struct DomainEvidence final {
  std::array<std::uint64_t, 5> values{};
  std::uint64_t graph{};
  std::uint64_t output{};
};

struct PrimitiveEvidence final {
  std::vector<std::uint64_t> values;
  std::uint64_t graph{};
  std::uint64_t output{};
};

template <class T> [[nodiscard]] constexpr auto Store(const auto &value) {
  if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>> ||
                std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
    return rund::compute::quantize<T>(value);
  } else {
    return value;
  }
}

template <class T> [[nodiscard]] T DomainValue(const std::int64_t value) {
  if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(static_cast<std::int32_t>(value));
  } else if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
    return T::from_raw(value);
  } else {
    return static_cast<T>(value);
  }
}

template <class T, std::size_t N>
[[nodiscard]] bool EqualDomain(const std::array<T, N> &lhs,
                               const std::array<T, N> &rhs) {
  for (std::size_t index = 0u; index < N; ++index) {
    if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>> ||
                  std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
      if (lhs[index].raw() != rhs[index].raw()) {
        return false;
      }
    } else if (lhs[index] != rhs[index]) {
      return false;
    }
  }
  return true;
}

template <class T> [[nodiscard]] std::uint64_t DomainBits(const T value) {
  if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>>) {
    return static_cast<std::uint32_t>(value.raw());
  } else if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
    return static_cast<std::uint64_t>(value.raw());
  } else if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
    return static_cast<std::uint32_t>(value);
  } else {
    return static_cast<std::uint64_t>(value);
  }
}

template <class T> [[nodiscard]] T DomainMin() {
  if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(std::numeric_limits<std::int32_t>::min());
  } else if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
    return T::from_raw(std::numeric_limits<std::int64_t>::min());
  } else {
    return std::numeric_limits<T>::min();
  }
}

template <class T> [[nodiscard]] T DomainMax() {
  if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(std::numeric_limits<std::int32_t>::max());
  } else if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
    return T::from_raw(std::numeric_limits<std::int64_t>::max());
  } else {
    return std::numeric_limits<T>::max();
  }
}

template <class T, std::size_t N, class Build>
[[nodiscard]] bool
CheckTwoInputPrimitive(const rund::compute::Backend backend,
                       const char *const name,
                       const std::array<std::uint32_t, N> &side, Build &&build,
                       PrimitiveEvidence &reference) {
  constexpr std::array<std::int64_t, 6> raw{3, 1, 4, 2, 5, 1};
  std::array<T, N> input{};
  for (std::size_t index = 0u; index < N; ++index) {
    input[index] = DomainValue<T>(raw[index % raw.size()]);
  }
  const auto target =
      rund::compute::on(rund::node::test_contract::target_for(backend, 2u));
  auto values = target.template map<T>(
      name, input.size(), [](auto value) { return Store<T>(value); });
  auto output = build(std::move(values), side.size());
  auto program = std::move(output).compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(std::span<const T>{input},
                               std::span<const std::uint32_t>{side});
  if (!job || !job->run()) {
    std::fprintf(stderr,
                 "compute segmented primitive=%s backend=%u bytes=%zu failed: "
                 "%.*s\n",
                 name, static_cast<unsigned>(backend), sizeof(T),
                 static_cast<int>(job.error().size()), job.error().data());
    return false;
  }
  const auto output_values = job->read();
  if (!output_values) {
    return false;
  }
  std::vector<std::uint64_t> bits;
  bits.reserve(output_values->size());
  for (const T value : *output_values) {
    bits.push_back(DomainBits(value));
  }
  const rund::compute::Stats stats = job->stats();
  if (backend == rund::compute::Backend::Cpu) {
    reference = PrimitiveEvidence{.values = std::move(bits),
                                  .graph = stats.graph_hash,
                                  .output = stats.output_hash};
    return reference.graph != 0u && reference.output != 0u;
  }
  return bits == reference.values && stats.graph_hash == reference.graph &&
         stats.output_hash == reference.output;
}

} // namespace rund_node_backend_contract
