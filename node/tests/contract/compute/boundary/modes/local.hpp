#pragma once

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include "../../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

namespace rund::node::test_contract::boundary_modes {

constexpr std::size_t kTail = 255u;

struct Hash final {
  std::uint64_t graph{};
  std::uint64_t output{};
};

struct DomainEvidence final {
  Hash movement{};
  Hash transpose{};
  Hash product{};
};

struct FixedEvidence final {
  Hash transform{};
  Hash factor{};
  Hash solve{};
  Hash spectrum{};
};

template <class T> [[nodiscard]] constexpr T Zero() {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::zero();
  } else {
    return T{0};
  }
}

template <class T> [[nodiscard]] constexpr T Small() {
  if constexpr (std::same_as<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(std::int32_t{1} << 27u);
  } else if constexpr (std::same_as<T, rund::compute::Fixed<1, 63>>) {
    return T::from_raw(std::int64_t{1} << 59u);
  } else {
    return T{1};
  }
}

template <class T> [[nodiscard]] constexpr T SmallProduct() {
  if constexpr (std::same_as<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(std::int32_t{1} << 23u);
  } else if constexpr (std::same_as<T, rund::compute::Fixed<1, 63>>) {
    return T::from_raw(std::int64_t{1} << 55u);
  } else {
    return T{1};
  }
}

template <class T> [[nodiscard]] constexpr T Minimum() {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::min();
  } else {
    return std::numeric_limits<T>::min();
  }
}

template <class T> [[nodiscard]] constexpr T Maximum() {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::max();
  } else {
    return std::numeric_limits<T>::max();
  }
}

[[nodiscard]] inline auto Target(const rund::compute::Backend backend) {
  using namespace rund::compute;
  return on(rund::node::test_contract::target_for(backend, 2u));
}

template <class Job> [[nodiscard]] auto Read(Job &job) {
  if constexpr (requires { job.read_all(); }) {
    return job.read_all();
  } else {
    return job.read();
  }
}

template <class Job, class Validate>
[[nodiscard]] bool CheckSuccess(Job &job, const rund::compute::Backend backend,
                                const char *const family, Hash &reference,
                                Validate &&validate) {
  if (!job.run()) {
    std::fprintf(stderr, "boundary modes run backend=%u family=%s\n",
                 static_cast<unsigned>(backend), family);
    return false;
  }
  auto output = Read(job);
  const rund::compute::Stats observed = job.stats();
  if (!output || !validate(*output) || observed.graph_hash == 0u ||
      observed.output_hash == 0u || observed.backend != backend) {
    std::fprintf(stderr, "boundary modes result backend=%u family=%s\n",
                 static_cast<unsigned>(backend), family);
    return false;
  }
  const Hash hash{.graph = observed.graph_hash, .output = observed.output_hash};
  if (reference.graph == 0u) {
    reference = hash;
    return true;
  }
  const bool same =
      reference.graph == hash.graph && reference.output == hash.output;
  if (!same) {
    std::fprintf(stderr,
                 "boundary modes parity backend=%u family=%s "
                 "graph=%llu/%llu output=%llu/%llu\n",
                 static_cast<unsigned>(backend), family,
                 static_cast<unsigned long long>(reference.graph),
                 static_cast<unsigned long long>(hash.graph),
                 static_cast<unsigned long long>(reference.output),
                 static_cast<unsigned long long>(hash.output));
  }
  return same;
}

template <class T> [[nodiscard]] std::vector<T> BoundaryValues() {
  std::vector<T> values(kTail, Zero<T>());
  values[0u] = Minimum<T>();
  values[1u] = Maximum<T>();
  return values;
}

[[nodiscard]] bool CheckDomainModes();
[[nodiscard]] bool CheckBoundedModes();
[[nodiscard]] bool CheckUnsignedModes();
[[nodiscard]] bool CheckFixedModes();

} // namespace rund::node::test_contract::boundary_modes
