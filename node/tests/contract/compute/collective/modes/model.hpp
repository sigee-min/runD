#pragma once

#include "local.hpp"

#include "../../../target/selection.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund_node_collective_modes {

using rund::node::test_contract::flow_on;

template <class T> [[nodiscard]] constexpr T Zero() noexcept {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::zero();
  } else {
    return T{0};
  }
}

template <class T> [[nodiscard]] constexpr T Minimum() noexcept {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::min();
  } else {
    return std::numeric_limits<T>::min();
  }
}

template <class T> [[nodiscard]] constexpr T Maximum() noexcept {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::max();
  } else {
    return std::numeric_limits<T>::max();
  }
}

template <class T> [[nodiscard]] constexpr T Small() noexcept {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::from_raw(1);
  } else {
    return T{1};
  }
}

template <class T> [[nodiscard]] constexpr T NegativeSmall() noexcept {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::from_raw(-1);
  } else {
    return T{-1};
  }
}

template <class T>
[[nodiscard]] constexpr T DomainValue(const std::int64_t value) noexcept {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    using Raw = typename T::Raw;
    const auto magnitude =
        static_cast<std::uint64_t>(value < 0 ? -value : value);
    const auto scaled = magnitude << T::fraction_bits;
    return T::from_raw(
        static_cast<Raw>(value < 0 ? -static_cast<std::int64_t>(scaled)
                                   : static_cast<std::int64_t>(scaled)));
  } else {
    return static_cast<T>(value);
  }
}

template <class T> [[nodiscard]] constexpr auto Store(const auto &value) {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return rund::compute::quantize<T>(value);
  } else {
    return value;
  }
}

template <class Job> [[nodiscard]] auto Read(Job &job) {
  if constexpr (requires { job.read_all(); }) {
    return job.read_all();
  } else {
    return job.read();
  }
}

[[nodiscard]] inline bool Warm(const rund::compute::Stats stats) noexcept {
  return stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.uploaded_bytes == 0u && stats.download_events == 0u &&
         stats.graph_hash != 0u;
}

template <class Job>
[[nodiscard]] bool SameSuccess(Job &job, const rund::compute::Backend backend,
                               const char *const family, Hash &reference) {
  const rund::compute::Status warmup = job.run();
  const rund::compute::Status run = warmup ? job.run() : warmup;
  if (!run) {
    std::fprintf(stderr,
                 "compute collective modes backend=%u family=%s reason=%.*s\n",
                 static_cast<unsigned>(backend), family,
                 static_cast<int>(run.error().size()), run.error().data());
    return false;
  }
  const rund::compute::Stats warm = job.stats();
  auto output = Read(job);
  const rund::compute::Stats read = job.stats();
  const Hash observed{.graph = warm.graph_hash, .output = read.output_hash};
  bool same = output && Warm(warm) && warm.backend == backend &&
              observed.graph != 0u && observed.output != 0u;
  if (same && backend == rund::compute::Backend::Cpu) {
    reference = observed;
  } else if (same) {
    same = reference.graph != 0u && reference.output != 0u &&
           reference.graph == observed.graph &&
           reference.output == observed.output;
  }
  if (!same) {
    std::fprintf(stderr,
                 "compute collective modes mismatch backend=%u family=%s "
                 "graph=%llu/%llu output=%llu/%llu\n",
                 static_cast<unsigned>(backend), family,
                 static_cast<unsigned long long>(reference.graph),
                 static_cast<unsigned long long>(observed.graph),
                 static_cast<unsigned long long>(reference.output),
                 static_cast<unsigned long long>(observed.output));
  }
  return same;
}

template <class Program, class... Ranges>
[[nodiscard]] bool
SameFailure(Program &program, const rund::compute::Backend backend,
            const char *const family, const std::string_view expected,
            rund::compute::graph::Fingerprint &reference, Ranges &...ranges) {
  auto job = program.resident(ranges...);
  if (!job) {
    std::fprintf(stderr,
                 "compute collective failure resident backend=%u family=%s "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), family,
                 static_cast<int>(job.error().size()), job.error().data());
    return false;
  }
  const rund::compute::Status status = job->run();
  const rund::compute::graph::Fingerprint fingerprint = program.fingerprint();
  bool same = !status && status.error() == expected && fingerprint;
  if (same && backend == rund::compute::Backend::Cpu) {
    reference = fingerprint;
  } else if (same) {
    same = reference && reference == fingerprint;
  }
  if (!same) {
    std::fprintf(stderr,
                 "compute collective failure mismatch backend=%u family=%s "
                 "actual=%.*s expected=%.*s\n",
                 static_cast<unsigned>(backend), family,
                 static_cast<int>(status.error().size()), status.error().data(),
                 static_cast<int>(expected.size()), expected.data());
  }
  return same;
}

template <class T> [[nodiscard]] std::vector<T> TailValues() {
  std::vector<T> values(kTail);
  for (std::size_t index = 0u; index < values.size(); ++index) {
    if constexpr (std::is_unsigned_v<T>) {
      values[index] = DomainValue<T>(static_cast<std::int64_t>(index % 5u));
    } else {
      values[index] = DomainValue<T>(static_cast<std::int64_t>(index % 5u) - 2);
    }
  }
  return values;
}

template <class T>
[[nodiscard]] constexpr std::int64_t
TailInteger(const std::size_t index) noexcept {
  if constexpr (std::is_unsigned_v<T>) {
    return static_cast<std::int64_t>(index % 5u);
  } else {
    return static_cast<std::int64_t>(index % 5u) - 2;
  }
}

template <class T>
[[nodiscard]] std::array<std::vector<T>, 3u>
ExpectedWindows(const std::vector<std::int64_t> &logical, const bool clip) {
  std::array<std::vector<T>, 3u> expected;
  for (auto &values : expected) {
    values.reserve(logical.size());
  }
  for (std::size_t index = 0u; index < logical.size(); ++index) {
    const bool has_left = index != 0u;
    const bool has_right = index + 1u != logical.size();
    const std::int64_t left = logical[has_left ? index - 1u : index];
    const std::int64_t value = logical[index];
    const std::int64_t right = logical[has_right ? index + 1u : index];
    const std::int64_t sum = value + (clip && !has_left ? 0 : left) +
                             (clip && !has_right ? 0 : right);
    std::int64_t minimum = value;
    std::int64_t maximum = value;
    if (!clip || has_left) {
      minimum = std::min(minimum, left);
      maximum = std::max(maximum, left);
    }
    if (!clip || has_right) {
      minimum = std::min(minimum, right);
      maximum = std::max(maximum, right);
    }
    expected[0u].push_back(DomainValue<T>(sum));
    expected[1u].push_back(DomainValue<T>(minimum));
    expected[2u].push_back(DomainValue<T>(maximum));
  }
  return expected;
}

[[nodiscard]] inline std::vector<std::uint32_t>
SegmentHeads(const std::size_t count) {
  std::vector<std::uint32_t> heads(count, 0u);
  for (std::size_t index = 0u; index < count; index += 16u) {
    heads[index] = 1u;
  }
  return heads;
}

template <class T> struct CancellationFixture final {
  std::vector<T> input;
  std::vector<T> inclusive;
  std::vector<T> exclusive;
  std::vector<std::uint32_t> heads;
};

template <class T>
[[nodiscard]] CancellationFixture<T> CrossBlockCancellation() {
  static_assert(std::is_signed_v<T> || rund::compute::detail::FixedValue<T>);
  constexpr std::size_t kCount = 512u;
  const auto parts = [] {
    std::array<T, 4u> values{};
    if constexpr (rund::compute::detail::FixedValue<T>) {
      using Raw = typename T::Raw;
      const Raw peak = T::max().raw();
      const Raw half = static_cast<Raw>(peak / Raw{2});
      values = {T::from_raw(static_cast<Raw>(-half)), T::from_raw(peak),
                T::from_raw(half), T::from_raw(static_cast<Raw>(peak - half))};
    } else {
      const T peak = std::numeric_limits<T>::max();
      const T half = static_cast<T>(peak / T{2});
      values = {static_cast<T>(-half), peak, half, static_cast<T>(peak - half)};
    }
    return values;
  }();
  const T negative = parts[0u];
  const T peak = parts[1u];
  const T half = parts[2u];
  const T merged = parts[3u];
  CancellationFixture<T> fixture{.input = std::vector<T>(kCount, Zero<T>()),
                                 .inclusive = std::vector<T>(kCount, peak),
                                 .exclusive = std::vector<T>(kCount, peak),
                                 .heads =
                                     std::vector<std::uint32_t>(kCount, 0u)};
  fixture.input[0u] = negative;
  fixture.input[256u] = peak;
  fixture.input[257u] = half;
  fixture.heads[0u] = 1u;
  std::fill(fixture.inclusive.begin(), fixture.inclusive.begin() + 256u,
            negative);
  fixture.inclusive[256u] = merged;
  fixture.exclusive[0u] = Zero<T>();
  std::fill(fixture.exclusive.begin() + 1u, fixture.exclusive.begin() + 257u,
            negative);
  fixture.exclusive[257u] = merged;
  return fixture;
}

} // namespace rund_node_collective_modes
