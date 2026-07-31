#pragma once

#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <utility>
#include <vector>

namespace package_compute {

struct PackageValue final {};
struct PackageWeight final {};
struct PackageScore final {};

[[nodiscard]] inline int FlowMismatch(const int line) {
  std::fprintf(stderr, "package flow-primitives mismatch at line %d\n", line);
  return 2;
}

inline constexpr std::array<std::uint32_t, 4> values{3u, 1u, 4u, 2u};
inline constexpr std::array<std::uint32_t, 4> indices{1u, 3u, 0u, 2u};
inline constexpr std::array<std::uint32_t, 4> flags{1u, 0u, 1u, 0u};
inline constexpr std::array<std::uint32_t, 4> heads{1u, 0u, 1u, 0u};
inline constexpr std::array<std::int64_t, 4> filter_values{1, 2, 3, 4};
inline constexpr std::array<std::uint32_t, 4u> combine_side{1u, 2u, 3u, 4u};

using InstalledCompactProgram =
    rund::compute::Program<rund::compute::Bounded<std::uint32_t, std::uint32_t>(
        std::uint32_t)>;

[[nodiscard]] int Filter();
[[nodiscard]] int Record();
[[nodiscard]] int Group();
[[nodiscard]] int Collective();
[[nodiscard]] int Compact();
[[nodiscard]] int Basic();
[[nodiscard]] int Math();
[[nodiscard]] int Program();

} // namespace package_compute
