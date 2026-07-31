#include "model.hpp"

#include <concepts>
#include <span>
#include <type_traits>

namespace package_compute {

static_assert(std::is_empty_v<rund::compute::input::Bound>);
static_assert(std::is_empty_v<rund::compute::input::Deferred>);
static_assert(
    !std::same_as<rund::compute::input::Bound, rund::compute::input::Deferred>);

using InstalledBoundFlow =
    decltype(rund::compute::on(rund::compute::Target::cpu(),
                               std::declval<std::array<std::uint32_t, 4> &>())
                 .map("installed-bound-type",
                      [](auto value) { return value; }));
static_assert(std::same_as<InstalledBoundFlow,
                           rund::compute::Flow<std::uint32_t(std::uint32_t),
                                               rund::compute::stage::Exact,
                                               rund::compute::input::Bound>>);

using InstalledCompactFlow =
    decltype(rund::compute::on(rund::compute::Target::cpu(2u))
                 .map<std::uint32_t>("installed-compact-type", 4u,
                                     [](auto value) { return value; })
                 .compact({.capacity = 3u}));
static_assert(std::same_as<
              InstalledCompactFlow,
              rund::compute::Flow<std::uint32_t(std::uint32_t),
                                  rund::compute::stage::Bounded<std::uint32_t>,
                                  rund::compute::input::Deferred>>);
static_assert(
    !std::same_as<InstalledCompactFlow,
                  rund::compute::Flow<std::uint32_t(std::uint32_t),
                                      rund::compute::stage::Exact,
                                      rund::compute::input::Deferred>>);
using InstalledCompactCompile =
    decltype(std::declval<InstalledCompactFlow &&>().compile());
static_assert(std::same_as<InstalledCompactCompile,
                           rund::compute::Result<InstalledCompactProgram>>);
using InstalledCompactResident =
    decltype(std::declval<const InstalledCompactProgram &>().resident(
        std::declval<std::span<const std::uint32_t>>()));
static_assert(std::same_as<
              InstalledCompactResident,
              rund::compute::Result<rund::compute::Job<rund::compute::Bounded<
                  std::uint32_t, std::uint32_t>(std::uint32_t)>>>);

} // namespace package_compute
