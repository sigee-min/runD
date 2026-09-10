#pragma once

#include "../backing.hpp"
#include "../local.hpp"
#include "../route.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund_node_test_virtual::product::window_detail {

inline constexpr std::size_t FrameElements = 16u;
inline constexpr std::size_t Radius = 2u;
inline constexpr std::size_t PayloadElements = FrameElements - Radius * 2u;
inline constexpr std::size_t WindowElements = 53u;
inline constexpr std::size_t WindowPages =
    (WindowElements + PayloadElements - 1u) / PayloadElements;

using WindowProgram = rund::compute::Program<std::int32_t(std::int32_t)>;
using WindowBuffer = rund::compute::VirtualBuffer<std::int32_t>;

[[nodiscard]] bool window_route_ok(const ProductRouteObservation &) noexcept;
[[nodiscard]] std::int32_t value(std::size_t) noexcept;
[[nodiscard]] std::int32_t expected_for(std::size_t, std::size_t) noexcept;
[[nodiscard]] std::int32_t expected(std::size_t) noexcept;
[[nodiscard]] std::int32_t expected_clip_min(std::size_t) noexcept;

[[nodiscard]] int CheckExactWindow(rund::compute::Backend,
                                   rund::compute::Device &,
                                   const WindowProgram &);
[[nodiscard]] int CheckClipWindow(rund::compute::Backend,
                                  rund::compute::Device &, const WindowBuffer &,
                                  std::span<std::byte>);
[[nodiscard]] int CheckTierWindow(rund::compute::Backend,
                                  rund::compute::Device &,
                                  const WindowProgram &,
                                  std::span<const std::int32_t>,
                                  std::span<std::byte>);

} // namespace rund_node_test_virtual::product::window_detail
