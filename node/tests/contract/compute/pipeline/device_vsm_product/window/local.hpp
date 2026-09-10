#pragma once

#include "../../persistent_product/local.hpp"
#include "../../persistent_product/model.hpp"

#include "../evidence.hpp"
#include "../route.hpp"

#include <rund/compute/virtual.hpp>

#include "src/accel/kernel/residency/device_vsm/geometry.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_test_device_vsm_product::window_test {

inline constexpr std::size_t FrameElements = 16u;
inline constexpr std::size_t Radius = 2u;
inline constexpr std::size_t PayloadElements = FrameElements - Radius * 2u;
inline constexpr std::uint64_t WindowRingPages = 3u;

struct I32WindowProduct final {
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state{};
  std::shared_ptr<rund::compute::VirtualBacking> input{};
  std::shared_ptr<rund::compute::VirtualBacking> output{};
  std::vector<std::int32_t> expected{};
  bool resident{};
};

enum class Operation : std::uint8_t {
  Sum,
  Minimum,
  Maximum,
};

enum class Boundary : std::uint8_t {
  Clamp,
  Clip,
};

enum class PipelineShape : std::uint8_t {
  Window,
  MapWindowMap,
  MapDagWindowMapDag,
  MapChainWindowMapChain,
  MapTripleChainWindowMapTripleChain,
};

[[nodiscard]] bool PrepareWindowProduct(
    rund::compute::Backend, std::uint64_t, Operation, Boundary, PipelineShape,
    rund_node_test_persistent_product::PreparedProduct &, bool &);
[[nodiscard]] bool
PrepareWindowRingProduct(rund::compute::Backend,
                         rund_node_test_persistent_product::PreparedProduct &,
                         bool &);
[[nodiscard]] bool ExactWindowOutput(
    const rund_node_test_persistent_product::PreparedProduct &) noexcept;
[[nodiscard]] bool PrepareI32WindowProduct(rund::compute::Backend,
                                           std::uint64_t, bool,
                                           I32WindowProduct &, bool &);
[[nodiscard]] bool ExactI32WindowOutput(const I32WindowProduct &) noexcept;

namespace model {

using DeviceVsmProductOwner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;
using WindowRingPlan = rund::node::accel::detail::DeviceVsmWindowRingPlan;

[[nodiscard]] std::uint64_t RingLogicalBytes(std::uint64_t) noexcept;
[[nodiscard]] std::uint64_t RingPeak(std::uint64_t) noexcept;
[[nodiscard]] std::uint64_t I32BackingVersion(
    const std::shared_ptr<rund::compute::VirtualBacking> &) noexcept;
[[nodiscard]] std::uint64_t I32BackingRecovery(
    const std::shared_ptr<rund::compute::VirtualBacking> &) noexcept;

[[nodiscard]] bool ExactRingStats(const rund::compute::Stats &, std::uint64_t,
                                  std::uint64_t, std::uint64_t) noexcept;
[[nodiscard]] bool
ExactRingProof(const rund_node_test_persistent_product::PreparedProduct &,
               const rund_node_test_device_vsm_product::RouteObservation &,
               const std::shared_ptr<DeviceVsmProductOwner> &,
               WindowRingPlan &) noexcept;
[[nodiscard]] bool
ExactRingFailure(const rund::compute::Status &,
                 const rund_node_test_device_vsm_product::RouteObservation &,
                 std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                 std::uint64_t, std::uint64_t,
                 const rund::compute::Stats &) noexcept;
[[nodiscard]] bool
PrepareRingShape(rund::compute::Backend, PipelineShape,
                 rund_node_test_persistent_product::PreparedProduct &, bool &);
[[nodiscard]] bool ExactI32Ring(
    const I32WindowProduct &, std::uint64_t, const rund::compute::Status &,
    const rund_node_test_device_vsm_product::RouteObservation &, std::uint64_t,
    std::uint64_t, std::uint64_t, std::uint64_t) noexcept;

} // namespace model

[[nodiscard]] bool
RunWindowRingProduct(rund::compute::Backend,
                     rund_node_test_persistent_product::NativeQueueCounter,
                     bool &, PipelineShape);
[[nodiscard]] bool
RunI32WindowProductCases(rund::compute::Backend,
                         rund_node_test_persistent_product::NativeQueueCounter,
                         bool &) noexcept;
[[nodiscard]] bool RunWindowProductCases(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;

} // namespace rund_node_test_device_vsm_product::window_test
