#pragma once

#include "../endpoint_mode.hpp"
#include "../route.hpp"

namespace rund::compute::detail::device_vsm_route_detail {

enum class GraphResidentDecisionKind : std::uint8_t {
  NotCandidate,
  Accepted,
  Decline,
};

struct GraphResidentDecision final {
  GraphResidentDecisionKind kind{GraphResidentDecisionKind::NotCandidate};
  device_vsm_product_detail::EndpointMode endpoint{
      device_vsm_product_detail::EndpointMode::Invalid};

  [[nodiscard]] bool accepted() const noexcept {
    return kind == GraphResidentDecisionKind::Accepted;
  }
  [[nodiscard]] bool declined() const noexcept {
    return kind == GraphResidentDecisionKind::Decline;
  }
};

[[nodiscard]] bool
graph_host_product_eligible(const VirtualPipelineState &,
                            std::span<VirtualBacking *const>,
                            const VirtualRunProjection &) noexcept;

[[nodiscard]] GraphResidentDecision
graph_resident_decision(const VirtualPipelineState &,
                        std::span<VirtualBacking *const>,
                        const VirtualRunProjection &) noexcept;

} // namespace rund::compute::detail::device_vsm_route_detail
