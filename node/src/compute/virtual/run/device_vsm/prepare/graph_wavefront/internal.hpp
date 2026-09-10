#pragma once

#include "../graph_wavefront.hpp"

#include "../../../../../type.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {
class Pool;
}

namespace rund::compute::detail::device_vsm_product_detail {

struct GraphResidentDraft final {
  const residency::TiledGraphPlan &plan;
  const residency::Pool &pool;
  std::span<const std::uint32_t> graph_input_resources;
  const node::accel::detail::DeviceVsmGraphWavefrontProof &wavefront;
  node::accel::detail::DeviceVsmGraphResidentProof &result;
  const char *&reason;
  std::span<const residency::TiledGraphResource> resources;
  std::span<const residency::TiledGraphStage> stages;
  std::span<const residency::TiledGraphPhysicalClass> owners;
  std::array<const residency::TiledGraphPhysicalClass *,
             node::accel::detail::DeviceVsmGraphResidentPhysicalCapacity>
      internal_owners{};
  std::size_t internal_count{};
  std::size_t ports{};
  std::size_t output_count{};
  Type root_type{Type::I32};
};

extern const char GraphResidentShapeInvalid[];
extern const char GraphResidentResourceOwnerInvalid[];
extern const char GraphResidentOwnerArenaInvalid[];
extern const char GraphResidentBankBindingInvalid[];
extern const char GraphResidentStagePortsInvalid[];
extern const char GraphResidentProofValidationInvalid[];

void remember_graph_resident_reason(const char *key,
                                    const char *&reason) noexcept;

[[nodiscard]] bool validate_graph_resident_shape(GraphResidentDraft &) noexcept;
[[nodiscard]] bool
project_graph_resident_resources(GraphResidentDraft &) noexcept;
[[nodiscard]] bool project_graph_resident_owners(GraphResidentDraft &) noexcept;
[[nodiscard]] bool project_graph_resident_stages(GraphResidentDraft &) noexcept;

} // namespace rund::compute::detail::device_vsm_product_detail
