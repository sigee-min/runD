#include "../../../../../src/compute/buffer/state.hpp"
#include "local.hpp"

#include "src/accel/context/capability.hpp"
#include "src/compute/device/residency/pool.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/pipeline/residency/model.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace rund_node_test_pipeline_residency {
namespace {

[[nodiscard]] rund::compute::detail::residency::PoolLayout
layout(const std::uint32_t frames,
       const rund::compute::detail::Type type =
           rund::compute::detail::Type::U64) noexcept {
  using namespace rund::compute::detail;
  return residency::PoolLayout{
      .input_type = type,
      .intermediate_type = type,
      .control_type = Type::U64,
      .output_type = type,
      .input_page_bytes = 128u,
      .intermediate_page_bytes = 128u,
      .control_page_bytes = 8u,
      .output_page_bytes = 8u,
      .frame_capacity = frames,
      .host_frame_capacity = frames,
      .host_output_frame_capacity = frames,
  };
}

} // namespace

int CheckPoolLending() {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  using namespace rund::compute::detail::residency;

  auto opened = open(Target::cpu(2u));
  if (!opened) {
    return 1;
  }
  const std::shared_ptr<DeviceState> device = DeviceAccess::state(*opened);
  if (device == nullptr || device->residency == nullptr) {
    return 2;
  }
  const std::array classes{
      TiledGraphPhysicalClass{.physical_id = 1u,
                              .role = GraphResourceRole::Input,
                              .type = Type::U64,
                              .page_bytes = 128u},
      TiledGraphPhysicalClass{.physical_id = 2u,
                              .role = GraphResourceRole::Intermediate,
                              .type = Type::U64,
                              .page_bytes = 128u},
      TiledGraphPhysicalClass{.physical_id = 3u,
                              .role = GraphResourceRole::Output,
                              .type = Type::U64,
                              .page_bytes = 8u},
      TiledGraphPhysicalClass{.physical_id = 4u,
                              .role = GraphResourceRole::Intermediate,
                              .type = Type::U64,
                              .color = 1u,
                              .page_bytes = 128u},
  };
  const std::shared_ptr<Pool> large =
      device->residency->acquire(device, layout(4u), classes);
  const std::shared_ptr<Pool> small =
      device->residency->acquire(device, layout(2u), classes);
  if (large == nullptr || small == nullptr || large == small) {
    return 3;
  }
  std::array<const PhysicalArena *, classes.size()> arenas{};
  for (std::size_t index = 0u; index < classes.size(); ++index) {
    const PoolPhysicalOwner *const outer =
        large->graph_owner(classes[index].physical_id);
    const PoolPhysicalOwner *const inner =
        small->graph_owner(classes[index].physical_id);
    if (outer == nullptr || inner == nullptr || outer->arena == nullptr ||
        outer->arena != inner->arena || outer->arena->frame_capacity != 4u) {
      return 4;
    }
    arenas[index] = outer->arena.get();
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (arenas[prior] == arenas[index]) {
        return 5;
      }
    }
    for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
      if (inner->buffers[bank] != inner->arena->buffers[bank] ||
          inner->bank_regions[bank].first !=
              inner->arena->bank_regions[bank].first ||
          inner->bank_regions[bank].count != 2u ||
          inner->arena->bank_regions[bank].count != 4u) {
        return 6;
      }
    }
  }
  std::uint64_t footprint = 0u;
  if (!graph_pool_footprint(*small, classes, footprint) || footprint != 392u) {
    return 7;
  }

  // Type/FixedFormat are semantic view identity, not raw allocation identity.
  // A second live layout with equal role/page geometry and a different scalar
  // type must retain the same two physical byte banks while exposing exact
  // typed BufferState views. Cache materialization identity still contains the
  // semantic type, so this is physical lending rather than a false content hit.
  const std::array u32_classes{
      TiledGraphPhysicalClass{.physical_id = 1u,
                              .role = GraphResourceRole::Input,
                              .type = Type::U32,
                              .page_bytes = 128u},
      TiledGraphPhysicalClass{.physical_id = 2u,
                              .role = GraphResourceRole::Intermediate,
                              .type = Type::U32,
                              .page_bytes = 128u},
      TiledGraphPhysicalClass{.physical_id = 3u,
                              .role = GraphResourceRole::Output,
                              .type = Type::U32,
                              .page_bytes = 8u},
      TiledGraphPhysicalClass{.physical_id = 4u,
                              .role = GraphResourceRole::Intermediate,
                              .type = Type::U32,
                              .color = 1u,
                              .page_bytes = 128u},
  };
  const std::shared_ptr<Pool> typed =
      device->residency->acquire(device, layout(2u, Type::U32), u32_classes);
  if (typed == nullptr || typed == large || typed == small) {
    return 8;
  }
  for (std::size_t index = 0u; index < classes.size(); ++index) {
    const PoolPhysicalOwner *const canonical =
        large->graph_owner(classes[index].physical_id);
    const PoolPhysicalOwner *const view =
        typed->graph_owner(u32_classes[index].physical_id);
    if (canonical == nullptr || view == nullptr ||
        canonical->arena != view->arena ||
        view->physical_class.type != Type::U32) {
      return 9;
    }
    for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
      if (view->buffers[bank] == nullptr ||
          view->buffers[bank]->type != Type::U32 ||
          view->buffers[bank]->bytes != canonical->buffers[bank]->bytes ||
          !canonical->buffers[bank]->memory_accounted ||
          view->buffers[bank]->memory_accounted ||
          !same_physical_buffer(*view->buffers[bank],
                                *canonical->buffers[bank]) ||
          view->bank_regions[bank].tier != canonical->bank_regions[bank].tier ||
          view->bank_regions[bank].role != canonical->bank_regions[bank].role ||
          view->bank_regions[bank].first !=
              canonical->bank_regions[bank].first ||
          view->bank_regions[bank].count != 2u) {
        return 10;
      }
    }
  }
  footprint = 0u;
  if (!graph_pool_footprint(*typed, u32_classes, footprint) ||
      footprint != 392u) {
    return 11;
  }
  const std::uint64_t small_metadata = small->retained_host_bytes();
  const std::uint64_t typed_metadata = typed->retained_host_bytes();
  if (small_metadata == 0u ||
      typed_metadata !=
          small_metadata + 2u * classes.size() * sizeof(BufferState)) {
    return 12;
  }

  // Equal committed bank bytes are the physical compatibility boundary, not
  // one page geometry.  This view halves page_bytes, therefore exposes C=8
  // semantic frames over the same C=4 raw banks.  Its Authority coordinates
  // are distinct so the two incompatible metadata views cannot be valid at
  // once, while prepared buffers retain the same native allocation.
  const PoolLayout reblocked_layout{
      .input_type = Type::U32,
      .intermediate_type = Type::U32,
      .control_type = Type::U64,
      .output_type = Type::U32,
      .input_page_bytes = 64u,
      .intermediate_page_bytes = 64u,
      .control_page_bytes = 8u,
      .output_page_bytes = 4u,
      .frame_capacity = 2u,
      .host_frame_capacity = 2u,
      .host_output_frame_capacity = 2u,
  };
  const std::array reblocked_classes{
      TiledGraphPhysicalClass{.physical_id = 1u,
                              .role = GraphResourceRole::Input,
                              .type = Type::U32,
                              .page_bytes = 64u},
      TiledGraphPhysicalClass{.physical_id = 2u,
                              .role = GraphResourceRole::Intermediate,
                              .type = Type::U32,
                              .page_bytes = 64u},
      TiledGraphPhysicalClass{.physical_id = 3u,
                              .role = GraphResourceRole::Output,
                              .type = Type::U32,
                              .page_bytes = 4u},
      TiledGraphPhysicalClass{.physical_id = 4u,
                              .role = GraphResourceRole::Intermediate,
                              .type = Type::U32,
                              .color = 1u,
                              .page_bytes = 64u},
  };
  const std::shared_ptr<Pool> reblocked =
      device->residency->acquire(device, reblocked_layout, reblocked_classes);
  if (reblocked == nullptr) {
    return 13;
  }
  for (std::size_t index = 0u; index < classes.size(); ++index) {
    const PoolPhysicalOwner *const canonical =
        large->graph_owner(classes[index].physical_id);
    const PoolPhysicalOwner *const view =
        reblocked->graph_owner(reblocked_classes[index].physical_id);
    if (canonical == nullptr || view == nullptr ||
        canonical->arena != view->arena || !view->owns_regions ||
        view->frame_capacity != 8u ||
        view->cache_regions == canonical->arena->bank_regions) {
      return 14;
    }
    for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
      if (!same_physical_buffer(*canonical->buffers[bank],
                                *view->buffers[bank]) ||
          view->buffers[bank]->bytes != canonical->buffers[bank]->bytes ||
          view->cache_regions[bank].count != 8u ||
          view->bank_regions[bank].count != 2u ||
          view->cache_regions[bank].role !=
              (reblocked_classes[index].role == GraphResourceRole::Input
                   ? FrameRole::Input
               : reblocked_classes[index].role ==
                       GraphResourceRole::Intermediate
                   ? FrameRole::Intermediate
                   : FrameRole::Output)) {
        return 15;
      }
    }
  }
  footprint = 0u;
  if (!graph_pool_footprint(*reblocked, reblocked_classes, footprint) ||
      footprint != 196u) {
    return 16;
  }

  // The accelerator view must retain the same authenticated native allocation,
  // not allocate and copy another typed payload. Adapter absence keeps this
  // structural contract portable; available Metal executes the exact owner
  // construction and footprint validation here.
  auto opened_metal = open(Target::metal());
  if (!opened_metal) {
    return opened_metal.reason() == Reason::AdapterUnavailable ? 0 : 17;
  }
  const std::shared_ptr<DeviceState> metal_device =
      DeviceAccess::state(*opened_metal);
  if (metal_device == nullptr || metal_device->residency == nullptr) {
    return 18;
  }
  const std::shared_ptr<Pool> metal_u64 =
      metal_device->residency->acquire(metal_device, layout(4u), classes);
  const std::shared_ptr<Pool> metal_u32 = metal_device->residency->acquire(
      metal_device, layout(2u, Type::U32), u32_classes);
  const std::shared_ptr<Pool> metal_reblocked =
      metal_device->residency->acquire(metal_device, reblocked_layout,
                                       reblocked_classes);
  if (metal_u64 == nullptr || metal_u32 == nullptr ||
      metal_reblocked == nullptr) {
    return 19;
  }
  for (std::size_t index = 0u; index < classes.size(); ++index) {
    const PoolPhysicalOwner *const canonical =
        metal_u64->graph_owner(classes[index].physical_id);
    const PoolPhysicalOwner *const view =
        metal_u32->graph_owner(u32_classes[index].physical_id);
    const PoolPhysicalOwner *const reblocked_view =
        metal_reblocked->graph_owner(reblocked_classes[index].physical_id);
    if (canonical == nullptr || view == nullptr || reblocked_view == nullptr ||
        canonical->arena != view->arena ||
        canonical->arena != reblocked_view->arena ||
        !reblocked_view->owns_regions || reblocked_view->frame_capacity != 8u) {
      return 20;
    }
    for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
      const AccelBufferState *const canonical_accel =
          canonical->buffers[bank] == nullptr
              ? nullptr
              : accel_buffer(*canonical->buffers[bank]);
      const AccelBufferState *const view_accel =
          view->buffers[bank] == nullptr ? nullptr
                                         : accel_buffer(*view->buffers[bank]);
      const AccelBufferState *const reblocked_accel =
          reblocked_view->buffers[bank] == nullptr
              ? nullptr
              : accel_buffer(*reblocked_view->buffers[bank]);
      const auto canonical_token =
          canonical_accel == nullptr
              ? nullptr
              : rund::node::accel::detail::LookupAccelBufferToken(
                    canonical_accel->buffer.handle);
      const auto view_token =
          view_accel == nullptr
              ? nullptr
              : rund::node::accel::detail::LookupAccelBufferToken(
                    view_accel->buffer.handle);
      const auto reblocked_token =
          reblocked_accel == nullptr
              ? nullptr
              : rund::node::accel::detail::LookupAccelBufferToken(
                    reblocked_accel->buffer.handle);
      if (canonical_accel == nullptr || view_accel == nullptr ||
          reblocked_accel == nullptr || canonical_token == nullptr ||
          view_token == nullptr || reblocked_token == nullptr ||
          !canonical->buffers[bank]->memory_accounted ||
          view->buffers[bank]->memory_accounted ||
          !same_physical_buffer(*canonical->buffers[bank],
                                *view->buffers[bank]) ||
          canonical_accel->buffer.buffer.id != view_accel->buffer.buffer.id ||
          canonical_token->backend_handle != view_token->backend_handle ||
          canonical_accel->buffer.buffer.id !=
              reblocked_accel->buffer.buffer.id ||
          canonical_token->backend_handle != reblocked_token->backend_handle ||
          reblocked_view->cache_regions[bank].count != 8u) {
        return 21;
      }
    }
  }
  footprint = 0u;
  std::uint64_t reblocked_footprint = 0u;
  return graph_pool_footprint(*metal_u32, u32_classes, footprint) &&
                 footprint == 392u &&
                 graph_pool_footprint(*metal_reblocked, reblocked_classes,
                                      reblocked_footprint) &&
                 reblocked_footprint == 196u
             ? 0
             : 22;
}

} // namespace rund_node_test_pipeline_residency
