#include "../../../../../src/compute/buffer/state.hpp"
#include "local.hpp"

#include "../../../target/selection.hpp"
#include "src/compute/device/residency/pool.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/pipeline/residency/model.hpp"

#include <rund/compute/pipeline.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>

namespace rund_node_test_pipeline_residency {
namespace {

using rund::compute::Backend;
using rund::compute::Reason;
using rund::compute::detail::DeviceAccess;
using rund::compute::detail::DeviceState;
using rund::compute::detail::Type;
using rund::compute::detail::residency::FrameRegion;
using rund::compute::detail::residency::FrameRole;
using rund::compute::detail::residency::GraphResourceRole;
using rund::compute::detail::residency::PhysicalArena;
using rund::compute::detail::residency::Pool;
using rund::compute::detail::residency::PoolLayout;
using rund::compute::detail::residency::TiledGraphPhysicalClass;

constexpr std::uint32_t kFrames = 2u;

[[nodiscard]] PoolLayout ordinary_layout() noexcept {
  return PoolLayout{
      .input_type = Type::I32,
      .intermediate_type = Type::I32,
      .control_type = Type::U64,
      .output_type = Type::I32,
      .input_page_bytes = 64u,
      .intermediate_page_bytes = 64u,
      .control_page_bytes = 8u,
      .output_page_bytes = 8u,
      .frame_capacity = kFrames,
      .host_frame_capacity = kFrames,
      .host_output_frame_capacity = kFrames,
  };
}

[[nodiscard]] PoolLayout
graph_layout(const Backend backend, const Type type,
             const std::uint64_t input_page_bytes,
             const std::uint64_t output_page_bytes) noexcept {
  return PoolLayout{
      .input_type = type,
      .intermediate_type = type,
      .control_type = Type::U64,
      .output_type = type,
      .input_page_bytes = input_page_bytes,
      .intermediate_page_bytes = input_page_bytes,
      .control_page_bytes = 8u,
      .output_page_bytes = output_page_bytes,
      .frame_capacity = kFrames,
      .host_frame_capacity = kFrames,
      .host_output_frame_capacity = kFrames,
      .graph_host_service = backend != Backend::Cpu,
  };
}

[[nodiscard]] auto graph_classes(const Type type,
                                 const std::uint64_t input_page_bytes,
                                 const std::uint64_t output_page_bytes) {
  return std::array{
      TiledGraphPhysicalClass{.physical_id = 1u,
                              .role = GraphResourceRole::Input,
                              .type = type,
                              .page_bytes = input_page_bytes},
      TiledGraphPhysicalClass{.physical_id = 2u,
                              .role = GraphResourceRole::Intermediate,
                              .type = type,
                              .page_bytes = input_page_bytes},
      TiledGraphPhysicalClass{.physical_id = 3u,
                              .role = GraphResourceRole::Output,
                              .type = type,
                              .page_bytes = output_page_bytes},
  };
}

[[nodiscard]] bool ordinary_unchanged(
    const std::shared_ptr<PhysicalArena> &arena,
    const std::array<std::shared_ptr<rund::compute::detail::BufferState>,
                     Pool::BankCount> &buffers,
    const std::array<FrameRegion, Pool::BankCount> &regions,
    const std::array<FrameRegion, Pool::BankCount> &input_regions) noexcept {
  return arena != nullptr && arena->extent == 0u && arena->view == 0u &&
         arena->bank_bytes == 128u && arena->frame_capacity == kFrames &&
         arena->physical_class.type == Type::I32 &&
         arena->physical_class.page_bytes == 64u && arena->buffers == buffers &&
         arena->bank_regions == regions &&
         std::all_of(buffers.begin(), buffers.end(),
                     [](const auto &buffer) {
                       return buffer != nullptr && buffer->type == Type::I32 &&
                              buffer->bytes == 128u &&
                              buffer->physical_owner == nullptr &&
                              buffer->memory_accounted;
                     }) &&
         input_regions == std::array<FrameRegion, Pool::BankCount>{
                              FrameRegion{regions[0].tier, FrameRole::Input,
                                          regions[0].first, kFrames},
                              FrameRegion{regions[1].tier, FrameRole::Input,
                                          regions[1].first, kFrames}};
}

[[nodiscard]] int check_backend(const Backend backend) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  using namespace rund::compute::detail::residency;

  auto opened = open(rund::node::test_contract::target_for(backend, 2u));
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  const std::shared_ptr<DeviceState> device = DeviceAccess::state(*opened);
  if (device == nullptr || device->backend != backend ||
      device->residency == nullptr) {
    return 2;
  }

  const std::shared_ptr<Pool> ordinary =
      device->residency->acquire(device, ordinary_layout());
  if (ordinary == nullptr || !ordinary->graph_owners.empty() ||
      ordinary->input_arena == nullptr || ordinary->input_arena->extent != 0u ||
      ordinary->input_arena->view != 0u ||
      ordinary->input_arena->bank_bytes != 128u) {
    return 3;
  }
  const std::shared_ptr<PhysicalArena> ordinary_arena = ordinary->input_arena;
  const auto ordinary_buffers = ordinary_arena->buffers;
  const auto ordinary_regions = ordinary_arena->bank_regions;

  const auto classes = graph_classes(Type::U64, 128u, 8u);
  const std::shared_ptr<Pool> graph = device->residency->acquire(
      device, graph_layout(backend, Type::U64, 128u, 8u), classes);
  if (graph == nullptr || graph->graph_owners.size() != classes.size()) {
    return 4;
  }
  constexpr std::array<std::uint64_t, 3u> graph_bank_bytes{256u, 256u, 16u};
  for (std::size_t index = 0u; index < classes.size(); ++index) {
    const auto *const owner = graph->graph_owner(classes[index].physical_id);
    if (owner == nullptr || owner->arena == nullptr ||
        owner->arena == ordinary_arena || owner->arena->extent == 0u ||
        owner->arena->view == 0u ||
        owner->arena->bank_bytes != graph_bank_bytes[index] ||
        owner->frame_capacity != kFrames) {
      return 5;
    }
    for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
      if (owner->buffers[bank] == nullptr ||
          owner->buffers[bank] == ordinary_buffers[bank] ||
          same_physical_buffer(*owner->buffers[bank],
                               *ordinary_buffers[bank])) {
        return 6;
      }
    }
  }
  if (!ordinary_unchanged(ordinary_arena, ordinary_buffers, ordinary_regions,
                          ordinary->input_regions)) {
    return 7;
  }

  const auto typed_classes = graph_classes(Type::U32, 128u, 8u);
  const std::shared_ptr<Pool> typed = device->residency->acquire(
      device, graph_layout(backend, Type::U32, 128u, 8u), typed_classes);
  if (typed == nullptr) {
    return 8;
  }
  for (std::size_t index = 0u; index < classes.size(); ++index) {
    const auto *const base = graph->graph_owner(classes[index].physical_id);
    const auto *const view =
        typed->graph_owner(typed_classes[index].physical_id);
    if (base == nullptr || view == nullptr || view->arena != base->arena ||
        view->view != base->view || view->owns_regions ||
        view->arena->extent == 0u || view->arena->view == 0u) {
      return 9;
    }
    for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
      if (view->buffers[bank] == nullptr ||
          view->buffers[bank]->type != Type::U32 ||
          !same_physical_buffer(*view->buffers[bank], *base->buffers[bank])) {
        return 10;
      }
    }
  }

  const auto reblocked_classes = graph_classes(Type::U32, 64u, 4u);
  const std::shared_ptr<Pool> reblocked = device->residency->acquire(
      device, graph_layout(backend, Type::U32, 64u, 4u), reblocked_classes);
  if (reblocked == nullptr) {
    return 11;
  }
  for (std::size_t index = 0u; index < classes.size(); ++index) {
    const auto *const base = graph->graph_owner(classes[index].physical_id);
    const auto *const view =
        reblocked->graph_owner(reblocked_classes[index].physical_id);
    if (base == nullptr || view == nullptr || view->arena != base->arena ||
        view->view == 0u || view->view == base->view || !view->owns_regions ||
        view->frame_capacity != 4u || view->cache_regions[0].count != 4u ||
        view->bank_regions[0].count != kFrames) {
      return 12;
    }
    for (std::size_t bank = 0u; bank < Pool::BankCount; ++bank) {
      if (view->buffers[bank] == nullptr ||
          !same_physical_buffer(*view->buffers[bank], *base->buffers[bank])) {
        return 13;
      }
    }
  }
  return ordinary_unchanged(ordinary_arena, ordinary_buffers, ordinary_regions,
                            ordinary->input_regions)
             ? 0
             : 14;
}

} // namespace

int CheckPoolExtentAdmission() {
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (const int result = check_backend(backend); result != 0) {
      std::fprintf(stderr, "pool extent admission backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return result;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
