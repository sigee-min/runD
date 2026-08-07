#include <rund/compute.hpp>
#include <rund/compute/abi/graph.hpp>

#include "allocation.hpp"

#include "../../../src/compute/device/info.hpp"
#include "../../../src/compute/device/state.hpp"
#include "../../../src/compute/graph/state.hpp"
#include "../../../src/compute/memory/arena.hpp"
#include "../../../src/compute/type.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

namespace {

using rund::compute::detail::Type;
constexpr Type kUnknownType = static_cast<Type>(0xffu);

struct TypeProjectionCase final {
  Type type;
  std::size_t bytes;
  rund::kernel::ComputeScalar scalar;
  rund::kernel::ComputeDomain domain;
  bool fixed;
  bool valid;
};

constexpr std::array kTypeProjectionCases{
    TypeProjectionCase{Type::I32, 4u, rund::kernel::ComputeScalar::Lane32,
                       rund::kernel::ComputeDomain::I32, false, true},
    TypeProjectionCase{Type::U32, 4u, rund::kernel::ComputeScalar::Lane32,
                       rund::kernel::ComputeDomain::U32, false, true},
    TypeProjectionCase{Type::I64, 8u, rund::kernel::ComputeScalar::Lane64,
                       rund::kernel::ComputeDomain::I64, false, true},
    TypeProjectionCase{Type::U64, 8u, rund::kernel::ComputeScalar::Lane64,
                       rund::kernel::ComputeDomain::U64, false, true},
    TypeProjectionCase{Type::FixedLane32, 4u,
                       rund::kernel::ComputeScalar::Lane32,
                       rund::kernel::ComputeDomain::Fixed, true, true},
    TypeProjectionCase{Type::FixedLane64, 8u,
                       rund::kernel::ComputeScalar::Lane64,
                       rund::kernel::ComputeDomain::Fixed, true, true},
    TypeProjectionCase{
        kUnknownType, 0u, static_cast<rund::kernel::ComputeScalar>(0u),
        static_cast<rund::kernel::ComputeDomain>(0u), false, false},
};

static_assert([] {
  for (const TypeProjectionCase entry : kTypeProjectionCases) {
    if (rund::compute::detail::type_bytes(entry.type) != entry.bytes ||
        rund::compute::detail::type_scalar(entry.type) != entry.scalar ||
        rund::compute::detail::type_domain(entry.type) != entry.domain ||
        rund::compute::detail::type_fixed(entry.type) != entry.fixed ||
        rund::compute::detail::valid_type(entry.type) != entry.valid) {
      return false;
    }
  }
  return true;
}());

[[nodiscard]] std::string
StrategyName(const rund::kernel::CpuSimdStrategy strategy) {
  switch (strategy) {
  case rund::kernel::CpuSimdStrategy::Scalar:
    return "scalar";
  case rund::kernel::CpuSimdStrategy::Sse2:
    return "sse2";
  case rund::kernel::CpuSimdStrategy::Avx2:
    return "avx2";
  case rund::kernel::CpuSimdStrategy::Avx512:
    return "avx512";
  case rund::kernel::CpuSimdStrategy::Neon:
    return "neon";
  }
  return {};
}

[[nodiscard]] bool
CpuInfoMatches(const rund::compute::detail::DeviceState &state,
               const rund::compute::DeviceInfo &info) {
  const auto *const cpu = rund::compute::detail::cpu_device(state);
  if (cpu == nullptr) {
    return false;
  }
  const std::string name = "cpu/" + StrategyName(cpu->caps.strategy);
  const std::string details =
      "workers=" + std::to_string(cpu->workers.requested_worker_width) +
      ";lane:bytes=" + std::to_string(cpu->caps.lane_bytes) +
      ";fixed:lane32:lanes=" + std::to_string(cpu->caps.fixed_lane32_lanes) +
      ";fixed:lane64:lanes=" + std::to_string(cpu->caps.fixed_lane64_lanes);
  return info.backend == rund::compute::Backend::Cpu && info.name == name &&
         info.driver == "rund/built-in-pool" &&
         info.driver_details == details && info.storage_alignment == 64u &&
         info.storage_bytes == 64u * 1024u * 1024u;
}

[[nodiscard]] bool AccelInfoIsOwning() {
  using namespace rund::compute;
  using namespace rund::compute::detail;

  auto state = std::make_shared<DeviceState>();
  state->backend = Backend::Metal;
  state->storage = AccelDeviceState{};
  auto *const accel = accel_device(*state);
  if (accel == nullptr) {
    return false;
  }

  accel->pick.backend_info = rund::AccelBackendInfo{
      .device_name = "device-only",
  };
  const auto device_only = snapshot_device_info(state);
  if (!device_only || device_only->name != "device-only" ||
      !device_only->driver.empty() || !device_only->driver_details.empty()) {
    return false;
  }

  accel->pick.backend_info = rund::AccelBackendInfo{
      .device_name = "device-name",
      .driver_name = "driver-name",
      .driver_info = "driver-details",
      .storage_alignment = 16u,
      .storage_bytes = 4096u,
  };
  auto snapshot = snapshot_device_info(state);
  if (!snapshot) {
    return false;
  }
  accel->pick.backend_info = {};
  state.reset();
  return snapshot->backend == Backend::Metal &&
         snapshot->name == "device-name" && snapshot->driver == "driver-name" &&
         snapshot->driver_details == "driver-details" &&
         snapshot->storage_alignment == 16u && snapshot->storage_bytes == 4096u;
}

[[nodiscard]] bool ArenaUsesStorageLimit() {
  using namespace rund::compute;
  using namespace rund::compute::detail;

  DeviceState state{};
  state.backend = Backend::Vulkan;
  state.storage = AccelDeviceState{};
  auto *const accel = accel_device(state);
  if (accel == nullptr) {
    return false;
  }
  accel->pick.caps.device_bytes = 20ull << 30u;
  accel->pick.backend_info.storage_bytes = (4ull << 30u) - 1u;
  return memory::arena_bytes(state) == ((4ull << 30u) - 256u);
}

[[nodiscard]] bool RejectsUnknownPrimitiveOptions(
    const std::shared_ptr<rund::compute::detail::DeviceState> &device) {
  using namespace rund::compute::detail;
  struct Case final {
    Primitive primitive;
    PrimitiveOptions options;
    std::string_view reason;
  };
  constexpr std::array cases{
      Case{Primitive::SegmentedScan,
           {.mode = 0xffu},
           "compute_scan_op_unsupported"},
      Case{Primitive::SegmentedReduce,
           {.mode = 0xffu},
           "compute_reduce_op_unsupported"},
      Case{Primitive::Reduce, {.mode = 0xffu}, "compute_reduce_op_unsupported"},
      Case{Primitive::Stencil,
           {.mode = 0xffu},
           "compute_stencil_op_unsupported"},
      Case{Primitive::Transform,
           {.mode = 0xffu},
           "compute_transform_direction_unsupported"},
      Case{Primitive::Matrix, {.mode = 0xffu}, "compute_matrix_op_unsupported"},
      Case{Primitive::Factor, {.mode = 0xffu}, "compute_factor_op_unsupported"},
      Case{Primitive::Solve, {.mode = 0xffu}, "compute_factor_op_unsupported"},
      Case{Primitive::Spectrum,
           {.mode = 0xffu},
           "compute_spectrum_op_unsupported"},
      Case{Primitive::Spectrum,
           {.fourth = 0xffu,
            .mode = static_cast<std::uint32_t>(rund::compute::SpectrumOp::Svd)},
           "compute_spectrum_vectors_unsupported"},
      Case{static_cast<Primitive>(0xffu), {}, "compute_primitive_unsupported"},
  };
  for (const Case &entry : cases) {
    auto graph = make_graph(device, "unknown-primitive-option", 1u);
    const std::uint32_t input = graph_input(graph, Type::I32);
    const std::array args{
        GraphArg{.value = input, .type = Type::I32, .count = 1u}};
    const GraphOut output =
        graph_primitive(graph, entry.primitive, args, entry.options);
    if (output.value != 0u || graph == nullptr || graph->status ||
        graph->status.error() != entry.reason) {
      return false;
    }
  }
  auto scan = make_graph(device, "unknown-scan-option", 1u);
  const std::uint32_t scan_input = graph_input(scan, Type::I32);
  if (graph_scan(scan, scan_input, static_cast<rund::compute::Scan>(0xffu)) !=
          0u ||
      scan == nullptr || scan->status ||
      scan->status.error() != "compute_scan_op_unsupported") {
    return false;
  }
  return true;
}

} // namespace

int RunComputeSdkContract() {
  auto device = rund::compute::open(rund::compute::Target::cpu());
  const auto selected_backend =
      device ? device->backend()
             : rund::compute::Result<rund::compute::Backend>::fail(
                   device.reason());
  if (!device || !selected_backend ||
      *selected_backend != rund::compute::Backend::Cpu) {
    return 1;
  }

  auto empty = device.value().buffer<std::int32_t>(0);
  if (!empty || empty->size() != 0u) {
    return 2;
  }

  auto target = device.value().buffer<std::int32_t>(4);
  if (!target || target->size() != 4) {
    return 3;
  }

  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto source = device.value().upload(std::span<const std::int32_t>{input});
  if (!source || source->size() != input.size()) {
    return 4;
  }

  auto cpu_state = rund::compute::detail::open_cpu(0u);
  if (!cpu_state ||
      rund::compute::detail::cpu_device(*cpu_state.value()) == nullptr ||
      rund::compute::detail::accel_device(*cpu_state.value()) != nullptr) {
    return 7;
  }
  const auto cpu_info =
      rund::compute::detail::snapshot_device_info(cpu_state.value());
  const auto public_info = device->info();
  if (!cpu_info || !public_info || *cpu_info != *public_info ||
      !CpuInfoMatches(*cpu_state.value(), *cpu_info)) {
    return 11;
  }
  node_compute_allocation::FailNext();
  const auto capacity = device->info();
  if (capacity || capacity.code() != rund::compute::Code::Capacity ||
      capacity.error() != "compute_device_info_capacity") {
    return 12;
  }
  if (!AccelInfoIsOwning()) {
    return 13;
  }
  if (!ArenaUsesStorageLimit()) {
    return 15;
  }
  if (!RejectsUnknownPrimitiveOptions(cpu_state.value())) {
    return 10;
  }
  auto cpu_storage = rund::compute::detail::make_buffer(
      cpu_state.value(), rund::compute::detail::Type::I32, 4u);
  if (!cpu_storage ||
      rund::compute::detail::accel_buffer(*cpu_storage.value()) != nullptr) {
    return 8;
  }
  const auto *const host =
      rund::compute::detail::cpu_buffer(*cpu_storage.value());
  if (host == nullptr ||
      reinterpret_cast<std::uintptr_t>(host->data.get()) % 64u != 0u) {
    return 9;
  }

  constexpr std::array explicit_targets{
      rund::compute::Target::metal(),
      rund::compute::Target::vulkan(),
  };
  for (const rund::compute::Target target : explicit_targets) {
    const auto requested = target.backend();
    auto selected = rund::compute::open(target);
    if (!selected) {
      if (selected.reason() != rund::compute::Reason::AdapterUnavailable) {
        return 6;
      }
      continue;
    }
    auto info = selected
                    ? selected->info()
                    : rund::compute::Result<rund::compute::DeviceInfo>::fail(
                          selected.reason());
    const auto backend =
        selected ? selected->backend()
                 : rund::compute::Result<rund::compute::Backend>::fail(
                       selected.reason());
    if (!selected || !backend || !info || *backend != requested ||
        info->backend != requested || info->storage_alignment == 0u ||
        info->storage_bytes == 0u) {
      const std::string_view error = selected.error();
      const char *const error_text = error.empty() ? "" : error.data();
      std::fprintf(stderr, "compute.sdk backend=%u code=%u error=%.*s\n",
                   static_cast<unsigned>(requested),
                   static_cast<unsigned>(selected.code()),
                   static_cast<int>(error.size()), error_text);
      return 6;
    }
    if (requested == rund::compute::Backend::Metal &&
        (info->name.empty() || info->driver != "Metal" ||
         !info->driver_details.empty())) {
      return 14;
    }
  }
  return 0;
}
