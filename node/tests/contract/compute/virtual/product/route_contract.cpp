#include "backing.hpp"
#include "golden.hpp"
#include "model.hpp"
#include "route.hpp"

#include "../../../target/selection.hpp"

#include "src/compute/virtual/state.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>

namespace rund_node_test_virtual::product {
namespace {

using rund::compute::Backend;
using rund::compute::Device;
using rund::compute::Reason;
using rund::compute::Status;
using rund::compute::VirtualPipeline;

using ProductPipeline = VirtualPipeline<std::int32_t(std::int32_t)>;
using RequiredPipeline =
    VirtualPipeline<std::uint32_t(std::uint32_t, std::uint32_t)>;

void SeedUnsignedInput(const std::span<std::uint32_t> values) noexcept {
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = static_cast<std::uint32_t>(SeedValue(index));
  }
}

template <class Pipeline>
[[nodiscard]] int
observe_route(Pipeline &prepared, Device &device, const Backend backend,
              const RouteDemand demand, const RouteKind expected,
              ProductRouteObservation *const captured = nullptr) noexcept {
  const auto state =
      rund::compute::detail::VirtualPipelineAccess::state(prepared);
  if (state == nullptr || state->pipeline == nullptr ||
      state->pipeline->device == nullptr) {
    return 1;
  }

  std::optional<DeviceVsmBypassScope> bypass;
  if (demand == RouteDemand::Bypassed) {
    bypass.emplace(device);
    if (!bypass || !*bypass) {
      return 2;
    }
  }

  ProductRouteObservation observation{};
  ProductRouteScope route{device, observation, demand};
  if (!route) {
    return 3;
  }

  const Status status = prepared.run();
  ResolveProductRoute(observation, backend, static_cast<bool>(status));
  if (captured != nullptr) {
    *captured = observation;
  }
  if (!status || observation.kind != expected) {
    return 4;
  }
  if (demand == RouteDemand::Bypassed &&
      (observation.device_vsm_prepare_called ||
       observation.device_vsm_execute_called ||
       observation.device_vsm_execute_accepted)) {
    return 5;
  }
  if (demand == RouteDemand::Required &&
      (!observation.device_vsm_prepare_called ||
       !observation.device_vsm_execute_called ||
       !observation.device_vsm_execute_accepted)) {
    return 6;
  }
  return 0;
}

[[nodiscard]] int
run_bypassed_case(Device &device,
                  const rund::compute::Program<std::int32_t(std::int32_t)>
                      &program) noexcept {
  std::array<std::int32_t, LogicalElements> values{};
  SeedInput(values);
  auto input_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  if (!input_backing->seed(std::as_bytes(std::span{values}))) {
    return 10;
  }
  auto input = rund::compute::virtual_buffer<std::int32_t>(LogicalElements,
                                                           input_backing);
  auto output = rund::compute::virtual_buffer<std::int32_t>(LogicalElements,
                                                            output_backing);
  auto prepared =
      input && output
          ? rund::compute::virtual_pipeline(program, *input, *output,
                                            rund::compute::ResidencyConfig{})
          : rund::compute::Result<ProductPipeline>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    return 11;
  }
  ProductRouteObservation route_observation{};
  const int result =
      observe_route(*prepared, device, Backend::Vulkan, RouteDemand::Bypassed,
                    RouteKind::Persistent, &route_observation);
  if (result != 0) {
    return 20 + result;
  }
  std::array<std::int32_t, LogicalElements> observed{};
  if (!output_backing->observe(std::as_writable_bytes(std::span{observed}))) {
    return 30;
  }
  for (std::size_t index = 0u; index < observed.size(); ++index) {
    if (observed[index] != ProductValue(values[index])) {
      std::fprintf(
          stderr,
          "route-selection output mismatch demand=%u kind=%u index=%zu "
          "actual=%d expected=%d\n",
          static_cast<unsigned>(route_observation.demand),
          static_cast<unsigned>(route_observation.kind), index, observed[index],
          ProductValue(values[index]));
      return 31;
    }
  }
  return 0;
}

[[nodiscard]] int run_required_case(
    Device &device,
    const rund::compute::Program<std::uint32_t(std::uint32_t, std::uint32_t)>
        &program) noexcept {
  std::array<std::uint32_t, LogicalElements> first{};
  std::array<std::uint32_t, LogicalElements> second{};
  SeedUnsignedInput(first);
  SeedUnsignedInput(second);
  auto first_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto second_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  if (!first_backing->seed(std::as_bytes(std::span{first})) ||
      !second_backing->seed(std::as_bytes(std::span{second}))) {
    return 40;
  }
  auto first_buffer = rund::compute::virtual_buffer<std::uint32_t>(
      LogicalElements, first_backing);
  auto second_buffer = rund::compute::virtual_buffer<std::uint32_t>(
      LogicalElements, second_backing);
  auto output = rund::compute::virtual_buffer<std::uint32_t>(LogicalElements,
                                                             output_backing);
  auto prepared = first_buffer && second_buffer && output
                      ? rund::compute::virtual_pipeline(
                            program, *first_buffer, *second_buffer, *output,
                            rund::compute::ResidencyConfig{})
                      : rund::compute::Result<RequiredPipeline>::fail(
                            Reason::PipelineInvalid);
  if (!prepared) {
    return 41;
  }
  const int result = observe_route(*prepared, device, Backend::Vulkan,
                                   RouteDemand::Required, RouteKind::DeviceVsm);
  if (result != 0) {
    return 42 + result;
  }
  std::array<std::uint32_t, LogicalElements> observed{};
  if (!output_backing->observe(std::as_writable_bytes(std::span{observed}))) {
    return 50;
  }
  for (std::size_t index = 0u; index < observed.size(); ++index) {
    if (observed[index] !=
        static_cast<std::uint32_t>(first[index] + second[index])) {
      return 51;
    }
  }
  return 0;
}

} // namespace

int RunComputeVirtualRouteSelectionContract() {
  using namespace rund::compute;
  auto opened = open(rund::node::test_contract::target_for(Backend::Vulkan));
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }

  auto program =
      on(*opened)
          .map<std::int32_t>("virtual-route-selection", PageElements,
                             [](auto value) { return (value + 5) * 3; })
          .compile();
  if (!program) {
    return program.reason() == Reason::BackendUnsupported ? 0 : 2;
  }

  if (const int result = run_bypassed_case(*opened, *program); result != 0) {
    return 10 + result;
  }
  auto required_program =
      on(*opened)
          .input<std::uint32_t>(PageElements)
          .zip_input<std::uint32_t>(PageElements)
          .branch([](auto first, auto second) {
            return zip(first, second).map("add", [](auto left, auto right) {
              return left + right;
            });
          })
          .compile();
  if (!required_program) {
    return required_program.reason() == Reason::BackendUnsupported ? 0 : 30;
  }
  if (const int result = run_required_case(*opened, *required_program);
      result != 0) {
    return 20 + result;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product

int RunComputeVirtualRouteSelectionContract() {
  return rund_node_test_virtual::product::
      RunComputeVirtualRouteSelectionContract();
}
