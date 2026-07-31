#include "model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/abi/graph.hpp>

#include "src/compute/expression/state.hpp"
#include "src/compute/flow/state.hpp"
#include "src/compute/graph/state.hpp"
#include "src/compute/map/build.hpp"
#include "src/compute/program/state.hpp"

#include "tests/contract/target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace compute_map_contract {
namespace {
enum class CarrierKind : unsigned char { CheckedOrdinal, BoundaryMask };

template <class Source, class Target, std::size_t Count>
[[nodiscard]] bool CheckCarrierMap(
    const std::shared_ptr<rund::compute::detail::DeviceState> &device,
    const rund::compute::Backend backend, const std::string_view name,
    const CarrierKind kind, const std::array<Source, Count> &input,
    const std::array<Target, Count> &expected) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  auto expression = make_expr();
  ExprRef value = detail::input(expression, type<Source>(), 0u);
  value = kind == CarrierKind::CheckedOrdinal
              ? checked_ordinal_expr(std::move(value), type<Target>())
              : boundary_mask_expr(std::move(value), type<Target>(),
                                   storage_format<Target>());
  auto flow =
      make_flow_on(device, type<Source>(), Count, {}, storage_format<Source>());
  flow_map(flow, name, std::move(value));
  auto program = compile_flow(flow);
  if (!program) {
    std::fprintf(stderr,
                 "carrier compile failed backend=%u name=%.*s reason=%.*s\n",
                 static_cast<unsigned>(backend), static_cast<int>(name.size()),
                 name.data(), static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  auto job = make_job(*program, std::span<const Source>{input});
  if (!job) {
    std::fprintf(
        stderr, "carrier resident failed backend=%u name=%.*s reason=%.*s\n",
        static_cast<unsigned>(backend), static_cast<int>(name.size()),
        name.data(), static_cast<int>(job.error().size()), job.error().data());
    return false;
  }
  const Status status = run_job(*job);
  if (!status) {
    std::fprintf(stderr,
                 "carrier run failed backend=%u name=%.*s reason=%.*s\n",
                 static_cast<unsigned>(backend), static_cast<int>(name.size()),
                 name.data(), static_cast<int>(status.error().size()),
                 status.error().data());
    return false;
  }
  auto output = read_job<Target>(*job);
  const Stats stats = job_stats(*job);
  const std::vector<Target> expected_values{expected.begin(), expected.end()};
  if (!output || *output != expected_values || stats.backend != backend) {
    std::fprintf(stderr, "carrier output failed backend=%u name=%.*s read=%d\n",
                 static_cast<unsigned>(backend), static_cast<int>(name.size()),
                 name.data(), output ? 1 : 0);
    return false;
  }
  return true;
}

template <class Source, class Target, std::size_t Count>
[[nodiscard]] bool CheckCarrierGraph(
    const rund::compute::Backend backend,
    const std::shared_ptr<rund::compute::detail::DeviceState> &device,
    const std::string_view name, const CarrierKind kind,
    const std::array<Source, Count> &input,
    const std::array<Target, Count> &expected) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  auto graph = make_graph(device, name, Count);
  const std::uint32_t graph_source = graph_input(graph, type<Source>());

  auto first_state = make_expr();
  ExprRef first_value = detail::input(first_state, type<Source>(), 0u);
  const ExprRef zero = constant(first_state, type<Source>(), 0u);
  first_value = binary(ExprOp::Add, std::move(first_value), zero);
  const std::array first_inputs{graph_source};
  const std::uint32_t intermediate =
      graph_map(graph, first_inputs, std::move(first_value), "carrier-source");

  auto second_state = make_expr();
  ExprRef second_value = detail::input(second_state, type<Source>(), 0u);
  second_value =
      kind == CarrierKind::CheckedOrdinal
          ? checked_ordinal_expr(std::move(second_value), type<Target>())
          : boundary_mask_expr(std::move(second_value), type<Target>(),
                               storage_format<Target>());
  const std::array second_inputs{intermediate};
  const std::uint32_t output_value =
      graph_map(graph, second_inputs, std::move(second_value), "carrier-write");
  graph_output(graph, output_value);
  const std::array input_types{type<Source>()};
  const std::array output_types{type<Target>()};
  auto program = compile_graph(graph, input_types, output_types);
  if (!program) {
    std::fprintf(stderr,
                 "carrier graph compile failed backend=%u name=%.*s "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), static_cast<int>(name.size()),
                 name.data(), static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  auto job = make_job(*program, std::span<const Source>{input});
  if (!job || !run_job(*job)) {
    std::fprintf(stderr, "carrier graph run failed backend=%u name=%.*s\n",
                 static_cast<unsigned>(backend), static_cast<int>(name.size()),
                 name.data());
    return false;
  }
  auto output = read_job<Target>(*job);
  const Stats stats = job_stats(*job);
  return output &&
         *output == std::vector<Target>{expected.begin(), expected.end()} &&
         stats.backend == backend &&
         (backend == Backend::Cpu || stats.dispatches == 1u);
}

template <class T, std::size_t Count>
[[nodiscard]] bool CheckSameDomainFlowIdentity(
    const rund::compute::Backend backend,
    const std::shared_ptr<rund::compute::detail::DeviceState> &device,
    const std::string_view name, const std::array<T, Count> &input) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  auto flow = make_flow_on(device, type<T>(), Count);
  auto expression = make_expr();
  ExprRef value = detail::input(expression, type<T>(), 0u);
  value = binary(ExprOp::Add, std::move(value),
                 constant(expression, type<T>(), 0u));
  flow_map(flow, name, std::move(value));
  if (flow_retype(flow, 2u, type<T>()) != 2u) {
    return false;
  }
  auto program = compile_flow(flow);
  if (!program) {
    return false;
  }
  auto job = make_job(*program, std::span<const T>{input});
  if (!job || !run_job(*job)) {
    return false;
  }
  auto output = read_job<T>(*job);
  const Stats stats = job_stats(*job);
  return output && *output == std::vector<T>{input.begin(), input.end()} &&
         stats.backend == backend;
}

} // namespace

[[nodiscard]] bool Carrier(const rund::compute::Backend backend,
                           rund::compute::graph::Fingerprint &envelope32,
                           rund::compute::graph::Fingerprint &envelope64) {
  auto opened = rund::compute::detail::open_target(
      rund::node::test_contract::target_for(backend));
  if (!opened) {
    std::fprintf(stderr, "carrier open failed backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(opened.error().size()),
                 opened.error().data());
    return false;
  }
  const auto &device = *opened;
  using Fixed16x16 = rund::compute::Fixed<16u, 16u>;
  using Fixed20x44 = rund::compute::Fixed<20u, 44u>;
  constexpr auto i32_min = std::numeric_limits<std::int32_t>::min();
  constexpr auto i32_max = std::numeric_limits<std::int32_t>::max();
  constexpr auto u32_max = std::numeric_limits<std::uint32_t>::max();
  constexpr auto i64_min = std::numeric_limits<std::int64_t>::min();
  constexpr auto i64_max = std::numeric_limits<std::int64_t>::max();
  constexpr auto u64_max = std::numeric_limits<std::uint64_t>::max();
  const std::array<std::int32_t, 5u> signed32{i32_min, -1, 0, 1, i32_max};
  const std::array<std::uint32_t, 5u> unsigned32{
      0u, 1u, static_cast<std::uint32_t>(i32_max),
      static_cast<std::uint32_t>(i32_max) + 1u, u32_max};
  const std::array<std::int64_t, 5u> signed64{i64_min, -1, 0, 1, i64_max};
  const std::array<std::uint64_t, 5u> unsigned64{
      0u, 1u, static_cast<std::uint64_t>(i64_max),
      static_cast<std::uint64_t>(i64_max) + 1u, u64_max};
  const std::array<std::uint32_t, 5u> checked_i32_to_u32{
      0u, 0u, 0u, 1u, static_cast<std::uint32_t>(i32_max)};
  const std::array<std::int32_t, 5u> checked_u32_to_i32{0, 1, i32_max, 0, 0};
  const std::array<std::uint64_t, 5u> checked_i64_to_u64{
      0u, 0u, 0u, 1u, static_cast<std::uint64_t>(i64_max)};
  const std::array<std::int64_t, 5u> checked_u64_to_i64{0, 1, i64_max, 0, 0};
  const std::array<std::uint32_t, 5u> boundary_i32_to_u32{1u, 1u, 0u, 1u, 1u};
  const std::array<std::int32_t, 5u> boundary_u32_to_i32{0, 1, 1, 1, 1};
  const std::array<std::uint64_t, 5u> boundary_i64_to_u64{1u, 1u, 0u, 1u, 1u};
  const std::array<std::int64_t, 5u> boundary_u64_to_i64{0, 1, 1, 1, 1};
  const std::array<Fixed16x16, 5u> boundary_i32_to_fixed{
      Fixed16x16::from_raw(1), Fixed16x16::from_raw(1), Fixed16x16::zero(),
      Fixed16x16::from_raw(1), Fixed16x16::from_raw(1)};
  const std::array<Fixed16x16, 5u> boundary_u32_to_fixed{
      Fixed16x16::zero(), Fixed16x16::from_raw(1), Fixed16x16::from_raw(1),
      Fixed16x16::from_raw(1), Fixed16x16::from_raw(1)};
  const std::array<Fixed20x44, 5u> boundary_i64_to_fixed{
      Fixed20x44::from_raw(1), Fixed20x44::from_raw(1), Fixed20x44::zero(),
      Fixed20x44::from_raw(1), Fixed20x44::from_raw(1)};
  const std::array<Fixed20x44, 5u> boundary_u64_to_fixed{
      Fixed20x44::zero(), Fixed20x44::from_raw(1), Fixed20x44::from_raw(1),
      Fixed20x44::from_raw(1), Fixed20x44::from_raw(1)};

  return CheckCarrierMap(device, backend, "checked-i32-u32",
                         CarrierKind::CheckedOrdinal, signed32,
                         checked_i32_to_u32) &&
         CheckCarrierMap(device, backend, "checked-u32-i32",
                         CarrierKind::CheckedOrdinal, unsigned32,
                         checked_u32_to_i32) &&
         CheckCarrierMap(device, backend, "checked-i64-u64",
                         CarrierKind::CheckedOrdinal, signed64,
                         checked_i64_to_u64) &&
         CheckCarrierMap(device, backend, "checked-u64-i64",
                         CarrierKind::CheckedOrdinal, unsigned64,
                         checked_u64_to_i64) &&
         CheckCarrierMap(device, backend, "boundary-i32-u32",
                         CarrierKind::BoundaryMask, signed32,
                         boundary_i32_to_u32) &&
         CheckCarrierMap(device, backend, "boundary-u32-i32",
                         CarrierKind::BoundaryMask, unsigned32,
                         boundary_u32_to_i32) &&
         CheckCarrierMap(device, backend, "boundary-i64-u64",
                         CarrierKind::BoundaryMask, signed64,
                         boundary_i64_to_u64) &&
         CheckCarrierMap(device, backend, "boundary-u64-i64",
                         CarrierKind::BoundaryMask, unsigned64,
                         boundary_u64_to_i64) &&
         CheckCarrierMap(device, backend, "boundary-i32-fixed",
                         CarrierKind::BoundaryMask, signed32,
                         boundary_i32_to_fixed) &&
         CheckCarrierMap(device, backend, "boundary-u32-fixed",
                         CarrierKind::BoundaryMask, unsigned32,
                         boundary_u32_to_fixed) &&
         CheckCarrierMap(device, backend, "boundary-i64-fixed",
                         CarrierKind::BoundaryMask, signed64,
                         boundary_i64_to_fixed) &&
         CheckCarrierMap(device, backend, "boundary-u64-fixed",
                         CarrierKind::BoundaryMask, unsigned64,
                         boundary_u64_to_fixed) &&
         CheckCarrierGraph(backend, device, "graph-checked-i32-u32",
                           CarrierKind::CheckedOrdinal, signed32,
                           checked_i32_to_u32) &&
         CheckCarrierGraph(backend, device, "graph-boundary-i64-fixed",
                           CarrierKind::BoundaryMask, signed64,
                           boundary_i64_to_fixed) &&
         CheckSameDomainFlowIdentity(backend, device, "identity-i32",
                                     signed32) &&
         CheckSameDomainFlowIdentity(backend, device, "identity-i64",
                                     signed64) &&
         Envelope(device, backend, envelope32, envelope64);
}

} // namespace compute_map_contract
