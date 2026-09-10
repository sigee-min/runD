#include "local.hpp"

namespace rund_node_test_virtual::product::reduce {
namespace {

[[nodiscard]] bool same_counter_shape(
    const rund::compute::MemoryCounter left,
    const rund::compute::MemoryCounter right) noexcept {
  return left.current == right.current && left.peak == right.peak &&
         left.budget == right.budget;
}

} // namespace

bool same_fixed_memory(const rund::compute::MemoryStats &left,
                       const rund::compute::MemoryStats &right) noexcept {
  return left.backend == right.backend && left.scope == right.scope &&
         same_counter_shape(left.host, right.host) &&
         same_counter_shape(left.frame, right.frame) &&
         same_counter_shape(left.tile, right.tile) &&
         same_counter_shape(left.resident, right.resident) &&
         same_counter_shape(left.staging, right.staging) &&
         same_counter_shape(left.device, right.device) &&
         same_counter_shape(left.transfer, right.transfer);
}

bool allocation_boundary_exact(const rund::compute::Backend backend,
                               const std::uint64_t observed_allocations)
    noexcept {
  // CPU closure allocation is completely visible to the process interceptor.
  // Accelerator runs also cross backend/framework service closures, so their
  // warm invariant is the producer-owned sample counter and fixed MemoryStats
  // checked at each call site below. Process-wide accelerator allocation is a
  // separately measured performance boundary, not a semantic run rejection.
  return backend != rund::compute::Backend::Cpu || observed_allocations == 0u;
}

bool observe_u64(MemoryVirtualBacking &backing, std::uint64_t &value) noexcept {
  std::array<std::byte, sizeof(value)> bytes{};
  if (!backing.observe(bytes)) {
    return false;
  }
  std::memcpy(&value, bytes.data(), sizeof(value));
  return true;
}

bool observe_i32(MemoryVirtualBacking &backing, std::int32_t &value) noexcept {
  std::array<std::byte, sizeof(value)> bytes{};
  if (!backing.observe(bytes)) {
    return false;
  }
  std::memcpy(&value, bytes.data(), sizeof(value));
  return true;
}

DelayedGraphBacking::DelayedGraphBacking(
    const std::span<const std::byte> bytes)
    : bytes_(bytes.begin(), bytes.end()) {}

std::uint64_t DelayedGraphBacking::size_bytes() const noexcept {
  return bytes_.size();
}

rund::compute::VirtualBackingTier
DelayedGraphBacking::tier() const noexcept {
  return rund::compute::VirtualBackingTier::Persistent;
}

std::uint32_t DelayedGraphBacking::max_parallel_reads() const noexcept {
  return 2u;
}

rund::compute::Status DelayedGraphBacking::read(
    const std::uint64_t offset,
    const std::span<std::byte> output) noexcept {
  if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
    return rund::compute::Status::fail(
        rund::compute::Reason::TransferInvalid);
  }
  const std::uint32_t active =
      active_reads_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
  std::uint32_t maximum = max_active_reads_.load(std::memory_order_relaxed);
  while (maximum < active && !max_active_reads_.compare_exchange_weak(
                                 maximum, active, std::memory_order_relaxed,
                                 std::memory_order_relaxed)) {
  }
  std::this_thread::sleep_for(std::chrono::milliseconds{10});
  std::memcpy(output.data(), bytes_.data() + offset, output.size());
  active_reads_.fetch_sub(1u, std::memory_order_release);
  return rund::compute::Status::success();
}

rund::compute::Status DelayedGraphBacking::write(
    std::uint64_t, std::span<const std::byte>) noexcept {
  return rund::compute::Status::fail(rund::compute::Reason::TransferInvalid);
}

std::uint32_t DelayedGraphBacking::max_active_reads() const noexcept {
  return max_active_reads_.load(std::memory_order_acquire);
}

int InitializeReduceFixture(ReduceFixture &fixture,
                            const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    if (device.reason() == Reason::AdapterUnavailable) {
      fixture.backend_unavailable = true;
      return 0;
    }
    return 1;
  }
  fixture.device.emplace(std::move(device).value());

  auto sum_flow = on(*fixture.device).input<std::uint64_t>(ReduceFrameElements);
  auto sum_program =
      std::move(sum_flow)
          .branch([](auto values) { return values.reduce(Reduce::Sum); })
          .compile();
  if (!sum_program || sum_program->graph().nodes.size() != 1u ||
      sum_program->graph().nodes.back().footprint.pattern !=
          graph::AccessPattern::Reduction) {
    return 2;
  }
  fixture.sum_program.emplace(std::move(sum_program).value());

  for (std::size_t index = 0u; index < fixture.values.size(); ++index) {
    fixture.values[index] = (index * 19u + 7u) % 101u;
    fixture.expected_sum += fixture.values[index];
  }
  fixture.input_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(fixture.values), ReduceFrameElements * sizeof(std::uint64_t));
  fixture.output_backing = std::make_shared<MemoryVirtualBacking>(
      sizeof(std::uint64_t), sizeof(std::uint64_t));
  if (!fixture.input_backing->seed(
          std::as_bytes(std::span{fixture.values}))) {
    return 3;
  }
  auto input =
      virtual_buffer<std::uint64_t>(ReduceElements, fixture.input_backing);
  auto output = virtual_buffer<std::uint64_t>(1u, fixture.output_backing);
  if (input) {
    fixture.input.emplace(std::move(input).value());
  }
  if (output) {
    fixture.output.emplace(std::move(output).value());
  }
  fixture.mapped_reduce_program = std::nullopt;
  auto mapped_reduce_program =
      on(*fixture.device)
          .map<std::uint64_t>("virtual-product-reduce-mapped",
                              ReduceFrameElements,
                              [](auto value) { return value + 1u; })
          .reduce(Reduce::Sum)
          .compile();
  if (mapped_reduce_program) {
    fixture.mapped_reduce_program.emplace(
        std::move(mapped_reduce_program).value());
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::reduce
