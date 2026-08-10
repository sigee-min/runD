#include "local.hpp"

#include "backing.hpp"
#include "model.hpp"

#include <rund/compute/virtual.hpp>

#include <array>
#include <concepts>
#include <memory>
#include <type_traits>

namespace rund_node_test_virtual::product {
namespace {

template <class T>
concept RunsVirtualPipeline = requires(T &value) {
  { value.run() } -> std::same_as<rund::compute::Status>;
  { value.run(std::uint64_t{0u}) } -> std::same_as<rund::compute::Status>;
};

static_assert(std::is_abstract_v<rund::compute::VirtualBacking>);
static_assert(std::is_copy_constructible_v<rund::compute::VirtualBuffer<int>>);
static_assert(
    !std::is_copy_constructible_v<rund::compute::VirtualPipeline<int(int)>>);
static_assert(std::is_nothrow_move_constructible_v<
              rund::compute::VirtualPipeline<int(int)>>);
static_assert(std::is_trivially_copyable_v<rund::compute::ResidencyConfig>);
static_assert(RunsVirtualPipeline<rund::compute::VirtualPipeline<int(int)>>);

} // namespace

int CheckProductSurface() {
  auto backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto short_backing = std::make_shared<MemoryVirtualBacking>(LogicalBytes - 1u,
                                                              ElementPageBytes);
  if (!backing->valid() || !backing->tail_poisoned()) {
    return 1;
  }
  auto valid =
      rund::compute::virtual_buffer<std::int32_t>(LogicalElements, backing);
  auto missing =
      rund::compute::virtual_buffer<std::int32_t>(LogicalElements, nullptr);
  auto mismatch = rund::compute::virtual_buffer<std::int32_t>(LogicalElements,
                                                              short_backing);
  if (!valid || !*valid || valid->size() != LogicalElements || missing ||
      missing.reason() != rund::compute::Reason::BufferCapacity || mismatch ||
      mismatch.reason() != rund::compute::Reason::ShapeMismatch) {
    return 2;
  }
  std::array<std::byte, 1u> byte{};
  const auto read_past = backing->read(LogicalBytes, byte);
  const auto write_past = backing->write(LogicalBytes, byte);
  return !read_past &&
                 read_past.reason() == rund::compute::Reason::ShapeMismatch &&
                 !write_past &&
                 write_past.reason() == rund::compute::Reason::ShapeMismatch &&
                 backing->facts().read_count == 0u &&
                 backing->facts().write_count == 0u && backing->tail_poisoned()
             ? 0
             : 3;
}

} // namespace rund_node_test_virtual::product
