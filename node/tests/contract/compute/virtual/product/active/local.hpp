#pragma once

#include "../backing.hpp"
#include "../golden.hpp"
#include "../model.hpp"
#include "../route.hpp"

#include "../../../../target/selection.hpp"
#include "../../../allocation.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace rund_node_test_virtual::product::active {

using I32Program = rund::compute::Program<std::int32_t(std::int32_t)>;
using I32Pipeline =
    rund::compute::VirtualPipeline<std::int32_t(std::int32_t)>;
using I32Buffer = rund::compute::VirtualBuffer<std::int32_t>;

struct ActiveFixture final {
  rund::compute::Backend backend{rund::compute::Backend::Cpu};
  std::optional<rund::compute::Device> device{};
  std::optional<I32Program> program{};
  std::shared_ptr<MemoryVirtualBacking> input_backing{};
  std::shared_ptr<MemoryVirtualBacking> output_backing{};
  std::optional<I32Buffer> input{};
  std::optional<I32Buffer> output{};
  std::optional<I32Pipeline> prepared{};
  std::array<std::int32_t, LogicalElements> seeded{};
  std::array<std::int32_t, LogicalElements> golden{};
  rund::compute::PipelinePlan plan{};
  rund::compute::MemoryStats memory{};
  const void *input_identity{};
  const void *output_identity{};
};

[[nodiscard]] int InitializeActiveFixture(ActiveFixture &fixture,
                                           rund::compute::Backend backend);

[[nodiscard]] BackingFacts Delta(const BackingFacts after,
                                 const BackingFacts before) noexcept;
[[nodiscard]] bool SameCapacity(const rund::compute::MemoryStats &left,
                                const rund::compute::MemoryStats &right) noexcept;
[[nodiscard]] bool PrefixAndTail(std::span<const std::byte> observed,
                                 std::size_t active_count) noexcept;
[[nodiscard]] bool IsNativeMode(RouteKind mode) noexcept;

[[nodiscard]] int CheckBase(ActiveFixture &fixture);
[[nodiscard]] int CheckGrowth(ActiveFixture &fixture);
[[nodiscard]] int CheckActiveEvidence(ActiveFixture &fixture,
                                       std::size_t active_count,
                                       const BackingFacts &input_before,
                                       const BackingFacts &output_before,
                                       std::uint64_t allocations);
[[nodiscard]] int CheckCache(ActiveFixture &fixture);

} // namespace rund_node_test_virtual::product::active
