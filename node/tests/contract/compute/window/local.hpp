#pragma once

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::test_contract::window {

using rund::compute::Backend;
using rund::compute::Device;
using rund::compute::Fixed;
using rund::compute::MemoryBudget;
using rund::compute::MemoryCounter;
using rund::compute::MemoryStats;
using rund::compute::Reason;
using rund::compute::graph::Fingerprint;

[[nodiscard]] constexpr std::size_t
CeilDiv(const std::size_t value, const std::size_t divisor) noexcept {
  return value / divisor + (value % divisor == 0u ? 0u : 1u);
}

[[nodiscard]] inline std::uint64_t Hash(const void *const data,
                                        const std::size_t bytes) noexcept {
  constexpr std::uint64_t offset = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  const auto *const values = static_cast<const std::uint8_t *>(data);
  std::uint64_t hash = offset;
  for (std::size_t index = 0u; index < bytes; ++index) {
    hash ^= values[index];
    hash *= prime;
  }
  return hash;
}

[[nodiscard]] constexpr bool Same(const MemoryCounter left,
                                  const MemoryCounter right) noexcept {
  return left.current == right.current && left.peak == right.peak &&
         left.cumulative == right.cumulative && left.reused == right.reused &&
         left.budget == right.budget;
}

[[nodiscard]] constexpr bool Same(const MemoryStats &left,
                                  const MemoryStats &right) noexcept {
  return left.backend == right.backend && left.scope == right.scope &&
         Same(left.host, right.host) && Same(left.frame, right.frame) &&
         Same(left.tile, right.tile) && Same(left.resident, right.resident) &&
         Same(left.staging, right.staging) && Same(left.device, right.device) &&
         Same(left.transfer, right.transfer);
}

[[nodiscard]] constexpr bool NoAllocation(const MemoryCounter before,
                                          const MemoryCounter after) noexcept {
  return after.current <= before.current && after.peak == before.peak &&
         after.cumulative == before.cumulative &&
         after.reused == before.reused && after.budget == before.budget;
}

[[nodiscard]] constexpr bool NoAllocation(const MemoryStats &before,
                                          const MemoryStats &after) noexcept {
  return before.backend == after.backend && before.scope == after.scope &&
         NoAllocation(before.host, after.host) &&
         NoAllocation(before.frame, after.frame) &&
         NoAllocation(before.tile, after.tile) &&
         NoAllocation(before.resident, after.resident) &&
         NoAllocation(before.staging, after.staging) &&
         NoAllocation(before.device, after.device) &&
         NoAllocation(before.transfer, after.transfer);
}

template <class T, std::size_t N> [[nodiscard]] auto Values() {
  std::array<T, N> values{};
  for (std::size_t index = 0u; index < N; ++index) {
    values[index] = T::from_raw(static_cast<typename T::Raw>(index + 1u));
  }
  return values;
}

template <class T, std::size_t N> [[nodiscard]] auto Filled(const T value) {
  std::array<T, N> values{};
  values.fill(value);
  return values;
}

template <class T, std::size_t Maximum, std::size_t Tile>
[[nodiscard]] auto Fold(Device &device) {
  using namespace rund::compute;
  return on(device)
      .template input<T>(1u)
      .template zip_input<T>(Maximum)
      .template zip_input<std::uint32_t>(1u)
      .template zip_input<std::uint32_t>(1u)
      .branch([](auto carried, auto invariant, auto total, auto ordinal) {
        auto window = resident<Maximum, Tile>(total, ordinal);
        const auto base = window.base();
        const auto count = window.count();
        const auto canonical = window.ordinal();
        (void)base;
        (void)count;
        (void)canonical;
        auto tile_sum = invariant.gather(window.items()).reduce(Reduce::Sum);
        return carried.combine(
            "resident-window-fold", tile_sum,
            [](auto value, auto sum) { return quantize<T>(value + sum); });
      })
      .compile();
}

constexpr std::size_t kMatrixMaximum = 8u;
constexpr std::size_t kMatrixTile = 3u;
constexpr std::size_t kMatrixWindows = CeilDiv(kMatrixMaximum, kMatrixTile);
constexpr std::size_t kMatrixWidth = 4u;
constexpr std::size_t kMatrixBacking = 9u;
constexpr std::uint32_t kCountLeft = 0x13579BDFu;
constexpr std::uint32_t kCountRight = 0x2468ACE0u;

static_assert(kMatrixWindows == 3u);
static_assert(
    rund::compute::ResidentWindow<kMatrixMaximum, kMatrixTile>::window_count ==
    kMatrixWindows);

[[nodiscard]] inline auto MatrixBody(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kMatrixWidth)
      .zip_input<std::uint32_t>(kMatrixWidth)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto first, auto second, auto witness, auto fail_at,
                 auto total, auto ordinal) {
        auto window = resident<kMatrixMaximum, kMatrixTile>(total, ordinal);
        auto fault_index =
            ordinal.combine("resident-window-fault-index", fail_at,
                            [](auto current, auto selected) {
                              return select(current == selected, 1u, 0u);
                            });
        auto checked = witness.gather(fault_index).scalar();
        auto sum = window.items().reduce(Reduce::Sum);
        auto next_first =
            first
                .combine("resident-window-matrix-first", sum,
                         [](auto value, auto increment) {
                           return value + increment;
                         })
                .combine("resident-window-matrix-first-check", checked,
                         [](auto value, auto check) { return value + check; });
        auto next_second =
            second
                .combine("resident-window-matrix-second", sum,
                         [](auto value, auto increment) {
                           return value + increment + increment;
                         })
                .combine("resident-window-matrix-second-check", checked,
                         [](auto value, auto check) { return value + check; });
        return outputs(next_first, next_second);
      })
      .compile();
}

template <class Program>
[[nodiscard]] bool SetControl(Device &device, const Program &copy,
                              rund::compute::Buffer<std::uint32_t> &target,
                              const std::uint32_t value) {
  const std::array<std::uint32_t, 1u> values{value};
  auto source =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{values});
  return source && copy.run(*source, target);
}

[[nodiscard]] constexpr std::array<std::uint32_t, 3u>
PackedCount(const std::uint32_t value) {
  return {kCountLeft, value, kCountRight};
}

template <class Program>
[[nodiscard]] bool SetCount(Device &device, const Program &copy,
                            rund::compute::Buffer<std::uint32_t> &target,
                            const std::uint32_t value) {
  const auto values = PackedCount(value);
  auto source =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{values});
  return source && copy.run(*source, target);
}

template <class Program, std::size_t N>
[[nodiscard]] bool Observe(const Program &copy,
                           const rund::compute::Buffer<std::uint32_t> &source,
                           rund::compute::Buffer<std::uint32_t> &scratch,
                           std::array<std::uint32_t, N> &output) {
  auto run = copy.run(source, scratch);
  return run && run->read(scratch, std::span<std::uint32_t>{output});
}

template <std::size_t N>
[[nodiscard]] constexpr std::array<std::uint32_t, N>
Projected(const std::array<std::uint32_t, N> &backing,
          const std::array<std::uint32_t, kMatrixWidth> &values,
          const std::size_t offset, const std::size_t stride) {
  auto result = backing;
  for (std::size_t index = 0u; index < values.size(); ++index) {
    result[offset + index * stride] = values[index];
  }
  return result;
}

[[nodiscard]] constexpr std::uint32_t PrefixSum(const std::uint32_t count) {
  return count * (count - (count == 0u ? 0u : 1u)) / 2u;
}

struct MatrixIdentity final {
  Fingerprint body{};
  Fingerprint pipeline{};
  std::uint64_t output{};
};

struct WindowOutputIdentity final {
  Fingerprint seed{};
  Fingerprint fold{};
  Fingerprint pipeline{};
  std::uint64_t output{};
  Fingerprint high_index_fold{};
  Fingerprint high_index_pipeline{};
  std::uint64_t high_index_output{};
  Fingerprint scatter_seed{};
  Fingerprint scatter_fold{};
  Fingerprint scatter_pipeline{};
  std::uint64_t scatter_output{};
};

struct Identity final {
  Fingerprint body32{};
  Fingerprint pipeline32{};
  std::uint64_t output32{};
  Fingerprint body64{};
  Fingerprint pipeline64{};
  std::uint64_t output64{};
  MatrixIdentity matrix{};
  MatrixIdentity terminal{};
  WindowOutputIdentity window_output{};
};

[[nodiscard]] int CheckPlan(Device &device);
[[nodiscard]] int CheckParity32(Device &device, Backend backend,
                                Identity &identity);
[[nodiscard]] int CheckParity64(Device &device, Backend backend,
                                Identity &identity);
[[nodiscard]] int CheckOverflow32(Device &device, Backend backend);
[[nodiscard]] int CheckOverflow64(Device &device, Backend backend);
[[nodiscard]] int CheckWindowMatrix(Device &device, Backend backend,
                                    MatrixIdentity &identity);
[[nodiscard]] int CheckWindowChain(Device &device, Backend backend);
[[nodiscard]] int CheckWindowFreeze(Device &device, Backend backend);
[[nodiscard]] int CheckWindowTerminal(Device &device, Backend backend,
                                      MatrixIdentity &identity);
[[nodiscard]] int CheckWindowOutput(Device &device, Backend backend,
                                    WindowOutputIdentity &identity);

} // namespace rund::node::test_contract::window
