#pragma once

#include <rund/compute/flow/stage.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute {

template <std::size_t Max, std::size_t Tile> class ResidentWindow final {
public:
  static constexpr std::size_t max_items = Max;
  static constexpr std::size_t tile_items = Tile;
  static constexpr std::size_t window_count = (Max + Tile - 1u) / Tile;

  [[nodiscard]] const StageRef<std::uint32_t,
                               stage::Bounded<std::uint32_t>> &
  items() const noexcept {
    return items_;
  }
  [[nodiscard]] StageRef<std::uint32_t, stage::Scalar> count() const noexcept {
    return items_.count();
  }
  [[nodiscard]] const StageRef<std::uint32_t, stage::Scalar> &
  base() const noexcept {
    return base_;
  }
  [[nodiscard]] const StageRef<std::uint32_t, stage::Scalar> &
  ordinal() const noexcept {
    return ordinal_;
  }

private:
  template <std::size_t M, std::size_t T>
  friend auto resident(const StageRef<std::uint32_t, stage::Exact> &,
                       const StageRef<std::uint32_t, stage::Exact> &);

  ResidentWindow(
      StageRef<std::uint32_t, stage::Bounded<std::uint32_t>> items,
      StageRef<std::uint32_t, stage::Scalar> base,
      StageRef<std::uint32_t, stage::Scalar> ordinal) noexcept
      : items_(std::move(items)), base_(std::move(base)),
        ordinal_(std::move(ordinal)) {}

  StageRef<std::uint32_t, stage::Bounded<std::uint32_t>> items_;
  StageRef<std::uint32_t, stage::Scalar> base_;
  StageRef<std::uint32_t, stage::Scalar> ordinal_;
};

template <std::size_t Max, std::size_t Tile>
[[nodiscard]] auto
resident(const StageRef<std::uint32_t, stage::Exact> &total,
         const StageRef<std::uint32_t, stage::Exact> &ordinal) {
  static_assert(Max != 0u, "compute resident maximum must be positive");
  static_assert(Tile != 0u && Tile <= Max,
                "compute resident tile must be within the maximum");
  static_assert(Max <= std::numeric_limits<std::uint32_t>::max(),
                "compute resident maximum must fit U32");
  static_assert(Tile < std::numeric_limits<std::uint32_t>::max(),
                "compute resident tile must leave one overflow witness");
  constexpr std::size_t Windows = (Max + Tile - 1u) / Tile;
  static_assert(Windows <= std::numeric_limits<std::uint32_t>::max(),
                "compute resident window count must fit U32");

  constexpr std::uint32_t maximum = static_cast<std::uint32_t>(Max);
  constexpr std::uint32_t width = static_cast<std::uint32_t>(Tile);
  constexpr std::uint32_t windows = static_cast<std::uint32_t>(Windows);
  (void)total.scalar();
  const auto ordinal_scalar = ordinal.scalar();
  auto items = total.expand(
      MaxItems{Tile}, ordinal_scalar,
      capture(
          [](auto count, auto index, auto limit, auto tile, auto bound) {
            const auto base = index * tile;
            const auto remaining = select(count > base, count - base, 0u);
            const auto active = select(remaining > tile, tile, remaining);
            return select((count > limit) || (index >= bound), tile + 1u,
                          active);
          },
          maximum, width, windows),
      capture([](auto, auto index, auto local, auto tile) {
        return index * tile + local;
      }, width));
  auto base = ordinal_scalar.map(
      "resident-base",
      capture([](auto index, auto tile) { return index * tile; }, width));
  return ResidentWindow<Max, Tile>{std::move(items), std::move(base),
                                   ordinal_scalar};
}

} // namespace rund::compute
