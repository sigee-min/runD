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
  const auto total_scalar = total.scalar();
  const auto ordinal_scalar = ordinal.scalar();
  auto base = ordinal_scalar.map(
      "resident-base",
      capture([](auto index, auto tile) { return index * tile; }, width));
  auto count = total_scalar.combine(
      "resident-count", ordinal_scalar,
      capture(
          [](auto total_count, auto index, auto limit, auto tile, auto bound) {
            const auto begin = index * tile;
            const auto remaining =
                select(total_count > begin, total_count - begin, 0u);
            const auto active = select(remaining > tile, tile, remaining);
            return select((total_count > limit) || (index >= bound),
                          tile + 1u, active);
          },
          maximum, width, windows));
  const auto state = detail::StageRefAccess::state(total);
  auto slots = detail::StageRefAccess::make<std::uint32_t, stage::Exact>(
      state, detail::flow_index(state, detail::Type::U32, Tile));
  auto values = slots.combine(
      "resident-item", base,
      [](auto local, auto begin) { return begin + local; });
  auto items = detail::StageRefAccess::make<
      std::uint32_t, stage::Bounded<std::uint32_t>>(
      state, detail::StageRefAccess::id(values),
      detail::StageRefAccess::id(count));
  return ResidentWindow<Max, Tile>{std::move(items), std::move(base),
                                   ordinal_scalar};
}

} // namespace rund::compute
