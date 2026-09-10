#pragma once

#include "local.hpp"

#include "../model.hpp"

#include "../../../../../../src/accel/range_aggregate/model/traits.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

namespace rund_node_collective_modes::bounded {

inline constexpr std::size_t kResidentWindowRadius =
    rund::node::accel::detail::kRangeWidths.back() + 1u;
inline constexpr std::size_t kResidentWindowCapacity =
    2u * kResidentWindowRadius + 1u;
inline constexpr std::size_t kResidentWindowSize = kResidentWindowCapacity;
inline constexpr std::size_t kResidentWindowSmallCount = 5u;

template <class T, bool = rund::compute::detail::FixedValue<T>>
struct ResidentBitsOf final {
  using Type = T;
};

template <class T> struct ResidentBitsOf<T, true> final {
  using Type = typename T::Raw;
};

template <class T> using ResidentBits = typename ResidentBitsOf<T>::Type;

template <class T>
using ResidentUnsigned = std::make_unsigned_t<ResidentBits<T>>;

template <class T>
[[nodiscard]] constexpr ResidentUnsigned<T> ResidentRaw(const T value) {
  using Bits = ResidentBits<T>;
  using Unsigned = ResidentUnsigned<T>;
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return static_cast<Unsigned>(value.raw());
  } else if constexpr (std::is_unsigned_v<T>) {
    return value;
  } else {
    return std::bit_cast<Unsigned>(static_cast<Bits>(value));
  }
}

template <class T>
[[nodiscard]] constexpr T ResidentFromRaw(const ResidentUnsigned<T> value) {
  using Bits = ResidentBits<T>;
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return T::from_raw(std::bit_cast<Bits>(value));
  } else if constexpr (std::is_unsigned_v<T>) {
    return value;
  } else {
    return std::bit_cast<T>(value);
  }
}

template <class T>
[[nodiscard]] constexpr T ResidentWrapAdd(const T left, const T right) {
  return ResidentFromRaw<T>(
      static_cast<ResidentUnsigned<T>>(ResidentRaw(left) + ResidentRaw(right)));
}

template <class T>
[[nodiscard]] constexpr T ResidentPattern(const std::size_t index) {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    using Raw = typename T::Raw;
    constexpr std::array<Raw, 8u> pattern{std::numeric_limits<Raw>::max(),
                                          Raw{2},
                                          std::numeric_limits<Raw>::min(),
                                          Raw{-2},
                                          Raw{17},
                                          Raw{-11},
                                          Raw{5},
                                          Raw{-7}};
    return T::from_raw(pattern[index % pattern.size()]);
  } else if constexpr (std::is_unsigned_v<T>) {
    constexpr std::array<T, 8u> pattern{
        std::numeric_limits<T>::max(),
        T{2},
        T{0},
        static_cast<T>(std::numeric_limits<T>::max() - T{1}),
        T{17},
        T{11},
        T{5},
        T{7}};
    return pattern[index % pattern.size()];
  } else {
    constexpr std::array<T, 8u> pattern{std::numeric_limits<T>::max(),
                                        T{2},
                                        std::numeric_limits<T>::min(),
                                        T{-2},
                                        T{17},
                                        T{-11},
                                        T{5},
                                        T{-7}};
    return pattern[index % pattern.size()];
  }
}

template <class T>
[[nodiscard]] constexpr T ResidentPoison(const std::size_t index) {
  if constexpr (rund::compute::detail::FixedValue<T>) {
    return index % 2u == 0u ? T::max() : T::min();
  } else if constexpr (std::is_unsigned_v<T>) {
    return index % 2u == 0u
               ? std::numeric_limits<T>::max()
               : static_cast<T>(std::numeric_limits<T>::max() - T{1});
  } else {
    return index % 2u == 0u ? std::numeric_limits<T>::max()
                            : std::numeric_limits<T>::min();
  }
}

enum class ResidentWindowOp : std::uint8_t { Sum, Min, Max };

template <class T>
[[nodiscard]] std::vector<T>
ResidentWindowOracle(const std::vector<T> &input, const std::size_t count,
                     const ResidentWindowOp operation, const bool clip) {
  std::vector<T> output;
  output.reserve(count);
  for (std::size_t center = 0u; center < count; ++center) {
    T aggregate = operation == ResidentWindowOp::Sum
                      ? Zero<T>()
                      : (operation == ResidentWindowOp::Min ? Maximum<T>()
                                                            : Minimum<T>());
    for (std::size_t slot = 0u; slot < kResidentWindowSize; ++slot) {
      const std::int64_t logical =
          static_cast<std::int64_t>(center) + static_cast<std::int64_t>(slot) -
          static_cast<std::int64_t>(kResidentWindowRadius);
      std::size_t source = 0u;
      if (logical < 0) {
        if (clip) {
          continue;
        }
      } else if (static_cast<std::uint64_t>(logical) >= count) {
        if (clip) {
          continue;
        }
        source = count - 1u;
      } else {
        source = static_cast<std::size_t>(logical);
      }
      const T value = input[source];
      if (operation == ResidentWindowOp::Sum) {
        aggregate = ResidentWrapAdd(aggregate, value);
      } else if (operation == ResidentWindowOp::Min) {
        aggregate = std::min(aggregate, value);
      } else {
        aggregate = std::max(aggregate, value);
      }
    }
    output.push_back(aggregate);
  }
  return output;
}

template <class T>
[[nodiscard]] std::array<std::vector<T>, 6u>
ResidentWindowOracle(const std::vector<T> &input, const std::size_t count) {
  return {ResidentWindowOracle(input, count, ResidentWindowOp::Sum, false),
          ResidentWindowOracle(input, count, ResidentWindowOp::Min, false),
          ResidentWindowOracle(input, count, ResidentWindowOp::Max, false),
          ResidentWindowOracle(input, count, ResidentWindowOp::Sum, true),
          ResidentWindowOracle(input, count, ResidentWindowOp::Min, true),
          ResidentWindowOracle(input, count, ResidentWindowOp::Max, true)};
}

template <class Output, class T>
[[nodiscard]] bool
ResidentOutputMatches(const Output &output,
                      const std::array<std::vector<T>, 6u> &expected) {
  return output && std::get<0>(*output) == expected[0u] &&
         std::get<1>(*output) == expected[1u] &&
         std::get<2>(*output) == expected[2u] &&
         std::get<3>(*output) == expected[3u] &&
         std::get<4>(*output) == expected[4u] &&
         std::get<5>(*output) == expected[5u];
}

template <class T>
[[nodiscard]] auto BuildResidentWindowProgram(
    const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto target = flow_on(backend, Target::cpu(2u));
  return std::move(target)
      .template input<Bounded<T>>(kResidentWindowCapacity)
      .map("resident-window-wrap-source", [](auto value) {
        if constexpr (detail::FixedValue<T>) {
          return quantize<T, Rounding::NearestEven, Overflow::Wrap>(value);
        } else {
          return value;
        }
      })
      .branch([](auto values) {
        return outputs(values.window({.op = Window::Sum,
                                      .radius = kResidentWindowRadius}),
                       values.window({.op = Window::Min,
                                      .radius = kResidentWindowRadius}),
                       values.window({.op = Window::Max,
                                      .radius = kResidentWindowRadius}),
                       values.window({.op = Window::Sum,
                                      .radius = kResidentWindowRadius,
                                      .edge = WindowEdge::Clip}),
                       values.window({.op = Window::Min,
                                      .radius = kResidentWindowRadius,
                                      .edge = WindowEdge::Clip}),
                       values.window({.op = Window::Max,
                                      .radius = kResidentWindowRadius,
                                      .edge = WindowEdge::Clip}));
      })
      .compile();
}

} // namespace rund_node_collective_modes::bounded
