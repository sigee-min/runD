#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <vector>

namespace rund::compute {

template <class... T> struct Outputs final {};
template <class T> struct Scalar final {};
template <class Tag, class T> struct Field final {};
template <class... T> struct Record final {};
template <class T>
using CountFor =
    std::conditional_t<sizeof(T) == 8u, std::uint64_t, std::uint32_t>;
template <class T, class Count = CountFor<T>> struct Bounded final {};
namespace detail {
template <class T> inline constexpr bool is_outputs = false;
template <class... T> inline constexpr bool is_outputs<Outputs<T...>> = true;
template <class T> inline constexpr bool is_bounded = false;
template <class T, class Count>
inline constexpr bool is_bounded<Bounded<T, Count>> = true;
template <class T> struct BoundedTraits;
template <class T, class Count> struct BoundedTraits<Bounded<T, Count>> final {
  using Value = T;
  using CountType = Count;
};
template <class T> inline constexpr std::size_t schema_leaf_count = 1u;
template <class T>
inline constexpr std::size_t schema_leaf_count<Scalar<T>> = 1u;
template <class T, class Count>
inline constexpr std::size_t schema_leaf_count<Bounded<T, Count>> = 2u;
template <class Tag, class T>
inline constexpr std::size_t schema_leaf_count<Field<Tag, T>> =
    schema_leaf_count<T>;
template <class... T>
inline constexpr std::size_t schema_leaf_count<Record<T...>> =
    (schema_leaf_count<T> + ... + 0u);
template <class... T>
inline constexpr std::size_t schema_leaf_count<Outputs<T...>> =
    (schema_leaf_count<T> + ... + 0u);
template <class T> struct HostValue final {
  using Type = std::vector<T>;
};
template <class T> struct HostValue<Scalar<T>> final {
  using Type = T;
};
template <class T, class Count> struct HostValue<Bounded<T, Count>> final {
  using Type = std::vector<T>;
};
template <class Tag, class T> struct HostValue<Field<Tag, T>> final {
  using Type = typename HostValue<T>::Type;
};
template <class... T> struct HostValue<Record<T...>> final {
  using Type = std::tuple<typename HostValue<T>::Type...>;
};
template <class T> using HostValueT = typename HostValue<T>::Type;
} // namespace detail

} // namespace rund::compute
