#pragma once

#include <rund/compute/buffer.hpp>
#include <rund/compute/fixed.hpp>

#include <array>
#include <memory>
#include <type_traits>
#include <utility>

namespace rund::compute {
class PipelineBuilder;
}

namespace rund::compute::detail {

template <class... T> struct TypeList final {};

template <std::size_t Index, class List> struct U32At : std::false_type {};
template <class Head, class... Tail>
struct U32At<0u, TypeList<Head, Tail...>>
    : std::bool_constant<std::is_same_v<Head, std::uint32_t>> {};
template <std::size_t Index, class Head, class... Tail>
struct U32At<Index, TypeList<Head, Tail...>>
    : U32At<Index - 1u, TypeList<Tail...>> {};

template <class Prefix, class List> struct StartsWith;
template <class... T>
struct StartsWith<TypeList<>, TypeList<T...>> : std::true_type {};
template <class H, class... P, class... T>
struct StartsWith<TypeList<H, P...>, TypeList<H, T...>>
    : StartsWith<TypeList<P...>, TypeList<T...>> {};
template <class... P, class... T>
struct StartsWith<TypeList<P...>, TypeList<T...>> final : std::false_type {};

template <class... Lists> struct Join;
template <> struct Join<> final {
  using type = TypeList<>;
};
template <class... T> struct Join<TypeList<T...>> final {
  using type = TypeList<T...>;
};
template <class... L, class... R, class... Rest>
struct Join<TypeList<L...>, TypeList<R...>, Rest...> final {
  using type = typename Join<TypeList<L..., R...>, Rest...>::type;
};

template <class List> struct Reverse;
template <> struct Reverse<TypeList<>> final {
  using type = TypeList<>;
};
template <class Head, class... Tail>
struct Reverse<TypeList<Head, Tail...>> final {
  using type = typename Join<typename Reverse<TypeList<Tail...>>::type,
                             TypeList<Head>>::type;
};

template <class Suffix, class List>
struct EndsWith final
    : std::bool_constant<StartsWith<typename Reverse<Suffix>::type,
                                    typename Reverse<List>::type>::value> {};

template <class Prefix, class List> struct DropPrefix;
template <class... T> struct DropPrefix<TypeList<>, TypeList<T...>> final {
  using type = TypeList<T...>;
};
template <class Head, class... Prefix, class... Tail>
struct DropPrefix<TypeList<Head, Prefix...>, TypeList<Head, Tail...>> final {
  using type =
      typename DropPrefix<TypeList<Prefix...>, TypeList<Tail...>>::type;
};

template <class Suffix, class List, bool Valid = EndsWith<Suffix, List>::value>
struct StripSuffix final {
  using type = TypeList<>;
  static constexpr bool valid = false;
};
template <class Suffix, class List>
struct StripSuffix<Suffix, List, true> final {
  using reversed = typename DropPrefix<typename Reverse<Suffix>::type,
                                       typename Reverse<List>::type>::type;
  using type = typename Reverse<reversed>::type;
  static constexpr bool valid = true;
};

template <class List> struct WindowCoordinateSplit final {
  using Prefix = TypeList<>;
  static constexpr bool valid = false;
};
template <class Count, class Ordinal>
struct WindowCoordinateSplit<TypeList<Count, Ordinal>> final {
  using Prefix = TypeList<>;
  static constexpr bool valid = std::is_same_v<Count, std::uint32_t> &&
                                std::is_same_v<Ordinal, std::uint32_t>;
};
template <class Head, class Next, class Last, class... Tail>
struct WindowCoordinateSplit<TypeList<Head, Next, Last, Tail...>> final {
private:
  using Rest = WindowCoordinateSplit<TypeList<Next, Last, Tail...>>;

public:
  using Prefix = typename Join<TypeList<Head>, typename Rest::Prefix>::type;
  static constexpr bool valid = Rest::valid;
};

template <class T> struct SchemaTypes final {
  using type = TypeList<T>;
};
template <class T> struct SchemaTypes<Scalar<T>> final {
  using type = TypeList<T>;
};
template <class T, class Count> struct SchemaTypes<Bounded<T, Count>> final {
  using type = TypeList<T, Count>;
};
template <class Tag, class T> struct SchemaTypes<Field<Tag, T>> final {
  using type = typename SchemaTypes<T>::type;
};
template <class... T> struct SchemaTypes<Record<T...>> final {
  using type = typename Join<typename SchemaTypes<T>::type...>::type;
};
template <class... T> struct SchemaTypes<Outputs<T...>> final {
  using type = typename Join<typename SchemaTypes<T>::type...>::type;
};

template <class> struct SignatureTypes;
template <class R, class... A> struct SignatureTypes<R(A...)> final {
  using Inputs = TypeList<A...>;
  using Outputs = typename SchemaTypes<R>::type;
};

template <class T> struct BufferElement final {
  static constexpr bool writable = false;
};
template <class T> struct BufferElement<View<T>> final {
  using type = std::remove_const_t<T>;
  static constexpr bool writable = !std::is_const_v<T>;
};
template <class T> struct BufferElement<Buffer<T>> final {
  using type = T;
  static constexpr bool writable = true;
};

template <class T> inline constexpr bool IsBuffer = false;
template <class T> inline constexpr bool IsBuffer<Buffer<T>> = true;
template <class T> inline constexpr bool IsView = false;
template <class T> inline constexpr bool IsView<View<T>> = true;

template <class B>
inline constexpr bool IsReadableBuffer =
    (IsBuffer<std::remove_cvref_t<B>> && std::is_lvalue_reference_v<B &&> &&
     !std::is_volatile_v<std::remove_reference_t<B>>) ||
    (IsView<std::remove_cvref_t<B>> &&
     !std::is_volatile_v<std::remove_reference_t<B>>);

template <class B>
inline constexpr bool IsWritableBuffer =
    IsReadableBuffer<B> && !std::is_const_v<std::remove_reference_t<B>> &&
    BufferElement<std::remove_cvref_t<B>>::writable;

struct BufferAccess final {
  template <class T>
  [[nodiscard]] static const std::shared_ptr<BufferState> &
  state(const Buffer<T> &buffer) noexcept {
    return buffer.state_;
  }
  template <class T>
  [[nodiscard]] static const std::shared_ptr<BufferState> &
  state(const View<T> &view) noexcept {
    return view.state_;
  }
  template <class T>
  [[nodiscard]] static ResourceView view(const Buffer<T> &buffer,
                                         const ResourceAccess access) noexcept {
    return ResourceView{.buffer = buffer.state_,
                        .type = type<T>(),
                        .format = storage_format<T>(),
                        .count = buffer_size(buffer.state_),
                        .element_bytes = sizeof(T),
                        .alignment = alignof(T),
                        .access = access};
  }
  template <class T>
  [[nodiscard]] static ResourceView view(const View<T> &view,
                                         const ResourceAccess access) noexcept {
    using Value = std::remove_const_t<T>;
    return ResourceView{.buffer = view.state_,
                        .type = type<Value>(),
                        .format = storage_format<Value>(),
                        .offset = view.offset_,
                        .count = view.count_,
                        .stride = view.stride_,
                        .element_bytes = sizeof(Value),
                        .alignment = view.alignment_,
                        .access = access};
  }
};

enum class BindingRole { Read, StepWrite, FinalWrite, WindowWrite, EachWrite };

template <BindingRole Role, class... T> class BindingPack final {
public:
  static constexpr std::size_t size = sizeof...(T);

  template <class... B>
  explicit BindingPack(B &&...buffers) noexcept
      : views_{BufferAccess::view(buffers, Role == BindingRole::Read
                                               ? ResourceAccess::Read
                                               : ResourceAccess::Write)...} {}

  BindingPack(const BindingPack &) = delete;
  BindingPack &operator=(const BindingPack &) = delete;
  BindingPack(BindingPack &&) noexcept = default;
  BindingPack &operator=(BindingPack &&) noexcept = default;

private:
  friend class ::rund::compute::PipelineBuilder;

  std::array<ResourceView, size> views_{};
};

template <class... T> using ReadPack = BindingPack<BindingRole::Read, T...>;
template <class... T>
using WritePack = BindingPack<BindingRole::StepWrite, T...>;
template <class... T>
using WriteFinalPack = BindingPack<BindingRole::FinalWrite, T...>;
template <class... T>
using WriteWindowPack = BindingPack<BindingRole::WindowWrite, T...>;
template <class... T>
using WriteEachPack = BindingPack<BindingRole::EachWrite, T...>;

} // namespace rund::compute::detail

namespace rund::compute {

template <class... B>
  requires(sizeof...(B) != 0u && (detail::IsReadableBuffer<B> && ...))
[[nodiscard]] auto read(B &&...buffers) noexcept {
  return detail::ReadPack<
      typename detail::BufferElement<std::remove_cvref_t<B>>::type...>{
      buffers...};
}

template <class... B>
  requires(sizeof...(B) != 0u && (detail::IsWritableBuffer<B> && ...))
[[nodiscard]] auto write(B &&...buffers) noexcept {
  return detail::WritePack<
      typename detail::BufferElement<std::remove_cvref_t<B>>::type...>{
      buffers...};
}

template <class... B>
  requires(sizeof...(B) != 0u && (detail::IsWritableBuffer<B> && ...))
[[nodiscard]] auto write_final(B &&...buffers) noexcept {
  return detail::WriteFinalPack<
      typename detail::BufferElement<std::remove_cvref_t<B>>::type...>{
      buffers...};
}

template <class... B>
  requires(sizeof...(B) != 0u && (detail::IsWritableBuffer<B> && ...))
[[nodiscard]] auto write_window(B &&...buffers) noexcept {
  return detail::WriteWindowPack<
      typename detail::BufferElement<std::remove_cvref_t<B>>::type...>{
      buffers...};
}

template <class... B>
  requires(sizeof...(B) != 0u && (detail::IsWritableBuffer<B> && ...))
[[nodiscard]] auto write_each(B &&...buffers) noexcept {
  return detail::WriteEachPack<
      typename detail::BufferElement<std::remove_cvref_t<B>>::type...>{
      buffers...};
}

} // namespace rund::compute
