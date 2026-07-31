#pragma once

#include <kernel/program/compute/ir.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rund::compute_dsl::detail {

template <std::size_t Size> struct FixedString {
  char value[Size]{};

  constexpr FixedString(const char (&text)[Size]) noexcept {
    for (std::size_t index = 0u; index < Size; ++index) {
      value[index] = text[index];
    }
  }

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return Size == 0u ? 0u : Size - 1u;
  }

  [[nodiscard]] constexpr std::string_view view() const noexcept {
    return std::string_view{value, size()};
  }
};

template <std::size_t Size>
FixedString(const char (&)[Size]) -> FixedString<Size>;

template <FixedString Lhs, FixedString Rhs>
[[nodiscard]] consteval bool FixedStringEqual() noexcept {
  if constexpr (sizeof(Lhs.value) != sizeof(Rhs.value)) {
    return false;
  }
  for (std::size_t index = 0u; index < sizeof(Lhs.value); ++index) {
    if (Lhs.value[index] != Rhs.value[index]) {
      return false;
    }
  }
  return true;
}

template <FixedString Name>
[[nodiscard]] constexpr std::string_view FixedStringView() noexcept {
  return Name.view();
}

enum class BindingKind : rund::kernel::u8 {
  Param = 1u,
  Read = 2u,
  Write = 3u,
};

enum class ScalarMode : rund::kernel::u8 {
  Unspecified = 0u,
  FixedLane32 = 1u,
  FixedLane64 = 2u,
  I32 = 3u,
  U32 = 4u,
  I64 = 5u,
  U64 = 6u,
};

template <FixedString Name, BindingKind Kind, typename Value>
struct BindingDecl {
  using value_type = Value;
  static constexpr BindingKind kind = Kind;

  template <FixedString Query>
  [[nodiscard]] static consteval bool named() noexcept {
    return FixedStringEqual<Name, Query>();
  }
};

template <FixedString Name, BindingKind Kind, typename Decl>
[[nodiscard]] consteval bool BindingMatches() noexcept {
  return Decl::kind == Kind && Decl::template named<Name>();
}

template <FixedString Name, typename Decl>
[[nodiscard]] consteval bool BindingNameMatches() noexcept {
  return Decl::template named<Name>();
}

template <FixedString Name, BindingKind Kind, typename... Decls>
inline constexpr bool HasBinding = (BindingMatches<Name, Kind, Decls>() || ...);

template <FixedString Name, typename... Decls>
inline constexpr bool HasBindingName =
    (BindingNameMatches<Name, Decls>() || ...);

struct BindingRuntime {
  BindingKind kind = BindingKind::Param;
  ScalarMode numeric_mode = ScalarMode::Unspecified;
  std::string name;
  rund::kernel::u32 element_bytes = 0u;
  const void *runtime_data = nullptr;
  void *runtime_write_data = nullptr;
  rund::kernel::u64 runtime_count = 0u;
  rund::kernel::u64 runtime_stride_bytes = 0u;
  std::vector<rund::kernel::u8> value_bytes{};
  bool floating_point_param = false;
  bool has_runtime_storage = false;
};

template <typename T>
using CleanType = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T> struct BufferRef {
  using element_type = CleanType<T>;
};

template <typename T> [[nodiscard]] constexpr BufferRef<T> buffer() noexcept {
  return BufferRef<T>{};
}

template <typename T> struct BufferTraits {
  static constexpr bool valid = false;
  static constexpr bool writable = false;
  using element_type = void;
};

template <typename T> struct BufferTraits<T *> {
  static constexpr bool valid = std::is_object_v<std::remove_cv_t<T>>;
  static constexpr bool writable = valid && !std::is_const_v<T>;
  using element_type = std::remove_cv_t<T>;
};

template <typename T> struct BufferTraits<T *const> : BufferTraits<T *> {};

template <typename T> struct BufferTraits<T *volatile> : BufferTraits<T *> {};

template <typename T>
struct BufferTraits<T *const volatile> : BufferTraits<T *> {};

template <typename T, std::size_t Count> struct BufferTraits<T[Count]> {
  static constexpr bool valid = true;
  static constexpr bool writable = valid && !std::is_const_v<T>;
  using element_type = std::remove_cv_t<T>;
};

template <typename T> struct BufferTraits<BufferRef<T>> {
  using raw_element_type = std::remove_reference_t<T>;
  static constexpr bool valid =
      std::is_object_v<std::remove_cv_t<raw_element_type>>;
  static constexpr bool writable = valid && !std::is_const_v<raw_element_type>;
  using element_type = std::remove_cv_t<raw_element_type>;
};

template <typename T> struct IsBufferRef : std::false_type {};

template <typename T> struct IsBufferRef<BufferRef<T>> : std::true_type {};

template <typename Buffer> using BufferArg = std::remove_reference_t<Buffer>;

template <typename Buffer>
inline constexpr bool ValidBuffer = BufferTraits<BufferArg<Buffer>>::valid;

template <typename Buffer>
inline constexpr bool WritableBuffer =
    BufferTraits<BufferArg<Buffer>>::writable;

template <typename Buffer>
using BufferElement = typename BufferTraits<BufferArg<Buffer>>::element_type;

template <typename Buffer>
concept ComputeBuffer = ValidBuffer<Buffer>;

template <typename Buffer>
concept ComputeWritableBuffer = ValidBuffer<Buffer> && WritableBuffer<Buffer>;

struct RuntimeBuffer {
  const void *data = nullptr;
  void *write_data = nullptr;
  rund::kernel::u64 count = 0u;
  rund::kernel::u64 stride_bytes = 0u;
  bool has_storage = false;
};

template <typename Buffer>
[[nodiscard]] constexpr RuntimeBuffer
RuntimeBufferFor(Buffer &&buffer, const rund::kernel::u64 tile_count) noexcept {
  using Arg = BufferArg<Buffer &&>;
  using Element = BufferElement<Buffer &&>;
  if constexpr (IsBufferRef<Arg>::value) {
    (void)buffer;
    (void)tile_count;
    return RuntimeBuffer{};
  } else if constexpr (std::is_pointer_v<Arg>) {
    return RuntimeBuffer{
        .data = buffer,
        .write_data = const_cast<std::remove_const_t<Element> *>(buffer),
        .count = tile_count,
        .stride_bytes = static_cast<rund::kernel::u64>(sizeof(Element)),
        .has_storage = buffer != nullptr && tile_count != 0u,
    };
  } else if constexpr (std::is_array_v<Arg>) {
    constexpr std::size_t count = std::extent_v<Arg>;
    return RuntimeBuffer{
        .data = count == 0u ? nullptr : std::addressof(buffer[0]),
        .write_data = count == 0u ? nullptr
                                  : const_cast<std::remove_const_t<Element> *>(
                                        std::addressof(buffer[0])),
        .count = static_cast<rund::kernel::u64>(count),
        .stride_bytes = static_cast<rund::kernel::u64>(sizeof(Element)),
        .has_storage = count != 0u,
    };
  } else {
    (void)buffer;
    (void)tile_count;
    return RuntimeBuffer{};
  }
}

template <typename T>
[[nodiscard]] constexpr bool FixedParamUnsupported() noexcept {
  using Clean = CleanType<T>;
  return std::is_floating_point_v<Clean>;
};

template <typename T>
[[nodiscard]] constexpr rund::kernel::u32 ElementBytes() noexcept {
  static_assert(ValidBuffer<T>, "compute read/write bindings require a buffer");
  using Element = BufferElement<T>;
  return static_cast<rund::kernel::u32>(sizeof(Element));
}

template <typename T>
void AppendIntegralBytes(std::vector<rund::kernel::u8> &out, const T value) {
  using Clean = CleanType<T>;
  using Unsigned = std::make_unsigned_t<Clean>;
  Unsigned encoded = static_cast<Unsigned>(value);
  for (std::size_t index = 0u; index < sizeof(Unsigned); ++index) {
    out.push_back(static_cast<rund::kernel::u8>((encoded >> (index * 8u)) &
                                                static_cast<Unsigned>(0xffu)));
  }
}

template <typename T>
void AppendFloatBytes(std::vector<rund::kernel::u8> &out, const T value) {
  static_assert(sizeof(T) == 4u || sizeof(T) == 8u);
  if constexpr (sizeof(T) == 4u) {
    rund::kernel::u32 encoded = 0u;
    std::memcpy(&encoded, &value, sizeof(encoded));
    AppendIntegralBytes(out, encoded);
  } else {
    rund::kernel::u64 encoded = 0u;
    std::memcpy(&encoded, &value, sizeof(encoded));
    AppendIntegralBytes(out, encoded);
  }
}

template <typename T>
[[nodiscard]] std::vector<rund::kernel::u8> ParamBytes(const T value) {
  using Clean = CleanType<T>;
  std::vector<rund::kernel::u8> bytes;
  bytes.reserve(sizeof(Clean));
  if constexpr (std::is_same_v<Clean, bool>) {
    bytes.push_back(value ? rund::kernel::u8{1u} : rund::kernel::u8{0u});
  } else if constexpr (std::is_enum_v<Clean>) {
    AppendIntegralBytes(bytes,
                        static_cast<std::underlying_type_t<Clean>>(value));
  } else if constexpr (std::is_integral_v<Clean>) {
    AppendIntegralBytes(bytes, value);
  } else if constexpr (std::is_floating_point_v<Clean>) {
    AppendFloatBytes(bytes, value);
  } else {
    static_assert(std::is_arithmetic_v<Clean> || std::is_enum_v<Clean>,
                  "compute params must be arithmetic or enum values");
  }
  return bytes;
}

[[nodiscard]] constexpr bool NumericMode(const ScalarMode mode) noexcept {
  return mode == ScalarMode::I32 || mode == ScalarMode::U32 ||
         mode == ScalarMode::I64 || mode == ScalarMode::U64 ||
         mode == ScalarMode::FixedLane32 || mode == ScalarMode::FixedLane64;
}

[[nodiscard]] constexpr bool WideMode(const ScalarMode mode) noexcept {
  return mode == ScalarMode::I64 || mode == ScalarMode::U64 ||
         mode == ScalarMode::FixedLane64;
}

[[nodiscard]] constexpr ScalarMode
WriteNumericMode(const ScalarMode graph_mode,
                 const rund::kernel::u32 element_bytes) noexcept {
  if (!NumericMode(graph_mode)) {
    return graph_mode;
  }
  const rund::kernel::u32 scalar_bytes = WideMode(graph_mode) ? 8u : 4u;
  if (element_bytes == scalar_bytes) {
    return graph_mode;
  }
  if (element_bytes == 4u) {
    return ScalarMode::U32;
  }
  if (element_bytes == 8u) {
    return ScalarMode::U64;
  }
  return ScalarMode::Unspecified;
}

[[nodiscard]] constexpr rund::kernel::ComputeScalar
ToComputeScalar(const ScalarMode mode) noexcept {
  switch (mode) {
  case ScalarMode::FixedLane32:
  case ScalarMode::I32:
  case ScalarMode::U32:
    return rund::kernel::ComputeScalar::Lane32;
  case ScalarMode::FixedLane64:
  case ScalarMode::I64:
  case ScalarMode::U64:
    return rund::kernel::ComputeScalar::Lane64;
  case ScalarMode::Unspecified:
    return static_cast<rund::kernel::ComputeScalar>(0u);
  }
  return static_cast<rund::kernel::ComputeScalar>(0u);
}

[[nodiscard]] constexpr rund::kernel::ComputeDomain
ToComputeDomain(const ScalarMode mode) noexcept {
  switch (mode) {
  case ScalarMode::Unspecified:
    return static_cast<rund::kernel::ComputeDomain>(0u);
  case ScalarMode::I32:
    return rund::kernel::ComputeDomain::I32;
  case ScalarMode::U32:
    return rund::kernel::ComputeDomain::U32;
  case ScalarMode::I64:
    return rund::kernel::ComputeDomain::I64;
  case ScalarMode::U64:
    return rund::kernel::ComputeDomain::U64;
  case ScalarMode::FixedLane64:
    return rund::kernel::ComputeDomain::Fixed;
  case ScalarMode::FixedLane32:
    return rund::kernel::ComputeDomain::Fixed;
  }
  return static_cast<rund::kernel::ComputeDomain>(0u);
}

} // namespace rund::compute_dsl::detail
