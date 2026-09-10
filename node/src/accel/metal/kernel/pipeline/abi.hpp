#pragma once

#include "../../../kernel/status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::node::accel::detail {

// Host/GPU parameter layouts only; no retained native resources or submission
// state.  The records are available on every host build so preparation code
// does not make the value ABI depend on the Metal SDK.
constexpr std::uint32_t kMetalPipelineReductionWidth = 128u;
static_assert(kMetalPipelineReductionWidth == 128u);

#define RUND_METAL_ABI_BEGIN(host, gpu) struct host final {
#define RUND_METAL_ABI_U32(name, initial, offset) std::uint32_t name{initial};
#define RUND_METAL_ABI_U64(name, initial, offset) std::uint64_t name{initial};
#define RUND_METAL_ABI_ARRAY64(name, count, offset)                            \
  std::array<std::uint64_t, count> name{};
#define RUND_METAL_ABI_POLICIES(host_name, gpu0, gpu1, gpu2, gpu3, offset)     \
  std::array<std::uint32_t, 4u> host_name{};
#define RUND_METAL_ABI_END(host, gpu, size, alignment)                         \
  }                                                                            \
  ;

#include "abi/schema/records.def"

#undef RUND_METAL_ABI_BEGIN
#undef RUND_METAL_ABI_U32
#undef RUND_METAL_ABI_U64
#undef RUND_METAL_ABI_ARRAY64
#undef RUND_METAL_ABI_POLICIES
#undef RUND_METAL_ABI_END

// Replay each schema for layout and default checks.  The field declarations
// above and these assertions therefore cannot drift into separate contracts.
#define RUND_METAL_ABI_BEGIN(host, gpu)                                        \
  namespace abi_layout_##host {                                                \
    using Record = host;
#define RUND_METAL_ABI_U32(name, initial, offset)                              \
  static_assert(offsetof(Record, name) == offset);                             \
  static_assert(Record{}.name == initial);
#define RUND_METAL_ABI_U64(name, initial, offset)                              \
  static_assert(offsetof(Record, name) == offset);                             \
  static_assert(Record{}.name == initial);
#define RUND_METAL_ABI_ARRAY64(name, count, offset)                            \
  static_assert(offsetof(Record, name) == offset);                             \
  static_assert(sizeof(Record{}.name) == count * sizeof(std::uint64_t));       \
  static_assert(Record{}.name[0u] == 0u);                                      \
  static_assert(Record{}.name[count - 1u] == 0u);
#define RUND_METAL_ABI_POLICIES(host_name, gpu0, gpu1, gpu2, gpu3, offset)     \
  static_assert(offsetof(Record, host_name) == offset);                        \
  static_assert(sizeof(Record{}.host_name) == 4u * sizeof(std::uint32_t));     \
  static_assert(Record{}.host_name[0u] == 0u);                                 \
  static_assert(Record{}.host_name[1u] == 0u);                                 \
  static_assert(Record{}.host_name[2u] == 0u);                                 \
  static_assert(Record{}.host_name[3u] == 0u);
#define RUND_METAL_ABI_END(host, gpu, size, alignment)                         \
  static_assert(sizeof(host) == size);                                         \
  static_assert(alignof(host) == alignment);                                   \
  static_assert(std::is_standard_layout_v<host>);                              \
  static_assert(std::is_trivially_copyable_v<host>);                           \
  }

#include "abi/schema/records.def"

#undef RUND_METAL_ABI_BEGIN
#undef RUND_METAL_ABI_U32
#undef RUND_METAL_ABI_U64
#undef RUND_METAL_ABI_ARRAY64
#undef RUND_METAL_ABI_POLICIES
#undef RUND_METAL_ABI_END

} // namespace rund::node::accel::detail
