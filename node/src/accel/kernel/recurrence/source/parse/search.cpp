#include "local.hpp"

#include "../../../backend/source/marker.hpp"

#include <kernel/program/compute/lowering/metal/syntax.hpp>
#include <kernel/program/compute/lowering/vulkan/syntax.hpp>

#include <array>
#include <string_view>

namespace rund::node::accel::detail::recurrence_source_detail {
namespace {

[[nodiscard]] bool ConsumeSymbol(const std::string_view source,
                                 std::size_t &cursor,
                                 const std::string_view access,
                                 const std::string_view name) noexcept {
  return backend_source_marker::consume_fragment(source, cursor, access) &&
         backend_source_marker::consume_safe_identifier(source, cursor, name);
}

template <typename Match>
[[nodiscard]] bool FindUniqueStructured(const std::string_view source,
                                        const std::string_view anchor,
                                        Match &&match, std::size_t &begin,
                                        std::size_t &end) noexcept {
  begin = std::string_view::npos;
  end = std::string_view::npos;
  std::size_t search = 0u;
  while (search < source.size()) {
    const std::size_t at = source.find(anchor, search);
    if (at == std::string_view::npos) {
      break;
    }
    std::size_t cursor = at;
    if (match(cursor)) {
      if (begin != std::string_view::npos) {
        return false;
      }
      begin = at;
      end = cursor;
    }
    search = at + 1u;
  }
  return begin != std::string_view::npos;
}

} // namespace

bool FindOne(const std::string_view source, const std::string_view needle,
             std::size_t &at) noexcept {
  if (needle.empty()) {
    return false;
  }
  at = source.find(needle);
  return at != std::string_view::npos &&
         source.find(needle, at + needle.size()) == std::string_view::npos;
}

bool FindInputLoad(const std::string_view source, const ComputeApi api,
                   const ComputeScalar scalar, const std::string_view name,
                   const bool uniform, std::size_t &begin,
                   std::size_t &end) noexcept {
  const std::string_view load =
      api == ComputeApi::Metal
          ? rund::kernel::compute_lowering_detail::MetalLoadFunction(scalar)
          : rund::kernel::compute_lowering_detail::VulkanLoadPrefix(scalar);
  return FindUniqueStructured(
      source, load,
      [&](std::size_t &cursor) noexcept {
        return backend_source_marker::consume_fragment(source, cursor, load) &&
               (api == ComputeApi::Metal
                    ? backend_source_marker::consume_fragment(source, cursor,
                                                              "(")
                    : backend_source_marker::consume_fragment(source, cursor,
                                                              "_") &&
                          ConsumeSymbol(source, cursor, "read_", name) &&
                          backend_source_marker::consume_fragment(
                              source, cursor, "(")) &&
               (api != ComputeApi::Metal ||
                (ConsumeSymbol(source, cursor, "read_", name) &&
                 backend_source_marker::consume_fragment(source, cursor,
                                                         ", "))) &&
               backend_source_marker::consume_fragment(source, cursor,
                                                       "RundBase_") &&
               ConsumeSymbol(source, cursor, "read_", name) &&
               (uniform || (backend_source_marker::consume_fragment(
                                source, cursor, " + gid * RundStride_") &&
                            ConsumeSymbol(source, cursor, "read_", name))) &&
               backend_source_marker::consume_fragment(source, cursor, ")");
      },
      begin, end);
}

bool FindOutputStore(const std::string_view source, const ComputeApi api,
                     const ComputeScalar scalar, const std::string_view name,
                     std::size_t &begin, std::size_t &value_begin,
                     std::size_t &value_end, std::size_t &end) noexcept {
  const std::string_view store =
      api == ComputeApi::Metal
          ? rund::kernel::compute_lowering_detail::MetalStoreFunction(scalar)
          : rund::kernel::compute_lowering_detail::VulkanStorePrefix(scalar);
  std::size_t prefix_end = 0u;
  if (!FindUniqueStructured(
          source, "  ",
          [&](std::size_t &cursor) noexcept {
            return backend_source_marker::consume_fragment(source, cursor,
                                                           "  ") &&
                   backend_source_marker::consume_fragment(source, cursor,
                                                           store) &&
                   (api == ComputeApi::Metal
                        ? backend_source_marker::consume_fragment(
                              source, cursor, "(") &&
                              ConsumeSymbol(source, cursor, "write_", name) &&
                              backend_source_marker::consume_fragment(
                                  source, cursor, ", ")
                        : backend_source_marker::consume_fragment(
                              source, cursor, "_") &&
                              ConsumeSymbol(source, cursor, "write_", name) &&
                              backend_source_marker::consume_fragment(
                                  source, cursor, "(")) &&
                   backend_source_marker::consume_fragment(source, cursor,
                                                           "RundBase_") &&
                   ConsumeSymbol(source, cursor, "write_", name) &&
                   backend_source_marker::consume_fragment(
                       source, cursor, " + gid * RundStride_") &&
                   ConsumeSymbol(source, cursor, "write_", name) &&
                   backend_source_marker::consume_fragment(source, cursor,
                                                           ", ");
          },
          begin, prefix_end)) {
    return false;
  }
  value_begin = prefix_end;
  value_end = source.find(");\n", value_begin);
  if (value_end == std::string_view::npos) {
    return false;
  }
  end = value_end + 3u;
  return true;
}

bool FindMetalKernelName(const std::string_view source,
                         const RecurrenceSourceRecipe &recipe,
                         std::size_t &begin, std::size_t &end) noexcept {
  return FindUniqueStructured(
      source, "rund_compute_map_",
      [&](std::size_t &cursor) noexcept {
        std::array<char, 16u> hi{};
        std::array<char, 16u> lo{};
        backend_source_recipe::FixedBufferSink<16u> hi_sink{hi};
        backend_source_recipe::FixedBufferSink<16u> lo_sink{lo};
        return backend_source_recipe::append_hex64_digits(
                   hi_sink, recipe.before.op_hash_hi) &&
               backend_source_recipe::append_hex64_digits(
                   lo_sink, recipe.before.op_hash_lo) &&
               backend_source_marker::consume_fragment(source, cursor,
                                                       "rund_compute_map_") &&
               backend_source_marker::consume_fragment(source, cursor,
                                                       hi_sink.text()) &&
               backend_source_marker::consume_fragment(source, cursor, "_") &&
               backend_source_marker::consume_fragment(source, cursor,
                                                       lo_sink.text());
      },
      begin, end);
}

} // namespace rund::node::accel::detail::recurrence_source_detail
