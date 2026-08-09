#include "src/accel/kernel/backend/source_recipe.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "src/accel/vulkan/kernel/reset_source.hpp"
#include "src/accel/vulkan/map/control.hpp"
#include "src/accel/vulkan/map/local.hpp"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "recipe.hpp"

namespace node_accel_contract {

namespace {

namespace source_recipe = rund::node::accel::detail::backend_source_recipe;

inline constexpr std::array<std::string_view, 3u> kBackendSourceRecipePrefix{
    "kernel void ", "",
    "rund_backend_source_recipe_contract_with_long_name(value="};
inline constexpr std::array<std::uint64_t, 6u> kBackendSourceRecipeDecimals{
    0u, 9u, 10u, 99u, 100u, std::numeric_limits<std::uint64_t>::max()};
inline constexpr std::array<std::string_view, 2u> kBackendSourceRecipeSuffix{
    "", ");"};

struct BackendSourceRecipeEmitter final {
  template <typename Sink>
  [[nodiscard]] bool operator()(Sink &sink) const noexcept(
      noexcept(source_recipe::append_fixed(sink, kBackendSourceRecipePrefix)) &&
      noexcept(source_recipe::append_decimal(sink, std::uint64_t{})) &&
      noexcept(sink.append(std::string_view{}))) {
    if (!source_recipe::append_fixed(sink, kBackendSourceRecipePrefix)) {
      return false;
    }
    for (std::size_t index = 0u; index < kBackendSourceRecipeDecimals.size();
         ++index) {
      if ((index != 0u && !sink.append(",")) ||
          !source_recipe::append_decimal(sink,
                                         kBackendSourceRecipeDecimals[index])) {
        return false;
      }
    }
    return source_recipe::append_fixed(sink, kBackendSourceRecipeSuffix);
  }
};

struct DivergentBackendSourceRecipeEmitter final {
  [[nodiscard]] bool operator()(source_recipe::CountSink &sink) const noexcept {
    return sink.append("x");
  }

  [[nodiscard]] bool operator()(source_recipe::StringSink &sink) const {
    return sink.append("xx");
  }
};

} // namespace

[[nodiscard]] bool BackendSourceRecipeIsCheckedAndCanonical() {
  constexpr std::string_view expected =
      "kernel void "
      "rund_backend_source_recipe_contract_with_long_name(value="
      "0,9,10,99,100,18446744073709551615);";
  std::uint64_t upper = 0u;
  const BackendSourceRecipeEmitter emit{};
  if (!source_recipe::bytes(emit, upper) || upper != expected.size() ||
      source_recipe::materialize(emit) != expected ||
      source_recipe::materialize(emit, upper) != expected ||
      source_recipe::materialize(emit, upper, upper + 37u) != expected ||
      !source_recipe::materialize(emit, upper - 1u).empty() ||
      !source_recipe::materialize(emit, upper, upper - 1u).empty() ||
      !source_recipe::materialize(emit, 0u).empty()) {
    return false;
  }

  // A failed count never overwrites the caller's previously frozen upper.
  const auto reject = [](source_recipe::CountSink &) noexcept { return false; };
  const auto empty = [](source_recipe::CountSink &) noexcept { return true; };
  upper = 41u;
  if (source_recipe::bytes(reject, upper) || upper != 41u ||
      source_recipe::bytes(empty, upper) || upper != 41u) {
    return false;
  }

  // Fragment and decimal construction use the same common builder for every
  // sink; no backend wrapper owns a second failure or formatting rule.
  std::string builder_text;
  source_recipe::StringSink builder_sink{builder_text};
  source_recipe::SourceBuilder builder{builder_sink};
  builder += "value=";
  if (!builder.decimal(42u) || !builder.valid() || builder_text != "value=42") {
    return false;
  }
  source_recipe::CountSink builder_overflow{
      std::numeric_limits<std::uint64_t>::max()};
  source_recipe::SourceBuilder rejected_builder{builder_overflow};
  if (rejected_builder.append("x") || rejected_builder.append("") ||
      rejected_builder.valid() ||
      builder_overflow.bytes() != std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }

  // Overflow is checked at the counter boundary. The rejected append leaves
  // the preceding exact count intact instead of wrapping it to zero.
  source_recipe::CountSink full{std::numeric_limits<std::uint64_t>::max()};
  if (full.append("x") ||
      full.bytes() != std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  source_recipe::CountSink fragments{std::numeric_limits<std::uint64_t>::max() -
                                     1u};
  constexpr std::array<std::string_view, 3u> overflow_fragments{"", "x", "yz"};
  if (source_recipe::append_fixed(fragments, overflow_fragments) ||
      fragments.bytes() != std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }

  // Count and materialization are one recipe authority. If an emitter ever
  // diverges by sink, materialization rejects the result rather than retaining
  // an under-reserved or semantically different source.
  if (!source_recipe::materialize(DivergentBackendSourceRecipeEmitter{})
           .empty()) {
    return false;
  }

  std::uint64_t storage_upper = 0u;
  std::uint64_t overflow_storage = 0u;
  const std::string materialized =
      source_recipe::materialize(emit, expected.size());
  if (!source_recipe::string_external_storage_upper_bytes(expected.size(),
                                                          storage_upper) ||
      source_recipe::string_external_storage_upper_bytes(
          std::numeric_limits<std::uint64_t>::max(), overflow_storage) ||
      materialized != expected ||
      !source_recipe::string_external_storage_within(materialized,
                                                     storage_upper)) {
    return false;
  }
  std::array<char, 39u> fixed_storage{};
  source_recipe::FixedBufferSink<39u> fixed{fixed_storage};
  if (!fixed.append("0x") ||
      !source_recipe::append_hex64_digits(
          fixed, std::numeric_limits<std::uint64_t>::max()) ||
      !fixed.append(":") ||
      !source_recipe::append_decimal(
          fixed, std::numeric_limits<std::uint64_t>::max()) ||
      fixed.text() != "0xffffffffffffffff:18446744073709551615") {
    return false;
  }
  std::string over_capacity{expected};
  over_capacity.reserve(static_cast<std::size_t>(storage_upper + 64u));
  return !source_recipe::string_external_storage_within(over_capacity,
                                                        storage_upper);
}

[[nodiscard]] bool VulkanMapAndResetSourceRecipesAreExact() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  VulkanMapTemplateResources prepared{};
  prepared.plan.scalar = rund::kernel::ComputeScalar::Lane32;
  prepared.plan.domain = rund::kernel::ComputeDomain::U32;
  prepared.plan.input_buffer_count = 2u;
  prepared.plan.output_buffer_count = 1u;
  prepared.plan.op_hash_hi = 0x1234u;
  prepared.plan.op_hash_lo = 0x5678u;
  prepared.checks = {
      VulkanMapCheck{
          .binding = 0u, .limit = 17u, .offset = 4096u, .stride = 8u},
      VulkanMapCheck{
          .binding = 1u, .limit = 1009u, .offset = 64u, .stride = 16u},
  };
  const rund::kernel::LoweringArtifact check = VulkanMapCheckArtifact(prepared);
  std::uint64_t check_upper = 0u;
  std::uint64_t check_storage = 0u;
  if (!check.ok ||
      !VulkanMapCheckSourceUpperBytes(2u, 6u, 3u, 6u, check_upper) ||
      check.source_text.size() != check_upper ||
      check.source_text_upper_bytes != check_upper ||
      !source_recipe::string_external_storage_upper_bytes(check_upper,
                                                          check_storage) ||
      !source_recipe::string_external_storage_within(check.source_text,
                                                     check_storage)) {
    return false;
  }

  rund::kernel::LoweringArtifact base{};
  base.key.op_hash_hi = prepared.plan.op_hash_hi;
  base.key.op_hash_lo = prepared.plan.op_hash_lo;
  base.source_text =
      std::string{vulkan_controlled_map_source_detail::CanonicalVariant} +
      "\n" + std::string{vulkan_controlled_map_source_detail::Entry} +
      std::string{vulkan_controlled_map_source_detail::Guard};
  base.source_text_upper_bytes = base.source_text.size();
  base.ok = true;
  base.reason = "ok";
  std::uint64_t controlled_upper = 0u;
  if (!VulkanControlledMapSourceUpperBytes(
          prepared.plan, base.source_text_upper_bytes, controlled_upper)) {
    return false;
  }
  const rund::kernel::LoweringArtifact controlled =
      VulkanControlledMapArtifact(std::move(base), prepared.plan);
  const std::string expected_binding =
      "layout(set = 0, binding = 4, std430) readonly buffer RundControlArgs";
  if (!controlled.ok || controlled.source_text.size() != controlled_upper ||
      controlled.source_text_upper_bytes != controlled_upper ||
      controlled.source_text.find(expected_binding) == std::string::npos ||
      controlled.source_text.find(
          vulkan_controlled_map_source_detail::ControlledGuard) ==
          std::string::npos ||
      controlled.source_text.find(
          vulkan_controlled_map_source_detail::ControlledVariant) ==
          std::string::npos ||
      controlled.source_text.find(
          vulkan_controlled_map_source_detail::CanonicalVariant) !=
          std::string::npos) {
    return false;
  }

  const rund::kernel::LoweringArtifact control =
      VulkanMapControlArtifact(prepared.plan);
  const rund::kernel::LoweringArtifact reset = VulkanResetArtifact();
  std::uint64_t control_source_bytes = 0u;
  return VulkanMapControlSourceBytes(control_source_bytes) && control.ok &&
         control.source_text.size() == control_source_bytes &&
         control.source_text_upper_bytes == control_source_bytes &&
         control.source_text.find("dispatch_count == 0u") !=
             std::string::npos &&
         control.source_text.find("1u + (dispatch_count - 1u) / 256u;") !=
             std::string::npos &&
         reset.ok && reset.source_text == VulkanResetSourceText() &&
         reset.source_text_upper_bytes == VulkanResetSourceText().size();
#else
  return true;
#endif
}

} // namespace node_accel_contract
