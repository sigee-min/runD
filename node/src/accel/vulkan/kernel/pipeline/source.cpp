#include "source.hpp"

#include "../../../kernel/backend/source_recipe.hpp"

#include <array>
#include <span>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] std::string_view VulkanTelemetrySourceText() noexcept {
  return R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 1) in;
layout(set=0,binding=0,std430) readonly buffer Primary { uint primary[]; };
layout(set=0,binding=1,std430) readonly buffer Count { uint count_words[]; };
layout(set=0,binding=2,std430) readonly buffer Predicate { uint predicate_words[]; };
layout(set=0,binding=3,std430) buffer Control { uint control[]; };
layout(set=0,binding=4,std430) readonly buffer States { uvec2 states[]; };
layout(push_constant) uniform Params {
  uint kind;
  uint primary_word_count;
  uint count_source;
  uint predicate_source;
  uint has_count;
  uint has_predicate;
  uint iteration;
  uint count_word_offset;
  uint predicate_word_offset;
  uint indirect_dispatch_count;
  uint declared_step;
  uint state;
  uint64_t capacity;
  uint64_t predicate_expected;
  uint64_t work_item_count;
} p;
uint64_t pair64(uint lo, uint hi) {
  return uint64_t(lo) | (uint64_t(hi) << 32u);
}
uint64_t count_scalar(uint at, uint source) {
  return source == 2u ? pair64(count_words[at], count_words[at + 1u])
                      : uint64_t(count_words[at]);
}
uint64_t predicate_scalar(uint at, uint source) {
  return source == 2u
      ? pair64(predicate_words[at], predicate_words[at + 1u])
      : uint64_t(predicate_words[at]);
}
uint64_t load_control(uint at) {
  return pair64(control[at], control[at + 1u]);
}
void store_control(uint at, uint64_t value) {
  control[at] = uint(value);
  control[at + 1u] = uint(value >> 32u);
}
uint64_t saturating_add(uint64_t left, uint64_t right) {
  const uint64_t maximum = uint64_t(0xfffffffffffffffful);
  return right > maximum - left ? maximum : left + right;
}
uint64_t saturating_multiply_u32(uint64_t left, uint right) {
  const uint64_t maximum = uint64_t(0xfffffffffffffffful);
  const uint64_t low_product = uint64_t(uint(left)) * uint64_t(right);
  const uint64_t high_product =
      uint64_t(uint(left >> 32u)) * uint64_t(right);
  if (high_product > uint64_t(0xffffffffu)) { return maximum; }
  const uint64_t upper = high_product << 32u;
  return low_product > maximum - upper ? maximum : upper + low_product;
}
void add_control(uint at, uint64_t value) {
  store_control(at, saturating_add(load_control(at), value));
}
void main() {
  if (p.state != 0xffffffffu && states[p.state].y != 0u) {
    return;
  }
  if (control[1] != 0u && control[2] != p.declared_step) {
    return;
  }
  if (p.kind == 2u) {
    if (p.primary_word_count >= 4u) {
      add_control(16u, uint64_t(primary[2]));
      if (primary[0] != 0u) {
        store_control(18u, min(load_control(18u), uint64_t(primary[1])));
      } else {
        const uint64_t logical = uint64_t(primary[1]);
        add_control(4u, logical);
        add_control(6u, p.capacity);
        add_control(8u, uint64_t(p.indirect_dispatch_count));
        add_control(10u, saturating_add(logical, p.work_item_count));
      }
    }
    return;
  }
  if (p.kind == 4u) {
    if (primary[0] != 0u) {
      store_control(18u, min(load_control(18u), uint64_t(primary[1])));
    } else {
      const uint64_t logical = uint64_t(primary[1]);
      add_control(4u, logical);
      add_control(6u, p.capacity);
      add_control(8u, uint64_t(p.indirect_dispatch_count));
      add_control(10u, logical);
    }
    return;
  }
  if (p.kind == 3u) {
    const uint64_t logical =
        count_scalar(p.count_word_offset, p.count_source);
    add_control(4u, logical);
    add_control(6u, p.capacity);
    if (logical > p.capacity) {
      store_control(18u, min(load_control(18u), p.capacity));
      return;
    }
    if (p.iteration != 0u) {
      if (logical == 0u) {
        add_control(14u, 1u);
        return;
      }
      add_control(12u, 1u);
    }
    add_control(8u, uint64_t(p.indirect_dispatch_count));
    add_control(10u,
                saturating_multiply_u32(logical,
                                        p.indirect_dispatch_count));
    return;
  }
  if (p.kind != 1u || p.primary_word_count == 0u ||
      (p.primary_word_count & 3u) != 0u) { return; }
  const uint64_t logical = p.has_count != 0u
      ? count_scalar(p.count_word_offset, p.count_source)
      : p.capacity;
  if (p.has_predicate != 0u) {
    const uint64_t predicate = predicate_scalar(
        p.predicate_word_offset, p.predicate_source);
    if (predicate != p.predicate_expected) {
      add_control(14u, 1u);
      return;
    }
    add_control(12u, 1u);
  }
  if (p.has_count == 0u) { return; }
  add_control(4u, logical);
  add_control(6u, p.capacity);
  if (logical > p.capacity) {
    store_control(18u, min(load_control(18u), p.capacity));
    return;
  }
  if (p.iteration != 0u && p.has_predicate == 0u) {
    if (logical == 0u) {
      add_control(14u, 1u);
      return;
    }
    add_control(12u, 1u);
  }
  add_control(8u, 1u);
  uint64_t active_count = 0u;
  for (uint index = 3u; index < p.primary_word_count; index += 4u) {
    active_count =
        saturating_add(active_count, uint64_t(primary[index]));
  }
  add_control(10u, active_count);
}
)GLSL";
}

namespace {

struct ProfileSourceEdit final {
  std::size_t begin{};
  std::size_t end{};
  std::string_view replacement{};

  [[nodiscard]] std::string_view text() const noexcept { return replacement; }
};

inline constexpr std::array<std::string_view, 4u> ProfileFrom{
    "layout(set=0,binding=3,std430) buffer Control { uint control[]; };\n",
    "  uint64_t work_item_count;\n} p;\n",
    "uint64_t load_control(uint at) {\n"
    "  return pair64(control[at], control[at + 1u]);\n"
    "}\n"
    "void store_control(uint at, uint64_t value) {\n"
    "  control[at] = uint(value);\n"
    "  control[at + 1u] = uint(value >> 32u);\n"
    "}\n",
    "void add_control(uint at, uint64_t value) {\n"
    "  store_control(at, saturating_add(load_control(at), value));\n"
    "}\n",
};

inline constexpr std::array<std::string_view, 4u> ProfileTo{
    "layout(set=0,binding=3,std430) buffer Control { uint control[]; };\n"
    "layout(set=0,binding=5,std430) buffer StepControl { uint "
    "step_control[]; };\n",
    "  uint64_t work_item_count;\n"
    "  uint declared_step_count;\n"
    "} p;\n",
    "uint step_word(uint at) {\n"
    "  const uint field = (at - 4u) >> 1u;\n"
    "  return 2u * (field * p.declared_step_count + p.declared_step);\n"
    "}\n"
    "uint64_t load_summary(uint at) {\n"
    "  return pair64(control[at], control[at + 1u]);\n"
    "}\n"
    "void store_summary(uint at, uint64_t value) {\n"
    "  control[at] = uint(value);\n"
    "  control[at + 1u] = uint(value >> 32u);\n"
    "}\n"
    "uint64_t load_control(uint at) {\n"
    "  const uint word = step_word(at);\n"
    "  return pair64(step_control[word], step_control[word + 1u]);\n"
    "}\n"
    "void store_step(uint at, uint64_t value) {\n"
    "  const uint word = step_word(at);\n"
    "  step_control[word] = uint(value);\n"
    "  step_control[word + 1u] = uint(value >> 32u);\n"
    "}\n"
    "void store_control(uint at, uint64_t value) {\n"
    "  store_step(at, value);\n"
    "  store_summary(at, at == 18u\n"
    "                        ? min(load_summary(at), value)\n"
    "                        : value);\n"
    "}\n",
    "void add_control(uint at, uint64_t value) {\n"
    "  store_summary(at, saturating_add(load_summary(at), value));\n"
    "  store_step(at, saturating_add(load_control(at), value));\n"
    "}\n",
};

[[nodiscard]] bool PlanProfileSourceEdits(
    std::array<ProfileSourceEdit, ProfileFrom.size()> &edits) noexcept {
  const std::string_view source = VulkanTelemetrySourceText();
  for (std::size_t index = 0u; index < edits.size(); ++index) {
    const std::size_t at = source.find(ProfileFrom[index]);
    if (at == std::string_view::npos ||
        source.find(ProfileFrom[index], at + ProfileFrom[index].size()) !=
            std::string_view::npos) {
      return false;
    }
    edits[index] = ProfileSourceEdit{
        .begin = at,
        .end = at + ProfileFrom[index].size(),
        .replacement = ProfileTo[index],
    };
  }
  return backend_source_recipe::canonicalize_edits(
      std::span<ProfileSourceEdit>{edits}, source.size());
}

} // namespace

std::uint64_t VulkanProfileSourceBytes() noexcept {
  std::array<ProfileSourceEdit, ProfileFrom.size()> edits{};
  if (!PlanProfileSourceEdits(edits)) {
    return 0u;
  }
  const backend_source_recipe::BasicSourceEditRecipe<ProfileSourceEdit> recipe{
      VulkanTelemetrySourceText(), std::span<const ProfileSourceEdit>{edits}};
  std::uint64_t bytes = 0u;
  return backend_source_recipe::bytes(recipe, bytes) ? bytes : 0u;
}

std::string VulkanProfileSource() {
  std::array<ProfileSourceEdit, ProfileFrom.size()> edits{};
  if (!PlanProfileSourceEdits(edits)) {
    return {};
  }
  const backend_source_recipe::BasicSourceEditRecipe<ProfileSourceEdit> recipe{
      VulkanTelemetrySourceText(), std::span<const ProfileSourceEdit>{edits}};
  std::uint64_t bytes = 0u;
  if (!backend_source_recipe::bytes(recipe, bytes)) {
    return {};
  }
  return backend_source_recipe::materialize(recipe, bytes);
}

[[nodiscard]] rund::kernel::ComputePlan VulkanTelemetryPlan() noexcept {
  return rund::kernel::ComputePlan{
      .op_hash_hi = 0x706970652e74656cull,
      .op_hash_lo = 0x656d657472792e31ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] rund::kernel::ComputePlan VulkanProfilePlan() noexcept {
  return rund::kernel::ComputePlan{
      .op_hash_hi = 0x706970652e70726full,
      .op_hash_lo = 0x66696c652e763100ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

#endif

} // namespace rund::node::accel::detail
