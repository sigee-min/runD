#include "source.hpp"

#include <kernel/program/compute/lowering/layout.hpp>
#include <kernel/program/compute/lowering/metal/syntax.hpp>
#include <kernel/program/compute/lowering/text.hpp>
#include <kernel/program/compute/lowering/vulkan/syntax.hpp>

#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace rund::node::accel::detail {
using rund::kernel::ArtifactKey;
using rund::kernel::ComputeApi;
using rund::kernel::ComputeBindingAccess;
using rund::kernel::ComputeScalar;
using rund::kernel::ExecutionMetadata;
using rund::kernel::LoweringArtifact;

namespace {

struct SourceBinding final {
  std::string symbol;
  std::uint64_t element_bytes{};
  bool uniform{};
};

[[nodiscard]] bool ReplaceOne(std::string &source,
                              const std::string_view needle,
                              const std::string_view replacement) {
  if (needle.empty()) {
    return false;
  }
  const std::size_t at = source.find(needle);
  if (at == std::string::npos ||
      source.find(needle, at + needle.size()) != std::string::npos) {
    return false;
  }
  source.replace(at, needle.size(), replacement);
  return true;
}

[[nodiscard]] std::string HexDigits(const std::uint64_t value) {
  std::string out;
  rund::kernel::compute_lowering_detail::AppendHex64Digits(out, value);
  return out;
}

[[nodiscard]] ArtifactKey RecurrenceKey(ArtifactKey source) noexcept {
  ArtifactKey key = source;
  // Variant is an orthogonal executable-identity dimension. Canonical graph
  // and operation hashes remain the sole semantic graph identity.
  key.variant = rund::kernel::LoweringArtifactVariant::Recurrence;
  return key;
}

[[nodiscard]] bool RewriteKey(std::string &source, const ArtifactKey &before,
                              const ArtifactKey &after) {
  if (before.variant != rund::kernel::LoweringArtifactVariant::Canonical ||
      after.variant != rund::kernel::LoweringArtifactVariant::Recurrence ||
      !ReplaceOne(source, "// artifact_variant=canonical",
                  "// artifact_variant=recurrence")) {
    return false;
  }
  if (before.api != ComputeApi::Metal) {
    return true;
  }
  const std::string old_name = "rund_compute_map_" +
                               HexDigits(before.op_hash_hi) + "_" +
                               HexDigits(before.op_hash_lo);
  const std::string new_name = "rund_compute_map_" +
                               HexDigits(after.op_hash_hi) + "_" +
                               HexDigits(after.op_hash_lo) + "_recurrence";
  return ReplaceOne(source, old_name, new_name);
}

[[nodiscard]] bool SourceBindings(const ExecutionMetadata &metadata,
                                  std::vector<SourceBinding> &inputs,
                                  std::vector<SourceBinding> &outputs) {
  if (!metadata.ok ||
      metadata.binding_accesses.size() != metadata.binding_names.size() ||
      metadata.input_element_bytes.size() != metadata.read_count ||
      metadata.output_element_bytes.size() != metadata.write_count) {
    return false;
  }
  inputs.reserve(static_cast<std::size_t>(metadata.read_count));
  outputs.reserve(static_cast<std::size_t>(metadata.write_count));
  std::size_t read = 0u;
  std::size_t write = 0u;
  for (std::size_t index = 0u; index < metadata.binding_accesses.size();
       ++index) {
    const ComputeBindingAccess access = metadata.binding_accesses[index];
    const std::string safe =
        rund::kernel::compute_lowering_detail::SafeIdentifier(
            metadata.binding_names[index]);
    if (access == ComputeBindingAccess::Read) {
      if (read >= metadata.input_element_bytes.size() || read >= 64u) {
        return false;
      }
      const std::uint64_t bit = std::uint64_t{1u} << read;
      const bool direct = (metadata.direct_read_mask & bit) != 0u;
      const bool uniform = (metadata.uniform_read_mask & bit) != 0u;
      if (direct == uniform) {
        return false;
      }
      inputs.push_back(SourceBinding{
          .symbol = "read_" + safe,
          .element_bytes = metadata.input_element_bytes[read],
          .uniform = uniform,
      });
      ++read;
    } else if (access == ComputeBindingAccess::Write) {
      if (write >= metadata.output_element_bytes.size()) {
        return false;
      }
      outputs.push_back(SourceBinding{
          .symbol = "write_" + safe,
          .element_bytes = metadata.output_element_bytes[write++],
      });
    } else {
      return false;
    }
  }
  return read == metadata.input_element_bytes.size() &&
         write == metadata.output_element_bytes.size();
}

[[nodiscard]] std::string MetalLoad(const ComputeScalar scalar,
                                    const SourceBinding &binding) {
  const std::string base =
      rund::kernel::compute_lowering_detail::BindingBaseSymbol(binding.symbol);
  const std::string stride =
      rund::kernel::compute_lowering_detail::BindingStrideSymbol(
          binding.symbol);
  return std::string{
             rund::kernel::compute_lowering_detail::MetalLoadFunction(scalar)} +
         "(" + binding.symbol + ", " + base +
         (binding.uniform ? ")" : " + gid * " + stride + ")");
}

[[nodiscard]] std::string VulkanLoad(const ComputeScalar scalar,
                                     const SourceBinding &binding) {
  const std::string base =
      rund::kernel::compute_lowering_detail::BindingBaseSymbol(binding.symbol);
  const std::string stride =
      rund::kernel::compute_lowering_detail::BindingStrideSymbol(
          binding.symbol);
  return std::string{
             rund::kernel::compute_lowering_detail::VulkanLoadPrefix(scalar)} +
         "_" + binding.symbol + "(" + base +
         (binding.uniform ? ")" : " + gid * " + stride + ")");
}

[[nodiscard]] bool ReplaceOutput(std::string &source, const ComputeApi api,
                                 const ComputeScalar scalar,
                                 const SourceBinding &binding,
                                 const std::size_t ordinal,
                                 std::string &final_store) {
  const char *const store =
      api == ComputeApi::Metal
          ? rund::kernel::compute_lowering_detail::MetalStoreFunction(scalar)
          : rund::kernel::compute_lowering_detail::VulkanStorePrefix(scalar);
  const std::string base =
      rund::kernel::compute_lowering_detail::BindingBaseSymbol(binding.symbol);
  const std::string stride =
      rund::kernel::compute_lowering_detail::BindingStrideSymbol(
          binding.symbol);
  const std::string address = base + " + gid * " + stride;
  const std::string prefix = api == ComputeApi::Metal
                                 ? "  " + std::string{store} + "(" +
                                       binding.symbol + ", " + address + ", "
                                 : "  " + std::string{store} + "_" +
                                       binding.symbol + "(" + address + ", ";
  const std::size_t begin = source.find(prefix);
  if (begin == std::string::npos ||
      source.find(prefix, begin + prefix.size()) != std::string::npos) {
    return false;
  }
  const std::size_t value_begin = begin + prefix.size();
  const std::size_t end = source.find(");\n", value_begin);
  if (end == std::string::npos) {
    return false;
  }
  const std::string value = source.substr(value_begin, end - value_begin);
  source.replace(begin, end + 3u - begin,
                 "    rund_next_" + std::to_string(ordinal) + " = " + value +
                     ";\n");
  final_store =
      api == ComputeApi::Metal
          ? "  " + std::string{store} + "(" + binding.symbol + ", " + address +
                ", rund_carry_" + std::to_string(ordinal) + ");\n"
          : "  " + std::string{store} + "_" + binding.symbol + "(" + address +
                ", rund_carry_" + std::to_string(ordinal) + ");\n";
  return true;
}

} // namespace

[[nodiscard]] bool TransformSource(LoweringArtifact &artifact,
                                   const std::uint64_t input_count,
                                   const std::uint64_t output_count) {
  const ComputeApi api = artifact.key.api;
  if ((api != ComputeApi::Metal && api != ComputeApi::Vulkan) ||
      output_count == 0u || output_count > input_count ||
      input_count > std::numeric_limits<std::uint32_t>::max() ||
      output_count > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  std::vector<SourceBinding> inputs;
  std::vector<SourceBinding> outputs;
  if (!SourceBindings(artifact.metadata, inputs, outputs) ||
      inputs.size() != input_count || outputs.size() != output_count) {
    return false;
  }
  const std::uint64_t scalar_bytes =
      rund::kernel::ComputeScalarBits(artifact.key.scalar) / 8u;
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    if (inputs[index].uniform || inputs[index].element_bytes != scalar_bytes ||
        outputs[index].element_bytes != scalar_bytes) {
      return false;
    }
  }

  const ArtifactKey source_key = artifact.key;
  const ArtifactKey recurrence_key = RecurrenceKey(source_key);
  if (!RewriteKey(artifact.source_text, source_key, recurrence_key)) {
    return false;
  }

  std::string body_anchor;
  std::string loop_count;
  if (api == ComputeApi::Metal) {
    const std::string gid = "    uint gid [[thread_position_in_grid]]) {\n";
    const std::string with_iterations =
        "    constant uint& rund_iterations [[buffer(" +
        std::to_string(input_count + output_count + 1u) + ")]],\n" + gid;
    if (!ReplaceOne(artifact.source_text, gid, with_iterations)) {
      return false;
    }
    body_anchor = with_iterations;
    loop_count = "rund_iterations";
  } else {
    const std::string dispatch =
        "layout(push_constant) uniform RundDispatch {\n"
        "  uint tile_count;\n"
        "  uint iterations;\n"
        "} rund_dispatch;\n";
    const std::size_t declaration = artifact.source_text.find(dispatch);
    if (declaration == std::string::npos ||
        artifact.source_text.find(dispatch, declaration + dispatch.size()) !=
            std::string::npos) {
      return false;
    }
    body_anchor = "  if (gid >= rund_dispatch.tile_count) { return; }\n";
    loop_count = "rund_dispatch.iterations";
  }

  const char *const scalar_type =
      api == ComputeApi::Metal
          ? rund::kernel::compute_lowering_detail::MetalType(
                artifact.key.scalar)
          : rund::kernel::compute_lowering_detail::VulkanType(
                artifact.key.scalar);
  std::string prelude;
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    const std::string load =
        api == ComputeApi::Metal
            ? MetalLoad(artifact.key.scalar, inputs[index])
            : VulkanLoad(artifact.key.scalar, inputs[index]);
    const std::string value = index < output_count
                                  ? "rund_carry_" + std::to_string(index)
                                  : "rund_invariant_" + std::to_string(index);
    if (!ReplaceOne(artifact.source_text, load, value)) {
      return false;
    }
    prelude += "  ";
    prelude += scalar_type;
    prelude += " ";
    prelude += value;
    prelude += " = ";
    prelude += load;
    prelude += ";\n";
  }
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    prelude += "  ";
    prelude += scalar_type;
    prelude += " rund_next_";
    prelude += std::to_string(index);
    prelude += " = rund_carry_";
    prelude += std::to_string(index);
    prelude += ";\n";
  }
  prelude += "  for (uint rund_iteration = 0u; rund_iteration < ";
  prelude += loop_count;
  prelude += "; ++rund_iteration) {\n";

  std::vector<std::string> final_stores(outputs.size());
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    if (!ReplaceOutput(artifact.source_text, api, artifact.key.scalar,
                       outputs[index], index, final_stores[index])) {
      return false;
    }
  }

  const std::size_t body = artifact.source_text.find(body_anchor);
  if (body == std::string::npos ||
      artifact.source_text.find(body_anchor, body + body_anchor.size()) !=
          std::string::npos) {
    return false;
  }
  artifact.source_text.insert(body + body_anchor.size(), prelude);

  const std::size_t end = artifact.source_text.rfind("}\n");
  if (end == std::string::npos || end < body) {
    return false;
  }
  std::string epilogue;
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    epilogue += "    rund_carry_" + std::to_string(index) + " = rund_next_" +
                std::to_string(index) + ";\n";
  }
  epilogue += "  }\n";
  for (const std::string &store : final_stores) {
    epilogue += store;
  }
  artifact.source_text.insert(end, epilogue);

  artifact.key = recurrence_key;
  artifact.metadata.map.op_hash_hi = recurrence_key.op_hash_hi;
  artifact.metadata.map.op_hash_lo = recurrence_key.op_hash_lo;
  return true;
}
} // namespace rund::node::accel::detail
