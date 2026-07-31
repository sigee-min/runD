#include <kernel/program/compute/dsl/build.hpp>

#include <utility>

namespace rund::compute_dsl::detail {
namespace {

void AppendU8(std::vector<rund::kernel::u8> &bytes,
              const rund::kernel::u8 value) {
  bytes.push_back(value);
}

void AppendU32(std::vector<rund::kernel::u8> &bytes,
               const rund::kernel::u32 value) {
  for (std::size_t index = 0u; index < sizeof(rund::kernel::u32); ++index) {
    bytes.push_back(
        static_cast<rund::kernel::u8>((value >> (index * 8u)) & 0xffu));
  }
}

void AppendBytes(std::vector<rund::kernel::u8> &bytes,
                 const std::string_view value) {
  AppendU32(bytes, static_cast<rund::kernel::u32>(value.size()));
  for (const char c : value) {
    bytes.push_back(static_cast<rund::kernel::u8>(c));
  }
}

void AppendBindings(std::vector<rund::kernel::u8> &bytes,
                    const std::vector<BindingRuntime> &bindings) {
  AppendU32(bytes, static_cast<rund::kernel::u32>(bindings.size()));
  for (const BindingRuntime &binding : bindings) {
    AppendU8(bytes, static_cast<rund::kernel::u8>(binding.kind));
    AppendU8(bytes, static_cast<rund::kernel::u8>(binding.numeric_mode));
    AppendBytes(bytes, binding.name);
    AppendU32(bytes, binding.element_bytes);
    AppendU8(bytes, binding.floating_point_param ? rund::kernel::u8{1u}
                                                 : rund::kernel::u8{0u});
    AppendU32(bytes,
              static_cast<rund::kernel::u32>(binding.value_bytes.size()));
    for (const rund::kernel::u8 byte : binding.value_bytes) {
      AppendU8(bytes, byte);
    }
  }
}

void AppendNodes(std::vector<rund::kernel::u8> &bytes,
                 const std::vector<rund::kernel::ComputeIrNode> &nodes,
                 const rund::kernel::ComputeFixedFormat graph_format) {
  AppendU32(bytes, static_cast<rund::kernel::u32>(nodes.size()));
  for (const rund::kernel::ComputeIrNode &node : nodes) {
    AppendU8(bytes, static_cast<rund::kernel::u8>(node.op));
    AppendU32(bytes, node.lhs);
    AppendU32(bytes, node.rhs);
    AppendU32(bytes, node.aux);
    const auto format =
        rund::kernel::ComputeFixedFormatAbsent(node.fixed_format)
            ? graph_format
            : node.fixed_format;
    AppendU8(bytes, format.integer_bits);
    AppendU8(bytes, format.fraction_bits);
    AppendU8(bytes, static_cast<rund::kernel::u8>(format.rounding));
    AppendU8(bytes, static_cast<rund::kernel::u8>(format.overflow));
    AppendU8(bytes, static_cast<rund::kernel::u8>(format.approximation));
  }
}

} // namespace

bool HasFloatingPointParam(
    const std::vector<BindingRuntime> &bindings) noexcept {
  for (const BindingRuntime &binding : bindings) {
    if (binding.kind == BindingKind::Param && binding.floating_point_param) {
      return true;
    }
  }
  return false;
}

rund::kernel::ComputeMap
BuildMapModel(const rund::kernel::ComputeIR &ir,
              const std::vector<BindingRuntime> &bindings) noexcept {
  rund::kernel::ComputeMap map{
      .op_hash_hi = ir.op_hash_hi,
      .op_hash_lo = ir.op_hash_lo,
      .scalar = ir.scalar,
      .domain = ir.domain,
      .fixed_format = ir.fixed_format,
      .output_buffer_count = 0u,
      .metadata_bytes_per_tile = sizeof(rund::kernel::u32),
  };
  for (const BindingRuntime &binding : bindings) {
    if (binding.kind == BindingKind::Param) {
      map.param_bytes += binding.value_bytes.size();
    } else if (binding.kind == BindingKind::Read) {
      ++map.input_buffer_count;
      map.input_bytes_per_tile += binding.element_bytes;
    } else if (binding.kind == BindingKind::Write) {
      ++map.output_buffer_count;
      map.output_bytes_per_tile += binding.element_bytes;
    }
  }
  return map;
}

rund::kernel::ComputeIR BuildCanonical(
    const std::string_view, const bool body_ok, const char *const body_reason,
    const ScalarMode mode, const rund::kernel::ComputeFixedFormat format,
    const std::vector<BindingRuntime> &bindings, const BuildContext &context) {
  const rund::kernel::ComputeScalar scalar = ToComputeScalar(mode);
  const rund::kernel::ComputeDomain domain = ToComputeDomain(mode);
  const char *reason = "ok";
  bool ok = true;

  if (!body_ok) {
    ok = false;
    reason = body_reason;
  } else if (!NumericMode(mode)) {
    ok = false;
    reason = "compute_scalar_unsupported";
  } else if (HasFloatingPointParam(bindings)) {
    ok = false;
    reason = "compute_param_float_unsupported";
  } else if (!context.ok()) {
    ok = false;
    reason = context.reason();
  } else if (context.write_count() == 0u) {
    ok = false;
    reason = "compute_write_missing";
  }

  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, std::string_view{});
  AppendU8(bytes, static_cast<rund::kernel::u8>(mode));
  AppendU8(bytes, format.integer_bits);
  AppendU8(bytes, format.fraction_bits);
  AppendU8(bytes, static_cast<rund::kernel::u8>(format.rounding));
  AppendU8(bytes, static_cast<rund::kernel::u8>(format.overflow));
  AppendU8(bytes, static_cast<rund::kernel::u8>(format.approximation));
  AppendBindings(bytes, bindings);
  AppendNodes(bytes, context.nodes(), format);

  const rund::kernel::compute_ir_detail::ComputeIrHash hash =
      rund::kernel::compute_ir_detail::HashComputeIrCanonicalBytes(
          bytes.empty() ? nullptr : bytes.data(),
          static_cast<rund::kernel::u64>(bytes.size()));

  return rund::kernel::ComputeIR{
      .scalar = scalar,
      .domain = domain,
      .fixed_format = format,
      .op_hash_hi = ok ? hash.hi : 0u,
      .op_hash_lo = ok ? hash.lo : 0u,
      .canonical_bytes = std::move(bytes),
      .ok = ok,
      .reason = reason,
  };
}

} // namespace rund::compute_dsl::detail
