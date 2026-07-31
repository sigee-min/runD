#include <kernel/program/compute/dsl/expression/context.hpp>

namespace rund::compute_dsl::detail {

BuildContext::BuildContext(
    const std::vector<BindingRuntime> &bindings, const ScalarMode scalar_mode,
    const rund::kernel::ComputeFixedFormat fixed_format) noexcept
    : bindings_(&bindings), scalar_mode_(scalar_mode),
      fixed_format_(fixed_format),
      lifetime_(std::make_shared<ContextLifetime>()) {}

void BuildContext::reject(const char *const reason) noexcept {
  if (ok_) {
    ok_ = false;
    reason_ = reason;
  }
}

bool BuildContext::ok() const noexcept { return ok_; }

const char *BuildContext::reason() const noexcept { return reason_; }

rund::kernel::u32 BuildContext::write_count() const noexcept {
  return write_count_;
}

const std::vector<rund::kernel::ComputeIrNode> &
BuildContext::nodes() const noexcept {
  return nodes_;
}

bool BuildContext::valid_node(const rund::kernel::u32 node) const noexcept {
  return node > 0u && node <= nodes_.size();
}

bool BuildContext::valid_binding(const rund::kernel::u32 binding,
                                 const BindingKind kind) const noexcept {
  return bindings_ != nullptr && binding < bindings_->size() &&
         (*bindings_)[binding].kind == kind;
}

ScalarMode BuildContext::scalar_mode() const noexcept { return scalar_mode_; }

ScalarMode
BuildContext::scalar_mode_node(const rund::kernel::u32 node) const noexcept {
  const auto value_domain = node_domain(node);
  switch (value_domain) {
  case rund::kernel::ComputeDomain::I32:
    return ScalarMode::I32;
  case rund::kernel::ComputeDomain::U32:
    return ScalarMode::U32;
  case rund::kernel::ComputeDomain::I64:
    return ScalarMode::I64;
  case rund::kernel::ComputeDomain::U64:
    return ScalarMode::U64;
  case rund::kernel::ComputeDomain::Fixed:
    return WideMode(scalar_mode_) ? ScalarMode::FixedLane64
                                  : ScalarMode::FixedLane32;
  }
  return ScalarMode::Unspecified;
}

rund::kernel::ComputeFixedFormat
BuildContext::fixed_format_node(const rund::kernel::u32 node) const noexcept {
  return fixed_mode() && valid_node(node) ? source_format(node)
                                          : rund::kernel::ComputeFixedFormat{};
}

rund::kernel::u32 BuildContext::binding_index(const std::string_view name,
                                              const BindingKind kind) noexcept {
  if (bindings_ == nullptr) {
    reject("compute_binding_missing");
    return 0u;
  }
  for (std::size_t index = 0u; index < bindings_->size(); ++index) {
    const BindingRuntime &binding = (*bindings_)[index];
    if (binding.kind == kind && binding.name == name) {
      return static_cast<rund::kernel::u32>(index);
    }
  }
  reject("compute_binding_missing");
  return kInvalidBinding;
}

bool BuildContext::fixed_mode() const noexcept {
  return scalar_mode_ == ScalarMode::FixedLane32 ||
         scalar_mode_ == ScalarMode::FixedLane64;
}

rund::kernel::ComputeDomain BuildContext::domain() const noexcept {
  return ToComputeDomain(scalar_mode_);
}

rund::kernel::ComputeDomain
BuildContext::binding_domain(const rund::kernel::u32 binding) const noexcept {
  return bindings_ != nullptr && binding < bindings_->size()
             ? ToComputeDomain((*bindings_)[binding].numeric_mode)
             : static_cast<rund::kernel::ComputeDomain>(0u);
}

bool BuildContext::binding_value_mode_valid(
    const rund::kernel::u32 binding) const noexcept {
  if (bindings_ == nullptr || binding >= bindings_->size()) {
    return false;
  }
  if (scalar_mode_ == ScalarMode::Unspecified) {
    return true;
  }
  const ScalarMode binding_mode = (*bindings_)[binding].numeric_mode;
  const auto value_domain = ToComputeDomain(binding_mode);
  if (!NumericMode(binding_mode) ||
      WideMode(binding_mode) != WideMode(scalar_mode_)) {
    return false;
  }
  return fixed_mode() ? value_domain == rund::kernel::ComputeDomain::Fixed
                      : value_domain != rund::kernel::ComputeDomain::Fixed;
}

rund::kernel::ComputeDomain
BuildContext::node_domain(const rund::kernel::u32 node) const noexcept {
  return valid_node(node) && node <= node_domains_.size()
             ? node_domains_[node - 1u]
             : static_cast<rund::kernel::ComputeDomain>(0u);
}

rund::kernel::ComputeDomain
BuildContext::unary_node_domain(const rund::kernel::u32 node) const noexcept {
  const auto value_domain = node_domain(node);
  return value_domain == static_cast<rund::kernel::ComputeDomain>(0u)
             ? domain()
             : value_domain;
}

rund::kernel::ComputeDomain
BuildContext::merge_domains(const rund::kernel::ComputeDomain lhs,
                            const rund::kernel::ComputeDomain rhs) noexcept {
  const auto unknown = static_cast<rund::kernel::ComputeDomain>(0u);
  const auto merged = rund::kernel::MergeComputeDomains(lhs, rhs);
  if (lhs != unknown && rhs != unknown && merged == unknown) {
    reject("compute_value_invalid");
    return unknown;
  }
  return merged;
}

rund::kernel::ComputeDomain
BuildContext::merged_node_domain(const rund::kernel::u32 lhs,
                                 const rund::kernel::u32 rhs) noexcept {
  const auto merged = merge_domains(node_domain(lhs), node_domain(rhs));
  return merged == static_cast<rund::kernel::ComputeDomain>(0u) ? domain()
                                                                : merged;
}

rund::kernel::ComputeDomain BuildContext::ternary_node_domain(
    const rund::kernel::IrOp op, const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs, const rund::kernel::u32 aux) noexcept {
  auto value_domain = op == rund::kernel::IrOp::Select
                          ? merge_domains(node_domain(rhs), node_domain(aux))
                          : merge_domains(node_domain(lhs), node_domain(rhs));
  if (op != rund::kernel::IrOp::Select) {
    value_domain = merge_domains(value_domain, node_domain(aux));
  }
  return value_domain == static_cast<rund::kernel::ComputeDomain>(0u)
             ? domain()
             : value_domain;
}

bool BuildContext::storage_unary(const rund::kernel::IrOp op) noexcept {
  return op == rund::kernel::IrOp::BitNot ||
         op == rund::kernel::IrOp::NegPositiveFixed ||
         op == rund::kernel::IrOp::Recip || op == rund::kernel::IrOp::Sqrt ||
         op == rund::kernel::IrOp::Rsqrt || op == rund::kernel::IrOp::Sin ||
         op == rund::kernel::IrOp::Cos || op == rund::kernel::IrOp::Tan ||
         op == rund::kernel::IrOp::Exp || op == rund::kernel::IrOp::Log;
}

bool BuildContext::storage_binary(const rund::kernel::IrOp op) noexcept {
  return op == rund::kernel::IrOp::MulWrap ||
         op == rund::kernel::IrOp::BitAnd || op == rund::kernel::IrOp::BitOr ||
         op == rund::kernel::IrOp::BitXor || op == rund::kernel::IrOp::AddSat ||
         op == rund::kernel::IrOp::AddSatUnsigned ||
         op == rund::kernel::IrOp::SubSat ||
         op == rund::kernel::IrOp::MulFixed ||
         op == rund::kernel::IrOp::MulFixedScaled ||
         op == rund::kernel::IrOp::MulUnsignedFixed ||
         op == rund::kernel::IrOp::DivFixed || op == rund::kernel::IrOp::Atan2;
}

bool BuildContext::approximate_unary(const rund::kernel::IrOp op) noexcept {
  return op == rund::kernel::IrOp::Recip || op == rund::kernel::IrOp::Sqrt ||
         op == rund::kernel::IrOp::Rsqrt || op == rund::kernel::IrOp::Sin ||
         op == rund::kernel::IrOp::Cos || op == rund::kernel::IrOp::Tan ||
         op == rund::kernel::IrOp::Exp || op == rund::kernel::IrOp::Log;
}

bool BuildContext::approximate_binary(const rund::kernel::IrOp op) noexcept {
  return op == rund::kernel::IrOp::DivFixed || op == rund::kernel::IrOp::Atan2;
}

std::weak_ptr<BuildContext::ContextLifetime>
BuildContext::lifetime() const noexcept {
  return lifetime_;
}

} // namespace rund::compute_dsl::detail
