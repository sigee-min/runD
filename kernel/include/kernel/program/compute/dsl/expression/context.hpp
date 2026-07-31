#pragma once

#include <kernel/program/compute/dsl/support.hpp>

#include <memory>
#include <string_view>
#include <vector>

namespace rund::compute_dsl::detail {

class BuildContext {
public:
  explicit BuildContext(
      const std::vector<BindingRuntime> &bindings,
      ScalarMode scalar_mode = ScalarMode::Unspecified,
      rund::kernel::ComputeFixedFormat fixed_format = {}) noexcept;

  BuildContext(const BuildContext &) = delete;
  BuildContext &operator=(const BuildContext &) = delete;
  BuildContext(BuildContext &&) = delete;
  BuildContext &operator=(BuildContext &&) = delete;

  [[nodiscard]] rund::kernel::u32 param_node(std::string_view name) noexcept;
  [[nodiscard]] rund::kernel::u32 read_node(rund::kernel::u32 binding) noexcept;
  [[nodiscard]] rund::kernel::u32
  read_node(rund::kernel::u32 binding,
            rund::kernel::ComputeFixedFormat format) noexcept;
  [[nodiscard]] rund::kernel::u32
  read_at_node(rund::kernel::u32 binding, rund::kernel::u32 index,
               rund::kernel::u32 count,
               rund::kernel::ComputeFixedFormat format) noexcept;
  [[nodiscard]] rund::kernel::u32 binary_node(rund::kernel::IrOp op,
                                              rund::kernel::u32 lhs,
                                              rund::kernel::u32 rhs) noexcept;
  [[nodiscard]] rund::kernel::u32
  binary_node(rund::kernel::IrOp op, rund::kernel::u32 lhs,
              rund::kernel::u32 rhs,
              rund::kernel::ComputeFixedFormat format) noexcept;
  [[nodiscard]] rund::kernel::u32 unary_node(rund::kernel::IrOp op,
                                             rund::kernel::u32 lhs) noexcept;
  [[nodiscard]] rund::kernel::u32
  unary_node(rund::kernel::IrOp op, rund::kernel::u32 lhs,
             rund::kernel::ComputeFixedFormat format) noexcept;
  [[nodiscard]] rund::kernel::u32
  storage_quantize_node(rund::kernel::u32 lhs) noexcept;
  [[nodiscard]] rund::kernel::u32
  const_shift_node(rund::kernel::IrOp op, rund::kernel::u32 lhs,
                   rund::kernel::u32 amount) noexcept;
  [[nodiscard]] rund::kernel::u32 ternary_node(rund::kernel::IrOp op,
                                               rund::kernel::u32 lhs,
                                               rund::kernel::u32 rhs,
                                               rund::kernel::u32 aux) noexcept;
  [[nodiscard]] rund::kernel::u32
  ternary_node(rund::kernel::IrOp op, rund::kernel::u32 lhs,
               rund::kernel::u32 rhs, rund::kernel::u32 aux,
               rund::kernel::ComputeFixedFormat format) noexcept;
  [[nodiscard]] rund::kernel::u32
  constant_node(rund::kernel::u64 bits) noexcept;
  [[nodiscard]] rund::kernel::u32
  constant_node(rund::kernel::u64 bits, rund::kernel::u32 anchor) noexcept;
  [[nodiscard]] rund::kernel::u32
  constant_node(rund::kernel::u64 bits, rund::kernel::u32 anchor,
                rund::kernel::ComputeFixedFormat format) noexcept;
  [[nodiscard]] rund::kernel::u32 typed_constant_node(rund::kernel::u64 bits,
                                                      ScalarMode mode) noexcept;
  [[nodiscard]] rund::kernel::u32
  storage_constant_node(rund::kernel::u64 bits,
                        rund::kernel::u32 anchor) noexcept;
  [[nodiscard]] rund::kernel::u32 index_node() noexcept;
  [[nodiscard]] rund::kernel::u32 index_node(ScalarMode mode) noexcept;

  void write_node(rund::kernel::u32 binding, rund::kernel::u32 value) noexcept;
  void write_checked_ordinal_node(rund::kernel::u32 binding,
                                  rund::kernel::u32 value) noexcept;
  void write_boundary_mask_node(
      rund::kernel::u32 binding, rund::kernel::u32 value,
      rund::kernel::ComputeFixedFormat target_format) noexcept;

  void reject(const char *reason) noexcept;
  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] const char *reason() const noexcept;
  [[nodiscard]] rund::kernel::u32 write_count() const noexcept;
  [[nodiscard]] const std::vector<rund::kernel::ComputeIrNode> &
  nodes() const noexcept;
  [[nodiscard]] bool valid_node(rund::kernel::u32 node) const noexcept;
  [[nodiscard]] bool valid_binding(rund::kernel::u32 binding,
                                   BindingKind kind) const noexcept;
  [[nodiscard]] ScalarMode scalar_mode() const noexcept;
  [[nodiscard]] ScalarMode
  scalar_mode_node(rund::kernel::u32 node) const noexcept;
  [[nodiscard]] rund::kernel::ComputeFixedFormat
  fixed_format_node(rund::kernel::u32 node) const noexcept;
  [[nodiscard]] rund::kernel::u32 binding_index(std::string_view name,
                                                BindingKind kind) noexcept;

private:
  struct ContextLifetime {};

  static constexpr rund::kernel::u32 kInvalidBinding = ~rund::kernel::u32{0u};

  [[nodiscard]] bool fixed_mode() const noexcept;
  [[nodiscard]] rund::kernel::ComputeDomain domain() const noexcept;
  [[nodiscard]] rund::kernel::ComputeDomain
  binding_domain(rund::kernel::u32 binding) const noexcept;
  [[nodiscard]] bool
  binding_value_mode_valid(rund::kernel::u32 binding) const noexcept;
  [[nodiscard]] rund::kernel::ComputeDomain
  node_domain(rund::kernel::u32 node) const noexcept;
  [[nodiscard]] rund::kernel::ComputeDomain
  unary_node_domain(rund::kernel::u32 node) const noexcept;
  [[nodiscard]] rund::kernel::ComputeDomain
  merge_domains(rund::kernel::ComputeDomain lhs,
                rund::kernel::ComputeDomain rhs) noexcept;
  [[nodiscard]] rund::kernel::ComputeDomain
  merged_node_domain(rund::kernel::u32 lhs, rund::kernel::u32 rhs) noexcept;
  [[nodiscard]] rund::kernel::ComputeDomain
  ternary_node_domain(rund::kernel::IrOp op, rund::kernel::u32 lhs,
                      rund::kernel::u32 rhs, rund::kernel::u32 aux) noexcept;
  [[nodiscard]] static bool storage_unary(rund::kernel::IrOp op) noexcept;
  [[nodiscard]] static bool storage_binary(rund::kernel::IrOp op) noexcept;
  [[nodiscard]] static bool approximate_unary(rund::kernel::IrOp op) noexcept;
  [[nodiscard]] static bool approximate_binary(rund::kernel::IrOp op) noexcept;
  [[nodiscard]] bool
  canonical_mask_source(rund::kernel::u32 source) const noexcept;
  [[nodiscard]] bool canonical_zero(rund::kernel::u32 source) const noexcept;
  [[nodiscard]] rund::kernel::ComputeDomain checked_ordinal_source_domain(
      rund::kernel::u32 source,
      rund::kernel::ComputeDomain target_domain) const noexcept;
  [[nodiscard]] rund::kernel::ComputeDomain
  boundary_mask_source_domain(rund::kernel::u32 source) const noexcept;
  [[nodiscard]] rund::kernel::u32 storage_node(rund::kernel::u32 node) noexcept;
  [[nodiscard]] rund::kernel::ComputeFixedFormat value_format() const noexcept;
  [[nodiscard]] rund::kernel::ComputeFixedFormat
  source_format(rund::kernel::u32 node) const noexcept;
  [[nodiscard]] rund::kernel::ComputeFixedFormat
  unary_format(rund::kernel::IrOp op, rund::kernel::u32 source) noexcept;
  [[nodiscard]] rund::kernel::ComputeFixedFormat
  binary_format(rund::kernel::IrOp op, rund::kernel::u32 lhs,
                rund::kernel::u32 rhs) noexcept;
  [[nodiscard]] rund::kernel::ComputeFixedFormat
  ternary_format(rund::kernel::IrOp op, rund::kernel::u32 lhs,
                 rund::kernel::u32 rhs, rund::kernel::u32 aux) noexcept;
  [[nodiscard]] std::weak_ptr<ContextLifetime> lifetime() const noexcept;
  [[nodiscard]] rund::kernel::u32
  append_node(rund::kernel::IrOp op, rund::kernel::u32 lhs,
              rund::kernel::u32 rhs, rund::kernel::u32 aux,
              rund::kernel::ComputeFixedFormat format,
              rund::kernel::ComputeDomain value_domain) noexcept;
  [[nodiscard]] rund::kernel::u32
  find_node(rund::kernel::IrOp op, rund::kernel::u32 lhs, rund::kernel::u32 rhs,
            rund::kernel::u32 aux, rund::kernel::ComputeFixedFormat format,
            rund::kernel::ComputeDomain value_domain) const noexcept;

  const std::vector<BindingRuntime> *bindings_ = nullptr;
  ScalarMode scalar_mode_ = ScalarMode::Unspecified;
  rund::kernel::ComputeFixedFormat fixed_format_{};
  std::shared_ptr<ContextLifetime> lifetime_;
  std::vector<rund::kernel::ComputeIrNode> nodes_;
  std::vector<rund::kernel::ComputeDomain> node_domains_;
  bool ok_ = true;
  const char *reason_ = "ok";
  rund::kernel::u32 write_count_ = 0u;

  friend class Expr;
  friend class ReadHandle;
  friend class WriteHandle;
  friend class WriteTarget;
  template <typename Body> friend class Access;
};

} // namespace rund::compute_dsl::detail
