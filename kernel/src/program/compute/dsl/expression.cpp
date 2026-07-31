#include <kernel/program/compute/dsl/expression/access.hpp>

namespace rund::compute_dsl::detail {
namespace {

[[nodiscard]] rund::kernel::IrOp DivisionOp(const ScalarMode mode) noexcept {
  switch (mode) {
  case ScalarMode::Unspecified:
    return static_cast<rund::kernel::IrOp>(0u);
  case ScalarMode::I32:
  case ScalarMode::I64:
    return rund::kernel::IrOp::DivSigned;
  case ScalarMode::U32:
  case ScalarMode::U64:
    return rund::kernel::IrOp::DivUnsigned;
  case ScalarMode::FixedLane32:
  case ScalarMode::FixedLane64:
    return rund::kernel::IrOp::DivFixed;
  }
  return static_cast<rund::kernel::IrOp>(0u);
}

} // namespace

Expr::Expr(InternalToken, BuildContext *const context,
           const rund::kernel::u32 node) noexcept
    : context_(context),
      lifetime_(context != nullptr
                    ? context->lifetime()
                    : std::weak_ptr<BuildContext::ContextLifetime>{}),
      node_(node) {}

BuildContext *Expr::context() const noexcept {
  return lifetime_.expired() ? nullptr : context_;
}

rund::kernel::u32 Expr::node() const noexcept { return node_; }

Expr DynamicRead(BuildContext &context,
                 const rund::kernel::u32 binding) noexcept {
  return Expr{Expr::InternalToken{}, &context, context.read_node(binding)};
}

Expr DynamicRead(BuildContext &context, const rund::kernel::u32 binding,
                 const rund::kernel::ComputeFixedFormat format) noexcept {
  return Expr{Expr::InternalToken{}, &context,
              context.read_node(binding, format)};
}

Expr DynamicReadAt(BuildContext &context, const rund::kernel::u32 binding,
                   const rund::kernel::u32 index,
                   const rund::kernel::u32 count,
                   const rund::kernel::ComputeFixedFormat format) noexcept {
  return Expr{Expr::InternalToken{}, &context,
              context.read_at_node(binding, index, count, format)};
}

Expr DynamicIndex(BuildContext &context) noexcept {
  return Expr{Expr::InternalToken{}, &context, context.index_node()};
}

Expr DynamicIndex(BuildContext &context, const ScalarMode mode) noexcept {
  return Expr{Expr::InternalToken{}, &context, context.index_node(mode)};
}

Expr TypedIndex(const Expr anchor, const ScalarMode mode) noexcept {
  BuildContext *const context = anchor.context();
  if (context == nullptr) {
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context, context->index_node(mode)};
}

void DynamicWrite(BuildContext &context, const rund::kernel::u32 binding,
                  const Expr value) noexcept {
  if (value.context() != &context) {
    context.reject("compute_expression_context_mismatch");
    return;
  }
  context.write_node(binding, value.node());
}

void DynamicCheckedOrdinalWrite(BuildContext &context,
                                const rund::kernel::u32 binding,
                                const Expr value) noexcept {
  if (value.context() != &context) {
    context.reject("compute_expression_context_mismatch");
    return;
  }
  context.write_checked_ordinal_node(binding, value.node());
}

void DynamicBoundaryMaskWrite(
    BuildContext &context, const rund::kernel::u32 binding, const Expr value,
    const rund::kernel::ComputeFixedFormat target_format) noexcept {
  if (value.context() != &context) {
    context.reject("compute_expression_context_mismatch");
    return;
  }
  context.write_boundary_mask_node(binding, value.node(), target_format);
}

Expr Binary(const rund::kernel::IrOp op, const Expr lhs,
            const Expr rhs) noexcept {
  BuildContext *const context = lhs.context();
  if (context == nullptr) {
    return Expr{};
  }
  if (rhs.context() != context) {
    context->reject("compute_expression_context_mismatch");
    return Expr{Expr::InternalToken{}, context, 0u};
  }
  return Expr{Expr::InternalToken{}, context,
              context->binary_node(op, lhs.node(), rhs.node())};
}

Expr Unary(const rund::kernel::IrOp op, const Expr value) noexcept {
  BuildContext *const context = value.context();
  if (context == nullptr) {
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context,
              context->unary_node(op, value.node())};
}

Expr ConstShift(const rund::kernel::IrOp op, const Expr value,
                const rund::kernel::u32 amount) noexcept {
  BuildContext *const context = value.context();
  if (context == nullptr) {
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context,
              context->const_shift_node(op, value.node(), amount)};
}

Expr Ternary(const rund::kernel::IrOp op, const Expr lhs, const Expr rhs,
             const Expr aux) noexcept {
  BuildContext *const context = lhs.context();
  if (context == nullptr) {
    return Expr{};
  }
  if (rhs.context() != context || aux.context() != context) {
    context->reject("compute_expression_context_mismatch");
    return Expr{Expr::InternalToken{}, context, 0u};
  }
  return Expr{Expr::InternalToken{}, context,
              context->ternary_node(op, lhs.node(), rhs.node(), aux.node())};
}

Expr Constant(const Expr anchor, const rund::kernel::u64 bits) noexcept {
  BuildContext *const context = anchor.context();
  if (context == nullptr) {
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context,
              context->constant_node(bits, anchor.node())};
}

Expr TypedConstant(const Expr anchor, const ScalarMode mode,
                   const rund::kernel::u64 bits) noexcept {
  BuildContext *const context = anchor.context();
  if (context == nullptr) {
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context,
              context->typed_constant_node(bits, mode)};
}

Expr FormattedConstant(const Expr anchor, const rund::kernel::u64 bits,
                       const rund::kernel::ComputeFixedFormat format) noexcept {
  BuildContext *const context = anchor.context();
  if (context == nullptr) {
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context,
              context->constant_node(bits, anchor.node(), format)};
}

Expr StorageConstant(const Expr anchor, const rund::kernel::u64 bits) noexcept {
  BuildContext *const context = anchor.context();
  if (context == nullptr) {
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context,
              context->storage_constant_node(bits, anchor.node())};
}

Expr StorageQuantize(const Expr value) noexcept {
  BuildContext *const context = value.context();
  if (context == nullptr) {
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context,
              context->storage_quantize_node(value.node())};
}

Expr UnaryFormatted(const rund::kernel::IrOp op, const Expr value,
                    const rund::kernel::ComputeFixedFormat format) noexcept {
  BuildContext *const context = value.context();
  if (context == nullptr) {
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context,
              context->unary_node(op, value.node(), format)};
}

Expr BinaryFormatted(const rund::kernel::IrOp op, const Expr lhs,
                     const Expr rhs,
                     const rund::kernel::ComputeFixedFormat format) noexcept {
  BuildContext *const context = lhs.context();
  if (context == nullptr || context != rhs.context()) {
    if (context != nullptr) {
      context->reject("compute_expression_context_mismatch");
    }
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context,
              context->binary_node(op, lhs.node(), rhs.node(), format)};
}

Expr TernaryFormatted(const rund::kernel::IrOp op, const Expr first,
                      const Expr second, const Expr third,
                      const rund::kernel::ComputeFixedFormat format) noexcept {
  BuildContext *const context = first.context();
  if (context == nullptr || context != second.context() ||
      context != third.context()) {
    if (context != nullptr) {
      context->reject("compute_expression_context_mismatch");
    }
    return Expr{};
  }
  return Expr{Expr::InternalToken{}, context,
              context->ternary_node(op, first.node(), second.node(),
                                    third.node(), format)};
}

ScalarMode ScalarModeOf(const Expr value) noexcept {
  BuildContext *const context = value.context();
  return context != nullptr ? context->scalar_mode_node(value.node())
                            : static_cast<ScalarMode>(0u);
}

rund::kernel::ComputeFixedFormat FixedFormatOf(const Expr value) noexcept {
  BuildContext *const context = value.context();
  return context != nullptr ? context->fixed_format_node(value.node())
                            : rund::kernel::ComputeFixedFormat{};
}

Expr operator+(const Expr lhs, const Expr rhs) noexcept {
  return Binary(rund::kernel::IrOp::Add, lhs, rhs);
}

Expr operator-(const Expr lhs, const Expr rhs) noexcept {
  return Binary(rund::kernel::IrOp::Sub, lhs, rhs);
}

Expr operator-(const Expr value) noexcept {
  return Unary(rund::kernel::IrOp::Neg, value);
}

Expr operator*(const Expr lhs, const Expr rhs) noexcept {
  return Binary(rund::kernel::IrOp::Mul, lhs, rhs);
}

Expr operator/(const Expr lhs, const Expr rhs) noexcept {
  return Binary(DivisionOp(ScalarModeOf(lhs)), lhs, rhs);
}

ReadHandle::ReadHandle(InternalToken, BuildContext *const context,
                       const rund::kernel::u32 binding) noexcept
    : context_(context),
      lifetime_(context != nullptr
                    ? context->lifetime()
                    : std::weak_ptr<BuildContext::ContextLifetime>{}),
      binding_(binding) {}

BuildContext *ReadHandle::context() const noexcept {
  return lifetime_.expired() ? nullptr : context_;
}

Expr ReadHandle::operator[](Index) const noexcept {
  BuildContext *const context = this->context();
  return Expr{Expr::InternalToken{}, context,
              context != nullptr ? context->read_node(binding_) : 0u};
}

WriteTarget::WriteTarget(InternalToken, BuildContext *const context,
                         const rund::kernel::u32 binding,
                         const bool supported_index) noexcept
    : context_(context),
      lifetime_(context != nullptr
                    ? context->lifetime()
                    : std::weak_ptr<BuildContext::ContextLifetime>{}),
      binding_(binding), supported_index_(supported_index) {}

BuildContext *WriteTarget::context() const noexcept {
  return lifetime_.expired() ? nullptr : context_;
}

void WriteTarget::operator=(const Expr value) && noexcept {
  BuildContext *const context = this->context();
  if (context == nullptr) {
    return;
  }
  if (!supported_index_) {
    context->reject("compute_write_index_unsupported");
    return;
  }
  if (value.context() != context) {
    context->reject("compute_expression_context_mismatch");
    return;
  }
  context->write_node(binding_, value.node());
}

WriteHandle::WriteHandle(InternalToken, BuildContext *const context,
                         const rund::kernel::u32 binding) noexcept
    : context_(context),
      lifetime_(context != nullptr
                    ? context->lifetime()
                    : std::weak_ptr<BuildContext::ContextLifetime>{}),
      binding_(binding) {}

BuildContext *WriteHandle::context() const noexcept {
  return lifetime_.expired() ? nullptr : context_;
}

WriteTarget WriteHandle::operator[](Index) const noexcept {
  BuildContext *const context = this->context();
  return WriteTarget{WriteTarget::InternalToken{}, context, binding_, true};
}

WriteTarget WriteHandle::operator[](Expr) const noexcept {
  BuildContext *const context = this->context();
  return WriteTarget{WriteTarget::InternalToken{}, context, binding_, false};
}

} // namespace rund::compute_dsl::detail
