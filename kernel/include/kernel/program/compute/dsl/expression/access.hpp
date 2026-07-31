#pragma once

#include <kernel/program/compute/dsl/expression/operator.hpp>

#include <memory>

namespace rund::compute_dsl::detail {

struct Index {};

class ReadHandle {
public:
  [[nodiscard]] Expr operator[](Index) const noexcept;

private:
  struct InternalToken {};

  ReadHandle(InternalToken, BuildContext *context,
             rund::kernel::u32 binding) noexcept;
  [[nodiscard]] BuildContext *context() const noexcept;

  BuildContext *context_ = nullptr;
  std::weak_ptr<BuildContext::ContextLifetime> lifetime_;
  rund::kernel::u32 binding_ = 0u;

  template <typename Body> friend class Access;
};

class WriteTarget {
public:
  void operator=(Expr value) && noexcept;

private:
  struct InternalToken {};

  WriteTarget(InternalToken, BuildContext *context, rund::kernel::u32 binding,
              bool supported_index) noexcept;
  [[nodiscard]] BuildContext *context() const noexcept;

  BuildContext *context_ = nullptr;
  std::weak_ptr<BuildContext::ContextLifetime> lifetime_;
  rund::kernel::u32 binding_ = 0u;
  bool supported_index_ = false;

  friend class WriteHandle;
};

class WriteHandle {
public:
  [[nodiscard]] WriteTarget operator[](Index) const noexcept;
  [[nodiscard]] WriteTarget operator[](Expr) const noexcept;

private:
  struct InternalToken {};

  WriteHandle(InternalToken, BuildContext *context,
              rund::kernel::u32 binding) noexcept;
  [[nodiscard]] BuildContext *context() const noexcept;

  BuildContext *context_ = nullptr;
  std::weak_ptr<BuildContext::ContextLifetime> lifetime_;
  rund::kernel::u32 binding_ = 0u;

  template <typename Body> friend class Access;
};

template <typename Body> class Access {
public:
  explicit Access(BuildContext &context) noexcept
      : context_(&context), lifetime_(context.lifetime()) {}

  template <FixedString Name>
    requires(Body::template has_param<Name>())
  [[nodiscard]] Expr param() const noexcept {
    BuildContext *const context = this->context();
    return Expr{
        Expr::InternalToken{}, context,
        context != nullptr ? context->param_node(FixedStringView<Name>()) : 0u};
  }

  [[nodiscard]] Expr constant(const rund::kernel::u64 bits) const noexcept {
    BuildContext *const context = this->context();
    return Expr{Expr::InternalToken{}, context,
                context != nullptr ? context->constant_node(bits) : 0u};
  }

  [[nodiscard]] Expr index() const noexcept {
    BuildContext *const context = this->context();
    return Expr{Expr::InternalToken{}, context,
                context != nullptr ? context->index_node() : 0u};
  }

  template <FixedString Name>
    requires(Body::template has_read<Name>())
  [[nodiscard]] ReadHandle read() const noexcept {
    BuildContext *const context = this->context();
    return ReadHandle{
        ReadHandle::InternalToken{}, context,
        context != nullptr
            ? context->binding_index(FixedStringView<Name>(), BindingKind::Read)
            : 0u};
  }

  template <FixedString Name>
    requires(Body::template has_write<Name>())
  [[nodiscard]] WriteHandle write() const noexcept {
    BuildContext *const context = this->context();
    return WriteHandle{WriteHandle::InternalToken{}, context,
                       context != nullptr
                           ? context->binding_index(FixedStringView<Name>(),
                                                    BindingKind::Write)
                           : 0u};
  }

private:
  [[nodiscard]] BuildContext *context() const noexcept {
    return lifetime_.expired() ? nullptr : context_;
  }

  BuildContext *context_ = nullptr;
  std::weak_ptr<BuildContext::ContextLifetime> lifetime_;
};

} // namespace rund::compute_dsl::detail
