#pragma once

#include <rund/compute/expr/function.hpp>

namespace rund::compute {
template <class T> class Predicate final {
public:
  Predicate(const Predicate &) noexcept = default;
  Predicate(Predicate &&) noexcept = default;
  Predicate &operator=(const Predicate &) noexcept = default;
  Predicate &operator=(Predicate &&) noexcept = default;
  [[nodiscard]] Predicate operator!() const {
    return Predicate{detail::unary(detail::ExprOp::PredicateNot, ref_)};
  }
  [[nodiscard]] Predicate operator&&(const Predicate &right) const {
    return binary(detail::ExprOp::PredicateAnd, right);
  }
  [[nodiscard]] Predicate operator||(const Predicate &right) const {
    return binary(detail::ExprOp::PredicateOr, right);
  }

private:
  template <class> friend class Expr;
  friend struct detail::ExprAccess;
  template <class C, class U>
    requires(sizeof(C) == sizeof(U))
  friend Expr<U> select(const Predicate<C> &, const Expr<U> &, const Expr<U> &);
  template <class U, class C>
    requires(sizeof(C) == sizeof(U))
  friend Expr<U> select(const Predicate<C> &, U, U);
  template <class U> friend Expr<std::uint32_t> mask(const Predicate<U> &);
  explicit Predicate(detail::ExprRef ref) : ref_(std::move(ref)) {}
  [[nodiscard]] Predicate binary(const detail::ExprOp operation,
                                 const Predicate &right) const {
    return Predicate{detail::binary(operation, ref_, right.ref_)};
  }
  detail::ExprRef ref_;
};
template <class T> class Expr final {
public:
  Expr(const Expr &) noexcept = default;
  Expr(Expr &&) noexcept = default;
  Expr &operator=(const Expr &) noexcept = default;
  Expr &operator=(Expr &&) noexcept = default;
  [[nodiscard]] Expr operator-() const {
    return Expr{detail::unary(detail::ExprOp::Negate, ref_)};
  }
  [[nodiscard]] Expr operator+(const Expr &right) const {
    return binary(detail::ExprOp::Add, right);
  }
  [[nodiscard]] Expr operator-(const Expr &right) const {
    return binary(detail::ExprOp::Subtract, right);
  }
  [[nodiscard]] Expr operator*(const Expr &right) const {
    return binary(detail::ExprOp::Multiply, right);
  }
  [[nodiscard]] Expr operator/(const Expr &right) const {
    return binary(detail::ExprOp::Divide, right);
  }
  [[nodiscard]] Expr operator&(const Expr &right) const {
    return binary(detail::ExprOp::BitAnd, right);
  }
  [[nodiscard]] Expr operator|(const Expr &right) const {
    return binary(detail::ExprOp::BitOr, right);
  }
  [[nodiscard]] Expr operator^(const Expr &right) const {
    return binary(detail::ExprOp::BitXor, right);
  }
  [[nodiscard]] Predicate<T> operator==(const Expr &right) const {
    return compare(detail::ExprOp::Equal, right);
  }
  [[nodiscard]] Predicate<T> operator!=(const Expr &right) const {
    return compare(detail::ExprOp::NotEqual, right);
  }
  [[nodiscard]] Predicate<T> operator<(const Expr &right) const {
    return compare(detail::ExprOp::Less, right);
  }
  [[nodiscard]] Predicate<T> operator<=(const Expr &right) const {
    return compare(detail::ExprOp::LessEqual, right);
  }
  [[nodiscard]] Predicate<T> operator>(const Expr &right) const {
    return compare(detail::ExprOp::Greater, right);
  }
  [[nodiscard]] Predicate<T> operator>=(const Expr &right) const {
    return compare(detail::ExprOp::GreaterEqual, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator+(const Expr<U> &right) const {
    return mixed(detail::ExprOp::Add, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator-(const Expr<U> &right) const {
    return mixed(detail::ExprOp::Subtract, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator*(const Expr<U> &right) const {
    return mixed(detail::ExprOp::Multiply, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator/(const Expr<U> &right) const {
    return mixed(detail::ExprOp::Divide, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator&(const Expr<U> &right) const {
    return mixed(detail::ExprOp::BitAnd, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator|(const Expr<U> &right) const {
    return mixed(detail::ExprOp::BitOr, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator^(const Expr<U> &right) const {
    return mixed(detail::ExprOp::BitXor, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator==(const Expr<U> &right) const {
    return mixed_compare(detail::ExprOp::Equal, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator!=(const Expr<U> &right) const {
    return mixed_compare(detail::ExprOp::NotEqual, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator<(const Expr<U> &right) const {
    return mixed_compare(detail::ExprOp::Less, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator<=(const Expr<U> &right) const {
    return mixed_compare(detail::ExprOp::LessEqual, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator>(const Expr<U> &right) const {
    return mixed_compare(detail::ExprOp::Greater, right);
  }
  template <class U>
    requires(!std::same_as<T, U> && std::integral<T> && std::integral<U> &&
             sizeof(T) == sizeof(U))
  [[nodiscard]] auto operator>=(const Expr<U> &right) const {
    return mixed_compare(detail::ExprOp::GreaterEqual, right);
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Expr operator+(const U right) const {
    return *this + literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Expr operator-(const U right) const {
    return *this - literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Expr operator*(const U right) const {
    return *this * literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Expr operator/(const U right) const {
    return *this / literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Expr operator&(const U right) const {
    return *this & literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Expr operator|(const U right) const {
    return *this | literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Expr operator^(const U right) const {
    return *this ^ literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Predicate<T> operator==(const U right) const {
    return *this == literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Predicate<T> operator!=(const U right) const {
    return *this != literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Predicate<T> operator<(const U right) const {
    return *this < literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Predicate<T> operator<=(const U right) const {
    return *this <= literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Predicate<T> operator>(const U right) const {
    return *this > literal(static_cast<T>(right));
  }
  template <class U>
    requires std::convertible_to<U, T>
  [[nodiscard]] Predicate<T> operator>=(const U right) const {
    return *this >= literal(static_cast<T>(right));
  }

private:
  template <class> friend class Expr;
  friend struct detail::ExprAccess;
  template <class, class, class> friend class Flow;
  template <class, class> friend class StageRef;
  template <class, class, class> friend class Groups;
  template <class...> friend class ZipRef;
  template <class, class> friend class GroupValuesRef;
  template <class C, class U>
    requires(sizeof(C) == sizeof(U))
  friend Expr<U> select(const Predicate<C> &, const Expr<U> &, const Expr<U> &);
  template <class C, class U, class V>
    requires(sizeof(C) == sizeof(U) && std::convertible_to<V, U>)
  friend Expr<U> select(const Predicate<C> &, const Expr<U> &, V);
  template <class C, class U, class V>
    requires(sizeof(C) == sizeof(U) && std::convertible_to<V, U>)
  friend Expr<U> select(const Predicate<C> &, V, const Expr<U> &);
  template <class U, class C>
    requires(sizeof(C) == sizeof(U))
  friend Expr<U> select(const Predicate<C> &, U, U);
  template <class U> friend Expr<std::uint32_t> mask(const Predicate<U> &);
  template <class U> friend Expr<U> min(const Expr<U> &, const Expr<U> &);
  template <class U, class V>
    requires std::convertible_to<V, U>
  friend Expr<U> min(const Expr<U> &, V);
  template <class U> friend Expr<U> max(const Expr<U> &, const Expr<U> &);
  template <class Target, Rounding Round, Overflow OverflowMode,
            Approximation ApproximationMode, class Source>
    requires(detail::FixedValue<Target> && detail::FixedValue<Source>)
  friend Expr<Target> quantize(const Expr<Source> &);
  template <class U, class V>
    requires std::convertible_to<V, U>
  friend Expr<U> max(const Expr<U> &, V);
  template <class U>
  friend Expr<U> clamp(const Expr<U> &, const Expr<U> &, const Expr<U> &);
  template <class U, class L, class H>
    requires(std::convertible_to<L, U> && std::convertible_to<H, U>)
  friend Expr<U> clamp(const Expr<U> &, L, H);
  explicit Expr(detail::ExprRef ref) : ref_(std::move(ref)) {
    if constexpr (detail::FixedValue<T>) {
      ref_ =
          detail::with_fixed_format(std::move(ref_), detail::fixed_format<T>());
    }
  }
  [[nodiscard]] Expr binary(const detail::ExprOp operation,
                            const Expr &right) const {
    return Expr{detail::binary(operation, ref_, right.ref_)};
  }
  [[nodiscard]] Predicate<T> compare(const detail::ExprOp operation,
                                     const Expr &right) const {
    return Predicate<T>{detail::binary(operation, ref_, right.ref_)};
  }
  template <class U>
  [[nodiscard]] auto mixed(const detail::ExprOp operation,
                           const Expr<U> &right) const {
    using Common = std::common_type_t<T, U>;
    return Expr<Common>{detail::binary(
        operation, detail::retype_expr(ref_, detail::type<Common>()),
        detail::retype_expr(right.ref_, detail::type<Common>()))};
  }
  template <class U>
  [[nodiscard]] auto mixed_compare(const detail::ExprOp operation,
                                   const Expr<U> &right) const {
    using Common = std::common_type_t<T, U>;
    return Predicate<Common>{detail::binary(
        operation, detail::retype_expr(ref_, detail::type<Common>()),
        detail::retype_expr(right.ref_, detail::type<Common>()))};
  }
  [[nodiscard]] Expr literal(const T value) const {
    std::uint64_t bits = 0u;
    if constexpr (detail::FixedValue<T> && sizeof(T) == sizeof(std::uint32_t)) {
      bits = static_cast<std::uint32_t>(value.raw());
    } else if constexpr (detail::FixedValue<T>) {
      bits = static_cast<std::uint64_t>(value.raw());
    } else if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
      bits = static_cast<std::uint32_t>(value);
    } else {
      bits = static_cast<std::uint64_t>(value);
    }
    if constexpr (detail::FixedValue<T>) {
      return Expr{
          detail::constant(ref_.state, detail::type<T>(), bits,
                           detail::fixed_literal_format<T>(ref_.fixed_format))};
    } else {
      return Expr{detail::constant(ref_.state, detail::type<T>(), bits)};
    }
  }
  detail::ExprRef ref_;
};
} // namespace rund::compute
