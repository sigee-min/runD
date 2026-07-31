#pragma once

#include <rund/compute/expr/value.hpp>

namespace rund::compute {
template <class Tag, class T> class ExprField final {
public:
  using TagType = Tag;
  using Value = T;
  static constexpr std::size_t size = 1u;

private:
  template <class U, class V> friend auto field(const Expr<V> &value);
  friend struct detail::ExprRecordAccess;
  explicit ExprField(detail::ExprRef ref) : ref_(std::move(ref)) {}
  detail::ExprRef ref_;
};

template <class Tag, class... Fields> class ExprRecordField final {
public:
  using TagType = Tag;
  using Value = ExprRecord<Fields...>;
  static constexpr std::size_t size = ExprRecord<Fields...>::size;

private:
  template <class U, class... Nested>
  friend auto field(const ExprRecord<Nested...> &value);
  friend struct detail::ExprRecordAccess;
  explicit ExprRecordField(ExprRecord<Fields...> value)
      : value_(std::move(value)) {}
  ExprRecord<Fields...> value_;
};

template <class... Fields> class ExprRecord final {
public:
  static constexpr std::size_t size = (Fields::size + ... + 0u);
  explicit ExprRecord(Fields... fields) : fields_(std::move(fields)...) {}

private:
  template <class... Values> friend auto record(Values &&...values);
  friend struct detail::ExprRecordAccess;
  std::tuple<Fields...> fields_;
};

namespace detail {
struct ExprAccess final {
  template <class T>
  [[nodiscard]] static const ExprRef &ref(const Expr<T> &value) noexcept {
    return value.ref_;
  }
  template <class T> [[nodiscard]] static Expr<T> make(ExprRef value) {
    return Expr<T>{std::move(value)};
  }
  template <class T>
  [[nodiscard]] static Predicate<T> predicate(ExprRef value) {
    return Predicate<T>{std::move(value)};
  }
  template <class T>
  [[nodiscard]] static const ExprRef &ref(const Predicate<T> &value) noexcept {
    return value.ref_;
  }
};

struct ExprRecordAccess final {
  template <class Tag, class T>
  [[nodiscard]] static auto refs(const ExprField<Tag, T> &field) {
    return std::array<ExprRef, 1u>{field.ref_};
  }
  template <class Tag, class... Fields>
  [[nodiscard]] static auto
  refs(const ExprRecordField<Tag, Fields...> &field) {
    return refs(field.value_);
  }
  template <class... Fields>
  [[nodiscard]] static auto refs(const ExprRecord<Fields...> &record) {
    const auto flattened = std::apply(
        [](const auto &...field) { return std::tuple_cat(refs(field)...); },
        record.fields_);
    return std::apply(
        [](const auto &...ref) {
          return std::array<ExprRef, sizeof...(ref)>{ref...};
        },
        flattened);
  }
};
} // namespace detail
} // namespace rund::compute
