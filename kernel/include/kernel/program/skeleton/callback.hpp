#pragma once

#include <functional>
#include <type_traits>

namespace rund::kernel::skeleton_detail {

template <typename T>
struct IsStdFunction : std::false_type {};

template <typename R, typename... Args>
struct IsStdFunction<std::function<R(Args...)>> : std::true_type {};

template <typename T>
struct IsReferenceWrapper : std::false_type {};

template <typename T>
struct IsReferenceWrapper<std::reference_wrapper<T>> : std::true_type {};

template <typename Callback>
concept DirectCallback =
    !std::is_pointer_v<std::remove_cvref_t<Callback>> &&
    !std::is_function_v<std::remove_reference_t<Callback>> &&
    !IsReferenceWrapper<std::remove_cvref_t<Callback>>::value &&
    !IsStdFunction<std::remove_cvref_t<Callback>>::value &&
    !std::is_polymorphic_v<std::remove_cvref_t<Callback>>;

} // namespace rund::kernel::skeleton_detail
