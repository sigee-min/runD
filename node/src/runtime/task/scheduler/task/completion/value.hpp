#pragma once

#include "../completion.hpp"

#include <rund/task/handle/typed.hpp>

#include <memory>
#include <new>
#include <utility>

namespace rund::node {

template <typename T>
::rund::detail::task::ResultHandle<T>
CompletionPool::observe(const CompletionLease lease) noexcept {
  return ::rund::detail::task::ResultHandle<T>{observe_ref(lease)};
}

template <typename T>
  requires std::is_nothrow_move_constructible_v<T>
task::Status CompletionPool::complete(const CompletionLease lease,
                                      T value) noexcept {
  const task::Status prepared =
      prepare_raw(lease, std::addressof(value), sizeof(T), alignof(T),
                  type_tag<T>(), &Move<T>, &Destroy<T>);
  return prepared ? publish(lease) : prepared;
}

template <typename T>
task::Result<T> CompletionPool::read(const CompletionLease lease) {
  alignas(T) std::byte storage[sizeof(T)];
  const task::Status copied = copy_raw(lease, storage, type_tag<T>(), &Copy<T>);
  if (!copied) {
    return task::Result<T>::fail(copied.code());
  }
  T *const value = reinterpret_cast<T *>(storage);
  task::Result<T> result = task::Result<T>::success(std::move(*value));
  value->~T();
  return result;
}

template <typename T> const void *CompletionPool::type_tag() noexcept {
  return ::rund::detail::task::ResultTag<T>();
}

template <typename T>
void CompletionPool::Move(void *const out, void *const value) noexcept {
  new (out) T(std::move(*static_cast<T *>(value)));
}

template <typename T>
void CompletionPool::Copy(void *const out, const void *const value) {
  new (out) T(*static_cast<const T *>(value));
}

template <typename T> void CompletionPool::Destroy(void *const value) noexcept {
  static_cast<T *>(value)->~T();
}

} // namespace rund::node
