#pragma once

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace rund::detail::task {

class Callable {
public:
  Callable(const Callable &) = delete;
  Callable &operator=(const Callable &) = delete;

  Callable(Callable &&other) noexcept { MoveFrom(std::move(other)); }

  Callable &operator=(Callable &&other) noexcept {
    if (this != &other) {
      Reset();
      MoveFrom(std::move(other));
    }
    return *this;
  }

  ~Callable() { Reset(); }

  [[nodiscard]] explicit operator bool() const noexcept {
    return invoke_ != nullptr;
  }

  [[nodiscard]] bool uses_inline_storage() const noexcept {
    return invoke_ != nullptr;
  }

  void reset() noexcept { Reset(); }

  void operator()() { invoke_(Pointer()); }

  Callable() noexcept = default;
  Callable(std::nullptr_t) noexcept {}

  template <typename Source> explicit Callable(Source &&callable) {
    using StoredCallable = std::decay_t<Source>;
    static_assert(std::is_void_v<std::invoke_result_t<StoredCallable &>>,
                  "task callables must return void");
    static_assert(std::is_move_constructible_v<StoredCallable>,
                  "task callables must be move constructible");
    Emplace<StoredCallable>(std::forward<Source>(callable));
  }

private:
  static constexpr std::size_t kInlineBytes = 64u;
  static constexpr std::size_t kInlineAlign = alignof(std::max_align_t);

  template <typename StoredCallable>
  static constexpr bool kUseInline =
      sizeof(StoredCallable) <= kInlineBytes &&
      alignof(StoredCallable) <= kInlineAlign &&
      std::is_nothrow_move_constructible_v<StoredCallable>;

  template <typename StoredCallable, typename Source>
  void Emplace(Source &&callable) {
    static_assert(kUseInline<StoredCallable>,
                  "task callable exceeds bounded inline storage");
    new (storage_) StoredCallable(std::forward<Source>(callable));
    invoke_ = [](void *const raw) { (*static_cast<StoredCallable *>(raw))(); };
    destroy_ = [](void *const raw) noexcept {
      static_cast<StoredCallable *>(raw)->~StoredCallable();
    };
    move_ = [](void *const dst, void *const src) noexcept {
      new (dst) StoredCallable(std::move(*static_cast<StoredCallable *>(src)));
      static_cast<StoredCallable *>(src)->~StoredCallable();
    };
  }

  void *Pointer() noexcept { return static_cast<void *>(storage_); }

  void MoveFrom(Callable &&other) noexcept {
    invoke_ = other.invoke_;
    destroy_ = other.destroy_;
    move_ = other.move_;
    if (other.invoke_ != nullptr) {
      move_(storage_, other.storage_);
    }
    other.invoke_ = nullptr;
    other.destroy_ = nullptr;
    other.move_ = nullptr;
  }

  void Reset() noexcept {
    if (invoke_ == nullptr) {
      return;
    }
    destroy_(storage_);
    invoke_ = nullptr;
    destroy_ = nullptr;
    move_ = nullptr;
  }

  alignas(std::max_align_t) unsigned char storage_[kInlineBytes]{};
  void (*invoke_)(void *) = nullptr;
  void (*destroy_)(void *) noexcept = nullptr;
  void (*move_)(void *, void *) noexcept = nullptr;
};

} // namespace rund::detail::task
