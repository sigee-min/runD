#pragma once

#include <kernel/program/compute/dsl/support.hpp>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rund::compute_dsl::detail {

template <ScalarMode Mode, typename... Decls> class BindBuilder {
public:
  explicit BindBuilder(const rund::kernel::u64 tile_count) noexcept
      : tile_count_(tile_count) {}

  BindBuilder(const rund::kernel::u64 tile_count,
              std::vector<BindingRuntime> bindings, const bool ok,
              const char *const reason,
              const rund::kernel::ComputeFixedFormat fixed_format = {})
      : tile_count_(tile_count), bindings_(std::move(bindings)), ok_(ok),
        reason_(reason), fixed_format_(fixed_format) {}

  template <rund::kernel::u8 IntegerBits, rund::kernel::u8 FractionBits,
            rund::kernel::ComputeRounding Round =
                rund::kernel::ComputeRounding::NearestEven,
            rund::kernel::ComputeOverflow Overflow =
                rund::kernel::ComputeOverflow::Saturate,
            rund::kernel::ComputeApproximation Approximation =
                rund::kernel::ComputeApproximation::Exact>
    requires(IntegerBits != 0u && FractionBits != 0u &&
             (static_cast<unsigned>(IntegerBits) + FractionBits == 32u ||
              static_cast<unsigned>(IntegerBits) + FractionBits == 64u))
  [[nodiscard]] auto fixed() const {
    constexpr auto mode =
        static_cast<unsigned>(IntegerBits) + FractionBits == 64u
            ? ScalarMode::FixedLane64
            : ScalarMode::FixedLane32;
    std::vector<BindingRuntime> bindings = bindings_;
    for (BindingRuntime &binding : bindings) {
      binding.numeric_mode = binding.kind == BindingKind::Write
                                 ? WriteNumericMode(mode, binding.element_bytes)
                                 : mode;
    }
    return BindBuilder<mode, Decls...>{tile_count_, std::move(bindings), ok_,
                                       reason_,
                                       rund::kernel::ComputeFixedFormat{
                                           .integer_bits = IntegerBits,
                                           .fraction_bits = FractionBits,
                                           .rounding = Round,
                                           .overflow = Overflow,
                                           .approximation = Approximation,
                                       }};
  }

  [[nodiscard]] auto i32() const { return with_mode<ScalarMode::I32>(); }
  [[nodiscard]] auto u32() const { return with_mode<ScalarMode::U32>(); }
  [[nodiscard]] auto i64() const { return with_mode<ScalarMode::I64>(); }
  [[nodiscard]] auto u64() const { return with_mode<ScalarMode::U64>(); }

  template <FixedString Name, typename T>
  [[nodiscard]] auto param(const T value) const {
    using Value = CleanType<T>;
    using NextDecl = BindingDecl<Name, BindingKind::Param, Value>;
    using Next = BindBuilder<Mode, Decls..., NextDecl>;
    std::vector<BindingRuntime> bindings = bindings_;
    bool ok = ok_;
    const char *reason = reason_;
    if constexpr (HasBindingName<Name, Decls...>) {
      ok = false;
      reason = "compute_binding_duplicate";
    }
    if constexpr (NumericMode(Mode) && FixedParamUnsupported<Value>()) {
      ok = false;
      reason = "compute_param_float_unsupported";
    }
    bindings.push_back(BindingRuntime{
        .kind = BindingKind::Param,
        .numeric_mode = Mode,
        .name = std::string{FixedStringView<Name>()},
        .element_bytes = static_cast<rund::kernel::u32>(sizeof(Value)),
        .value_bytes = ParamBytes(value),
        .floating_point_param = std::is_floating_point_v<Value>,
    });
    return Next{tile_count_, std::move(bindings), ok, reason, fixed_format_};
  }

  template <FixedString Name, typename Buffer>
    requires ComputeBuffer<Buffer>
  [[nodiscard]] auto read(Buffer &&buffer) const {
    using Element = BufferElement<Buffer &&>;
    using NextDecl = BindingDecl<Name, BindingKind::Read, Element>;
    using Next = BindBuilder<Mode, Decls..., NextDecl>;
    std::vector<BindingRuntime> bindings = bindings_;
    bool ok = ok_;
    const char *reason = reason_;
    const RuntimeBuffer runtime =
        RuntimeBufferFor(std::forward<Buffer>(buffer), tile_count_);
    if constexpr (HasBindingName<Name, Decls...>) {
      ok = false;
      reason = "compute_binding_duplicate";
    }
    bindings.push_back(BindingRuntime{
        .kind = BindingKind::Read,
        .numeric_mode = Mode,
        .name = std::string{FixedStringView<Name>()},
        .element_bytes = ElementBytes<Buffer &&>(),
        .runtime_data = runtime.data,
        .runtime_count = runtime.count,
        .runtime_stride_bytes = runtime.stride_bytes,
        .has_runtime_storage = runtime.has_storage,
    });
    return Next{tile_count_, std::move(bindings), ok, reason, fixed_format_};
  }

  template <FixedString Name, typename Buffer>
    requires ComputeWritableBuffer<Buffer>
  [[nodiscard]] auto write(Buffer &&buffer) const {
    using Element = BufferElement<Buffer &&>;
    using NextDecl = BindingDecl<Name, BindingKind::Write, Element>;
    using Next = BindBuilder<Mode, Decls..., NextDecl>;
    std::vector<BindingRuntime> bindings = bindings_;
    bool ok = ok_;
    const char *reason = reason_;
    const RuntimeBuffer runtime =
        RuntimeBufferFor(std::forward<Buffer>(buffer), tile_count_);
    if constexpr (HasBindingName<Name, Decls...>) {
      ok = false;
      reason = "compute_binding_duplicate";
    }
    bindings.push_back(BindingRuntime{
        .kind = BindingKind::Write,
        .numeric_mode = WriteNumericMode(Mode, ElementBytes<Buffer &&>()),
        .name = std::string{FixedStringView<Name>()},
        .element_bytes = ElementBytes<Buffer &&>(),
        .runtime_data = runtime.data,
        .runtime_write_data = runtime.write_data,
        .runtime_count = runtime.count,
        .runtime_stride_bytes = runtime.stride_bytes,
        .has_runtime_storage = runtime.has_storage,
    });
    return Next{tile_count_, std::move(bindings), ok, reason, fixed_format_};
  }

  template <FixedString Name>
  [[nodiscard]] static consteval bool has_param() noexcept {
    return HasBinding<Name, BindingKind::Param, Decls...>;
  }

  template <FixedString Name>
  [[nodiscard]] static consteval bool has_read() noexcept {
    return HasBinding<Name, BindingKind::Read, Decls...>;
  }

  template <FixedString Name>
  [[nodiscard]] static consteval bool has_write() noexcept {
    return HasBinding<Name, BindingKind::Write, Decls...>;
  }

  [[nodiscard]] static constexpr ScalarMode scalar_mode() noexcept {
    return Mode;
  }
  [[nodiscard]] constexpr rund::kernel::u64 tile_count() const noexcept {
    return tile_count_;
  }
  [[nodiscard]] const std::vector<BindingRuntime> &bindings() const noexcept {
    return bindings_;
  }
  [[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
  fixed_format() const noexcept {
    return fixed_format_;
  }
  [[nodiscard]] bool ok() const noexcept { return ok_; }
  [[nodiscard]] const char *reason() const noexcept { return reason_; }

private:
  template <ScalarMode NewMode> [[nodiscard]] auto with_mode() const {
    std::vector<BindingRuntime> bindings = bindings_;
    for (BindingRuntime &binding : bindings) {
      binding.numeric_mode =
          binding.kind == BindingKind::Write
              ? WriteNumericMode(NewMode, binding.element_bytes)
              : NewMode;
    }
    return BindBuilder<NewMode, Decls...>{tile_count_, std::move(bindings), ok_,
                                          reason_,
                                          rund::kernel::ComputeFixedFormat{}};
  }

  rund::kernel::u64 tile_count_ = 0u;
  std::vector<BindingRuntime> bindings_;
  bool ok_ = true;
  const char *reason_ = "ok";
  rund::kernel::ComputeFixedFormat fixed_format_{};
};

} // namespace rund::compute_dsl::detail
