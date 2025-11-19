#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

std::pair<int64_t, int64_t> get_dims(const ffi::Buffer<ffi::F32>& buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }

  std::cout << "get_dims size = " << dims.size() << '\n';
  for (auto dim : dims) {
    std::cout << dim << ' ';
  }
  std::cout << std::endl;

  return std::make_pair(buffer.element_count(), dims.back());
}

void compute_add(int64_t size, const float* x, const float* y, float* z) {
  for (int64_t n = 0; n < size; ++n) {
    z[n] = x[n] + y[n];
  }
}

ffi::Error add_impl(ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> y, ffi::ResultBuffer<ffi::F32> result) {
  auto [x_total_size, x_last_dim] = get_dims(x);
  if (x_last_dim == 0) {
    return ffi::Error::InvalidArgument("x input must be an array");
  }

  if (!(y.dimensions() == x.dimensions())) {
    return ffi::Error::InvalidArgument("y must have same dimensions as x");
  }

  for (int64_t n = 0; n < x_total_size; n += x_last_dim) {
    std::cout << "compute add from n = " << n << '\n';
    compute_add(
      x_last_dim,
      &(x.typed_data()[n]),
      &(y.typed_data()[n]),
      &(result->typed_data()[n])
    );
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    add, add_impl,
    ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());

template <typename T> nb::capsule encapsulate_ffi_handler(T* fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                "An encapsulated function must be an XLA FFI handler");
  return nb::capsule(reinterpret_cast<void*>(fn));
}

NB_MODULE(_add, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["add"] = encapsulate_ffi_handler(add);
    return registrations;
  });
}
