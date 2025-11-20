#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// Uncomment for print debugging in internals
#define DEBUG_PRINT

std::ostream& operator<<(std::ostream& os, const ffi::Buffer<ffi::F32>::Dimensions& dims) {
  os << '(';
  size_t n_dims = dims.size();
  for (size_t i = 0, n = dims.size(); i != n; ++i) {
    os << dims[i];
    if (i != n - 1) {
      os << ", ";
    }
  }
  os << ')';
  return os;
}

std::pair<int64_t, int64_t> get_dims(const ffi::Buffer<ffi::F32>& buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }

  return std::make_pair(buffer.element_count(), dims.back());
}

void compute_add(int64_t size, const float* x, const float* y, float* z) {
  for (int64_t n = 0; n < size; ++n) {
    z[n] = x[n] + y[n];
  }
}

ffi::Error add_impl(ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> y, ffi::ResultBuffer<ffi::F32> result) {
#ifdef DEBUG_PRINT
  std::cout << "x dims = " << x.dimensions() << std::endl;
  std::cout << "x size_bytes = " << x.size_bytes() << std::endl;
  std::cout << "y dims = " << y.dimensions() << std::endl;
  std::cout << "y size_bytes = " << y.size_bytes() << std::endl;
#endif // DEBUG_PRINT

  auto [result_total_size, result_last_dim] = get_dims(*result);
  if (result_last_dim == 0) {
    return ffi::Error::InvalidArgument("result output must be an array");
  }

  // FIXME temporarily disabled to demonstrate new code at bottom of test_vmap.py
  // If concept sound, this equality should be replaced with all but last dim.
#if 0
  if (!(result->dimensions() == x.dimensions())) {
    return ffi::Error::InvalidArgument("x must have same dimensions as result");
  }

  if (!(result->dimensions() == y.dimensions())) {
    return ffi::Error::InvalidArgument("y must have same dimensions as result");
  }
#endif

  for (int64_t n = 0; n < result_total_size; n += result_last_dim) {
#ifdef DEBUG_PRINT
    std::cout << "compute add from n = " << n << '\n';
#endif // DEBUG_PRINT
    compute_add(
      result_last_dim,
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
