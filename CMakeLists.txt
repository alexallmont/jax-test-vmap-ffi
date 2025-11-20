cmake_minimum_required(VERSION 3.15...3.30)

project(vmap_cpp LANGUAGES CXX)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module("_add" NB_STATIC "vmap_cpp/add.cpp")
target_include_directories("_add" PUBLIC ${XLA_DIR})
set_target_properties("_add" PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_LIST_DIR}/vmap_cpp
    LIBRARY_OUTPUT_NAME "_add"
)
