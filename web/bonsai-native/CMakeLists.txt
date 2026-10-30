cmake_minimum_required(VERSION 3.20)
project(omg_bonsai_web C CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Build with emcmake cmake")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_WASM_MEM64 OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_HTML OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
add_compile_options(-O3 -msimd128 -fexceptions)
add_subdirectory(../../vendor/llama-cpp-sys-2/llama.cpp llama)
add_executable(bonsai bridge.cpp)
target_link_libraries(bonsai PRIVATE llama)
target_link_options(bonsai PRIVATE
  -O3 -msimd128 -fexceptions
  "-sDISABLE_EXCEPTION_CATCHING=0"
  "-sMODULARIZE=1" "-sEXPORT_ES6=1" "-sENVIRONMENT=worker,node"
  "-sALLOW_MEMORY_GROWTH=1" "-sINITIAL_MEMORY=67108864" "-sMAXIMUM_MEMORY=4294967296"
  "-sSTACK_SIZE=5242880" "-sFORCE_FILESYSTEM=1"
  "-sEXPORTED_RUNTIME_METHODS=['ccall','FS','WORKERFS','HEAP32','HEAPF32']"
  "-sEXPORTED_FUNCTIONS=['_malloc','_free']"
  -lworkerfs.js)
set_target_properties(bonsai PROPERTIES SUFFIX ".js")
