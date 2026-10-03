cmake_minimum_required(VERSION 3.16)
project(vektor_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Architecture-specific SIMD optimization flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -fPIC")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
endif()

include_directories(include)

add_library(vektor_core SHARED
    src/engine_c_api.cpp
    src/simddistance.cpp
)

set_target_properties(vektor_core PROPERTIES
    OUTPUT_NAME "vektor_core"
    PREFIX "lib"
)
