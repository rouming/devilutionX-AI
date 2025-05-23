include(functions/FetchContent_ExcludeFromAll_backport)

set(BUILD_HOST_SELFTEST OFF)

include(FetchContent)
FetchContent_Declare_ExcludeFromAll(libzt
  GIT_REPOSITORY https://github.com/diasurgical/libzt.git
  GIT_TAG db7b642a4ce9f0f5e0ba7f293bd7ffa7897e4831)
FetchContent_MakeAvailable_ExcludeFromAll(libzt)

if(NOT ANDROID)
  set(libzt_LIB_NAME zt-static)
else()
  set(libzt_LIB_NAME zt-shared)
endif()

foreach(
  lib_name
  zto_obj
  libnatpmp_obj
  libzt_obj
  lwip_obj
  miniupnpc_obj
  zto_pic
  natpmp_pic
  lwip_pic
  miniupnpc_pic
  ztcore
  ${libzt_LIB_NAME}
)
  if(TARGET ${lib_name})
    # External library, ignore all warnings
    target_compile_options(${lib_name} PRIVATE -w)
  endif()
endforeach(lib_name)

target_include_directories(${libzt_LIB_NAME} INTERFACE
  "${libzt_SOURCE_DIR}/include"
  "${libzt_SOURCE_DIR}/src"
  "${libzt_SOURCE_DIR}/ext/lwip/src/include")

if(WIN32)
  target_include_directories(${libzt_LIB_NAME} INTERFACE
    "${libzt_SOURCE_DIR}/ext/lwip-contrib/ports/win32/include")
else()
  target_include_directories(${libzt_LIB_NAME} INTERFACE
    "${libzt_SOURCE_DIR}/ext/lwip-contrib/ports/unix/port/include")
endif()

if(MINGW_CROSS)
  include(zt_defs REQUIRED)
endif()

if(MSVC)
  target_compile_definitions(libnatpmp_obj PRIVATE -DSTATICLIB)
endif()
