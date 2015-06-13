# build tbb
set(tbb_PREFIX ${CMAKE_BINARY_DIR}/contrib/tbb-prefix)
set(tbb_INSTALL ${CMAKE_BINARY_DIR}/contrib/tbb-install)

set(tbb_SRC ${tbb_PREFIX}/src/tbb)

if (FORCE_TSX_OFF)
  set(TBB_PATCH_COMMAND patch -p1 -t -N < ${PROJECT_SOURCE_DIR}/contrib/tbb-disable-tsx.patch)
else()
  set(TBB_PATCH_COMMAND "")
endif()

ExternalProject_Add(tbb
    PREFIX ${tbb_PREFIX}
    GIT_REPOSITORY "https://github.com/nsubtil/tbb.git"
    GIT_TAG "4.3-20150611"
    BUILD_IN_SOURCE 1
    PATCH_COMMAND ${TBB_PATCH_COMMAND}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${MAKE}
    INSTALL_COMMAND ${PROJECT_SOURCE_DIR}/contrib/tbb-install.sh ${tbb_SRC} ${tbb_INSTALL}
    LOG_DOWNLOAD 1
    )

include_directories(${tbb_INSTALL}/include)
set(tbb_LIB ${tbb_INSTALL}/lib/libtbb.a)
