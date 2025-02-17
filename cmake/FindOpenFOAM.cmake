# Find OpenFOAM
#
# This module defines:
#  OPENFOAM_FOUND - System has OpenFOAM
#  OPENFOAM_INCLUDE_DIRS - The OpenFOAM include directories
#  OPENFOAM_LIBRARIES - The OpenFOAM libraries
#  OPENFOAM_VERSION - The OpenFOAM version

# Ensure OpenFOAM is sourced
if(NOT DEFINED ENV{WM_PROJECT_DIR})
    message(FATAL_ERROR "OpenFOAM environment is not sourced. Please source OpenFOAM's bashrc file.")
endif()

set(OPENFOAM_DIR $ENV{WM_PROJECT_DIR})

set(OPENFOAM_INCLUDE_DIRS
    $ENV{WM_PROJECT_DIR}/src/OpenFOAM/lnInclude
    $ENV{WM_PROJECT_DIR}/src/finiteVolume/lnInclude
    $ENV{WM_PROJECT_DIR}/src/meshTools/lnInclude
    $ENV{WM_PROJECT_DIR}/src/fvOptions/lnInclude
    $ENV{WM_PROJECT_DIR}/src/OSspecific/POSIX/lnInclude
)

# Compiler Flags for OpenFOAM
set(OPENFOAM_COMPILE_FLAGS
    -m64
    -DOPENFOAM=$ENV{WM_PROJECT_VERSION}
    -DWM_DP
    -DWM_LABEL_SIZE=32
    -Wall
    -Wno-pedantic
    -DNoRepository
    -ftemplate-depth-100
    -O3
    -fPIC
)

# Add the definitions for OpenFOAM support
set(OPENFOAM_COMPILE_DEFINITIONS
    -DOPENSN_WITH_OPENFOAM
    -DOPENFOAM=$ENV{WM_PROJECT_VERSION}
    -DWM_DP
    -DWM_LABEL_SIZE=$ENV{WM_LABEL_SIZE}
)


# Find OpenFOAM libraries
find_library(
    OPENFOAM_LIBRARY
    OpenFOAM
    PATHS $ENV{FOAM_LIBBIN}
)

find_library(
    FINITE_VOLUME_LIBRARY
    finiteVolume
    PATHS $ENV{FOAM_LIBBIN}
)

find_library(
    MESH_TOOLS_LIBRARY
    meshTools
    PATHS $ENV{FOAM_LIBBIN}
)

find_library(
    FV_OPTIONS_LIBRARY
    fvOptions
    PATHS $ENV{FOAM_LIBBIN}
)

# Collect all OpenFOAM libraries
set(OPENFOAM_LIBRARIES
    ${OPENFOAM_LIBRARY}
    ${FINITE_VOLUME_LIBRARY}
    ${MESH_TOOLS_LIBRARY}
    ${FV_OPTIONS_LIBRARY}
)

# Detect OpenFOAM version
set(OPENFOAM_VERSION $ENV{WM_PROJECT_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    OpenFOAM
    REQUIRED_VARS OPENFOAM_LIBRARIES OPENFOAM_INCLUDE_DIRS
    VERSION_VAR OPENFOAM_VERSION
)
