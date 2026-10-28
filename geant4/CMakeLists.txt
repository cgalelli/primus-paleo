#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.16...3.27)
project(Run)

#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default
# You can set WITH_GEANT4_UIVIS to OFF via the command line or ccmake/cmake-gui
# to build a batch mode only executable
#
option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
#
include(${Geant4_USE_FILE})

#----------------------------------------------------------------------------
# Locate sources and headers for this project
#
include_directories(${PROJECT_SOURCE_DIR}/include 
                    ${Geant4_INCLUDE_DIR})
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
add_executable(Run Run.cc ${sources} ${headers})
target_link_libraries(Run ${Geant4_LIBRARIES} )

set(G4_DATA_PATH_ENTRIES "")
foreach(_ds IN LISTS Geant4_DATASETS)
  set(_var  "${Geant4_DATASET_${_ds}_ENVVAR}")
  set(_path "${Geant4_DATASET_${_ds}_PATH}")
  if(NOT EXISTS "${_path}")
    get_filename_component(_dir  "${_path}" DIRECTORY)
    get_filename_component(_name "${_path}" NAME)
    string(REGEX REPLACE "^G4" "" _name "${_name}")
    set(_path "${_dir}/${_name}")
  endif()
  if(EXISTS "${_path}")
    string(APPEND G4_DATA_PATH_ENTRIES "    {\"${_var}\", \"${_path}\"},\n")
    message(STATUS "Geant4 dataset ${_var} -> ${_path}")
  else()
    message(WARNING "Geant4 dataset ${_ds} (${_var}) not found; it must be set at run time")
  endif()
endforeach()
configure_file(${PROJECT_SOURCE_DIR}/G4DataPaths.hh.in ${PROJECT_BINARY_DIR}/G4DataPaths.hh @ONLY)
include_directories(${PROJECT_BINARY_DIR})   # applies to the executables defined below
#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
install(TARGETS Run DESTINATION bin)

