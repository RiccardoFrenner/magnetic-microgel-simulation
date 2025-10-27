module load compiler/gnu/14.1 mpi/openmpi/4.1
CLUSTER_FFTW3_VERSION=3.3.10
CLUSTER_BOOST_VERSION=1.82.0
CLUSTER_PYTHON_VERSION=3.12.4
export BOOST_ROOT="${HOME}/bin/boost_mpi_${CLUSTER_BOOST_VERSION//./_}"
export Boost_DIR="${HOME}/bin/boost_mpi_${CLUSTER_BOOST_VERSION//./_}/lib/cmake/Boost-1.82.0"
export FFTW3_ROOT="${HOME}/bin/fftw_${CLUSTER_FFTW3_VERSION//./_}"

export LD_LIBRARY_PATH="${BOOST_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${FFTW3_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"${HOME}/bin/cpython-${CLUSTER_PYTHON_VERSION}/bin/python3" -m venv "${HOME}/venv"
source "${HOME}/venv/bin/activate"

git clone --recursive --branch 4.2 --origin upstream \
    https://github.com/espressomd/espresso.git espresso-4.2
cd espresso-4.2
python3 -m pip install -c "requirements.txt" cython setuptools numpy scipy vtk cmake
mkdir build
cd build
cp "${HOME}/microgels_myconfig.hpp" myconfig.hpp
cmake .. -D CMAKE_BUILD_TYPE=Release \
    -D ESPRESSO_BUILD_WITH_CCACHE=OFF \
    -D ESPRESSO_BUILD_WITH_SCAFACOS=OFF \
    -D ESPRESSO_BUILD_WITH_HDF5=OFF

make -j 16