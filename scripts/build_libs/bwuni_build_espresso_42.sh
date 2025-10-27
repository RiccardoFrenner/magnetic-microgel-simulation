module load compiler/gnu/11.2 mpi/openmpi/4.1 devel/cmake/3.23.3 devel/python/3.11.7_gnu_11.2
CLUSTER_FFTW3_VERSION=3.3.10
CLUSTER_BOOST_VERSION=1.82.0
export BOOST_ROOT="${HOME}/bin/boost_mpi_${CLUSTER_BOOST_VERSION//./_}"
export FFTW3_ROOT="${HOME}/bin/fftw_${CLUSTER_FFTW3_VERSION//./_}"
export LD_LIBRARY_PATH="${BOOST_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${FFTW3_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# export Python3_ROOT_DIR="/opt/bwhpc/common/devel/python/3.11.7_gnu_11.2"
# export Python_ROOT_DIR="/opt/bwhpc/common/devel/python/3.11.7_gnu_11.2"
# export PYTHON_INCLUDE_DIRS="/opt/bwhpc/common/devel/python/3.11.7_gnu_11.2/include/python3.11"
# export PYTHON_INSTDIR="/opt/bwhpc/common/devel/python/3.11.7_gnu_11.2"


git clone --recursive --branch 4.2 --origin upstream \
    https://github.com/espressomd/espresso.git espresso-4.2
cd espresso-4.2
python3 -m pip install --user -c "requirements.txt" jinja2 networkx cython setuptools numpy scipy vtk h5py seaborn matplotlib
mkdir build
cd build
cp "${HOME}/configs/espresso/mmgel/myconfig.hpp" myconfig.hpp
cmake .. -D CMAKE_BUILD_TYPE=Release -D WITH_CUDA=OFF \
    # -D PYTHON_EXECUTABLE="/opt/bwhpc/common/devel/python/3.11.7_gnu_11.2" \
    # -D PYTHON_INCLUDE_DIRS="/opt/bwhpc/common/devel/python/3.11.7_gnu_11.2/include/python3.11" \
    # -D PYTHON_INSTDIR="/opt/bwhpc/common/devel/python/3.11.7_gnu_11.2" \
    -D WITH_CCACHE=OFF -D WITH_SCAFACOS=OFF -D WITH_HDF5=OFF
make -j 12
