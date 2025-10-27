module load compiler/gnu/11.2 mpi/openmpi/4.1
mkdir fftw-build
cd fftw-build
FFTW3_VERSION=3.3.10
FFTW3_ROOT="${HOME}/bin/fftw_${FFTW3_VERSION//./_}"
echo 'Downloading now!'
curl -sL "https://www.fftw.org/fftw-${FFTW3_VERSION}.tar.gz" | tar xz
cd "fftw-${FFTW3_VERSION}"
./configure --enable-shared --enable-mpi --enable-threads --enable-openmp \
    --disable-fortran --enable-avx --prefix="${FFTW3_ROOT}"
echo 'Compiling now!'
make -j 8
echo 'Installing now!'
make install
make clean