# last update: December 2024
module load compiler/gnu/12.1 mpi/openmpi/4.1
mkdir boost-build
cd boost-build
BOOST_VERSION=1.82.0
BOOST_DOMAIN="https://archives.boost.io"
BOOST_ROOT="${HOME}/bin/boost_mpi_${BOOST_VERSION//./_}"
mkdir -p "${BOOST_ROOT}"
curl -sL "${BOOST_DOMAIN}/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION//./_}.tar.bz2" | tar xj
cd "boost_${BOOST_VERSION//./_}"
echo 'using mpi ;' > tools/build/src/user-config.jam
./bootstrap.sh --with-libraries=filesystem,system,mpi,serialization,test
./b2 -j 4 install --prefix="${BOOST_ROOT}"