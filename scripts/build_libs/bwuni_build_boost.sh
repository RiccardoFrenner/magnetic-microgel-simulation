module load compiler/gnu/11.2 mpi/openmpi/4.1
mkdir boost-build
cd boost-build
echo 'Building boost now!'
BOOST_VERSION=1.82.0
BOOST_DOMAIN="https://boostorg.jfrog.io/artifactory/main"
BOOST_ROOT="${HOME}/bin/boost_mpi_${BOOST_VERSION//./_}"
mkdir -p "${BOOST_ROOT}"
curl -sL "${BOOST_DOMAIN}/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION//./_}.tar.bz2" | tar xj
cd "boost_${BOOST_VERSION//./_}"
echo 'using mpi ;' > tools/build/src/user-config.jam
./bootstrap.sh --with-libraries=filesystem,system,mpi,serialization,test
./b2 -j 8 install --prefix="${BOOST_ROOT}"