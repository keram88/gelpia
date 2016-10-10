#!/bin/bash

set -e

SCRIPT_LOCATION="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SOURCE_LOCATION=$SCRIPT_LOCATION/Sources
mkdir -p $SOURCE_LOCATION

# Rust
cd $SOURCE_LOCATION
rm -f rust_nightly.tar.gz
wget https://static.rust-lang.org/dist/2016-05-12/rust-nightly-x86_64-unknown-linux-gnu.tar.gz -O rust_nightly.tar.gz
mkdir -p rust_nightly && tar -xf rust_nightly.tar.gz -C rust_nightly --strip-components 1
cd rust_nightly
./install.sh --prefix=$SCRIPT_LOCATION > /dev/null

# CRLibM
cd $SOURCE_LOCATION
rm -f crlibm.tar.gz
wget http://lipforge.ens-lyon.fr/frs/download.php/162/crlibm-1.0beta4.tar.gz -O crlibm.tar.gz
mkdir -p crlibm && tar -xf crlibm.tar.gz -C crlibm --strip-components 1
cd crlibm
export CFLAGS=-fPIC $CFLAGS
export LDFLAGS=-fPIC $LDFLAGS
./configure --enable-sse2 --prefix=$SCRIPT_LOCATION > /dev/null
make >& /dev/null
make install > /dev/null
export LIBRARY_PATH=$SCRIPT_LOCATION/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$SCRIPT_LOCATION/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$SCRIPT_LOCATION/include:$CPLUS_INCLUDE_PATH

# GAOL
cd $SOURCE_LOCATION
rm -f gaol.tar.gz
wget http://downloads.sourceforge.net/project/gaol/gaol/4.2.0/gaol-4.2.0.tar.gz -O gaol.tar.gz
tar xf gaol.tar.gz
patch -p0 < $SCRIPT_LOCATION/../documents/gaol-4.2.0.patch > /dev/null
mv gaol-4.2.0 gaol
cd gaol
export CFLAGS=" -msse3 "; export CXXFLAGS=" -msse3 ";
./configure  --with-mathlib=crlibm --enable-simd --enable-preserve-rounding=yes\
	     --disable-debug --enable-optimize --disable-verbose-mode \
	     --prefix=$SCRIPT_LOCATION >& /dev/null
make >& /dev/null
make install > /dev/null

# Cleanup
cd $SCRIPT_LOCATION
rm -rf $SOURCE_LOCATION

# Debug enviroment source file
echo "export PATH=$SCRIPT_LOCATION/bin:\$PATH" > debug_eniroment.sh
echo "export LIBRARY_PATH=$SCRIPT_LOCATION/lib:\$LIBRARY_PATH" >> debug_eniroment.sh
echo "export LD_LIBRARY_PATH=$SCRIPT_LOCATION/lib:\$LD_LIBRARY_PATH" >> debug_eniroment.sh
echo "export CPLUS_INCLUDE_PATH=$SCRIPT_LOCATION/include:\$CPLUS_INCLUDE_PATH" >> debug_eniroment.sh
