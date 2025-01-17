#!/bin/bash

# If 2nd argument is provided, then SCI build will be modified. See SCI readme.
NO_REVEAL_OUTPUT=$2

# Now we build all the components.
ROOT="$(pwd)"
#Build Ezpc
cd EzPC/EzPC
eval `opam env`
make


# Build SCI
cd SCI
mkdir -p build
cd build

if [[ "$NO_REVEAL_OUTPUT" == "NO_REVEAL_OUTPUT" ]]; then
	cmake -DCMAKE_INSTALL_PREFIX=./install ../ -DNO_REVEAL_OUTPUT=ON
else
  cmake -DCMAKE_INSTALL_PREFIX=./install ../
fi

cmake --build . --target install --parallel

