rm -rf build
mkdir build
cd build
cmake ..
make -w
./RGB2GRAY ../cinque_terre_small.jpg output.png reference.png
