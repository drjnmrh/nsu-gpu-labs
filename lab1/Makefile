build/main: main.cu
	mkdir -p ${dir $@}
	nvcc  -std=c++17 -O3 -DLAB_NUMBER=1 -I../inc main.cu -o build/main

clean:
	rm -rf build
