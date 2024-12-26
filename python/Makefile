.PHONY: lib, pybind, clean, format, all

all: lib

lib:
	@mkdir -p build
	@cd build; cmake -DUSE_ICC=ON ..
	@cd build; $(MAKE) -j$(nproc --all)

format:
	python3 -m black .
	clang-format -i cpp/*.cpp


clean:
	rm -rf build AutoSparse/backend/*.so
