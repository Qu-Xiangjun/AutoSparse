.PHONY: lib, pybind, clean, all

all: lib


lib:
	@mkdir -p build
	@cd build; cmake -DUSE_ICC=ON ..
	@cd build; $(MAKE)

clean:
	rm -rf build ../AutoSparse/backend/*.so
