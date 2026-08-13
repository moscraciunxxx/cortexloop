# CortexLoop — works without CMake (useful on exFAT / network volumes).
CC ?= clang
CXX ?= clang++
CFLAGS ?= -O3 -mcpu=native -fPIC -I native/include -I native/src -std=c11
CXXFLAGS ?= -O3 -mcpu=native -fPIC -I native/include -I native/src -std=c++17
SCALARFLAGS = -O2 -fPIC -fno-vectorize -fno-slp-vectorize -I native/include -I native/src -std=c11
BUILD ?= build
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LIB = $(BUILD)/libcortexloop.dylib
  LDFLAGS = -dynamiclib -install_name @rpath/libcortexloop.dylib
else
  LIB = $(BUILD)/libcortexloop.so
  LDFLAGS = -shared -lm
  CFLAGS += -march=armv8.6-a+dotprod+i8mm
  CXXFLAGS += -march=armv8.6-a+dotprod+i8mm
  SCALARFLAGS += -march=armv8.2-a
endif

OBJS = $(BUILD)/util.o $(BUILD)/scalar.o $(BUILD)/image.o $(BUILD)/gemm.o \
       $(BUILD)/cnn.o $(BUILD)/api.o $(BUILD)/kleidiai_backend.o

.PHONY: all clean bench

all: $(LIB) $(BUILD)/cortexloop-bench

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/scalar.o: native/src/scalar.c | $(BUILD)
	$(CC) $(SCALARFLAGS) -c $< -o $@

$(BUILD)/kleidiai_backend.o: native/src/kleidiai_backend.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: native/src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	$(CXX) $(LDFLAGS) -Wl,-rpath,@loader_path -o $@ $(OBJS)

$(BUILD)/cortexloop-bench: native/src/cli.c $(LIB)
	$(CC) $(CFLAGS) native/src/cli.c -o $@ $(LIB) -Wl,-rpath,@loader_path

clean:
	rm -rf $(BUILD)
