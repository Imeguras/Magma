ROCM_PATH ?= /opt/rocm
HIPCC ?= hipcc
BUILD_DIR ?= build
CFLAGS  = -std=c++17 -fPIC -Iinclude -I$(ROCM_PATH)/include -O3
LDFLAGS = -shared -L$(ROCM_PATH)/lib -lamdhip64

TARGET = $(BUILD_DIR)/libmagmabiformer-parser.so

SRCS = src/parse.cpp
HIP_SRCS = src/shapes.hip
OBJS = $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS:.cpp=.o)) $(notdir $(HIP_SRCS:.hip=.o)))

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

all: $(BUILD_DIR) $(TARGET)

$(TARGET): $(OBJS)
	$(HIPCC) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/parse.o: src/parse.cpp include/magma_parser_api.h | $(BUILD_DIR)
	$(HIPCC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/shapes.o: src/shapes.hip include/magma_parser_api.h | $(BUILD_DIR)
	$(HIPCC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean legacy
