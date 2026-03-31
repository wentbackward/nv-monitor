CC ?= cc
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
USER_BINDIR ?= $(HOME)/.local/bin

COMMON_CFLAGS = -Wall -Wextra -std=gnu11
# Detect Clang even when it is invoked as `cc`; we avoid defaulting to `-flto`
# there to sidestep linker-plugin mismatches on GNU ld-based setups.
CC_VERSION_TEXT := $(shell $(CC) --version 2>/dev/null | sed -n '1p')
ifneq ($(findstring clang,$(CC_VERSION_TEXT)),)
LTO_CFLAGS ?=
else
LTO_CFLAGS ?= -flto
endif

# Project defaults live in the *_CFLAGS variables below.
# Use `make portable` to drop `-march=native`; `CFLAGS=...` adds user flags.
NATIVE_CFLAGS = -O3 -march=native $(LTO_CFLAGS)
# Portable drops `-march=native`; LTO is retained where the compiler supports it.
PORTABLE_CFLAGS = -O3 $(LTO_CFLAGS)
DEMO_CFLAGS = -O2
TEST_CFLAGS = -O0

VERSION_CPPFLAGS = -DVERSION='"$(VERSION)"'
NV_MONITOR_LDLIBS = -lncursesw -ldl -lpthread
# Keep demo-load isolated unless the caller explicitly opts into extra linker flags.
DEMO_LDFLAGS ?=
DEMO_LDLIBS = -lpthread -ldl -lm
# test_meminfo is intentionally self-contained and does not need ncurses/NVML libs.
TEST_LDLIBS =
TARGET  = nv-monitor

all: $(TARGET)

$(TARGET): nv-monitor.c
	$(CC) $(CPPFLAGS) $(VERSION_CPPFLAGS) $(COMMON_CFLAGS) $(NATIVE_CFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS) $(NV_MONITOR_LDLIBS)

demo-load: demo-load.c
	$(CC) $(CPPFLAGS) $(COMMON_CFLAGS) $(DEMO_CFLAGS) $(CFLAGS) -o demo-load demo-load.c $(DEMO_LDFLAGS) $(DEMO_LDLIBS)

portable:
	$(CC) $(CPPFLAGS) $(VERSION_CPPFLAGS) $(COMMON_CFLAGS) $(PORTABLE_CFLAGS) $(CFLAGS) -o $(TARGET) nv-monitor.c $(LDFLAGS) $(LDLIBS) $(NV_MONITOR_LDLIBS)

test: test_meminfo.c
	$(CC) $(COMMON_CFLAGS) $(TEST_CFLAGS) -o test_meminfo test_meminfo.c $(TEST_LDLIBS)
	./test_meminfo

clean:
	rm -f $(TARGET) demo-load test_meminfo

install: $(TARGET)
	install -d $(BINDIR)
	install -m 755 $(TARGET) $(BINDIR)/

install-user: $(TARGET)
	install -d $(USER_BINDIR)
	install -m 755 $(TARGET) $(USER_BINDIR)/

.PHONY: all portable test clean install install-user
