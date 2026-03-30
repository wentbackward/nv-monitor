CC      = gcc
VERSION = $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
CFLAGS  = -O3 -march=native -flto -Wall -Wextra -std=gnu11 -DVERSION='"$(VERSION)"'
CFLAGS_PORTABLE = -O3 -flto -Wall -Wextra -std=gnu11 -DVERSION='"$(VERSION)"'
LDFLAGS = -lncursesw -ldl -lpthread
TARGET  = nv-monitor

all: $(TARGET)

$(TARGET): nv-monitor.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

demo-load: demo-load.c
	$(CC) -O2 -Wall -Wextra -o demo-load demo-load.c -lpthread -ldl -lm

portable:
	$(CC) $(CFLAGS_PORTABLE) -o $(TARGET) nv-monitor.c $(LDFLAGS)

test: test_meminfo.c
	$(CC) -O0 -Wall -Wextra -o test_meminfo test_meminfo.c
	./test_meminfo

clean:
	rm -f $(TARGET) demo-load test_meminfo

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

.PHONY: all portable test clean install
