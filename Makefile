CC      = gcc
CFLAGS  = -O3 -march=native -flto -Wall -Wextra -std=gnu11
CFLAGS_PORTABLE = -O3 -flto -Wall -Wextra -std=gnu11
LDFLAGS = -lncursesw -ldl -lpthread
TARGET  = nv-monitor

all: $(TARGET)

$(TARGET): nv-monitor.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

portable:
	$(CC) $(CFLAGS_PORTABLE) -o $(TARGET) nv-monitor.c $(LDFLAGS)

test: test_meminfo.c
	$(CC) -O0 -Wall -Wextra -o test_meminfo test_meminfo.c
	./test_meminfo

clean:
	rm -f $(TARGET) test_meminfo

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

.PHONY: all portable test clean install
