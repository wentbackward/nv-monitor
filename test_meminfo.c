/*
 * Tests for meminfo_calc() — the derived memory calculation logic.
 *
 * Build & run: gcc -O0 -Wall -o test_meminfo test_meminfo.c && ./test_meminfo
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Pull in just the struct and calc function */
typedef struct {
    unsigned long long total_kb;
    unsigned long long free_kb;
    unsigned long long avail_kb;
    unsigned long long buffers_kb;
    unsigned long long cached_kb;
    unsigned long long swap_total_kb;
    unsigned long long swap_free_kb;
    unsigned long long app_kb;
    unsigned long long bufcache_kb;
    unsigned long long swap_used_kb;
} MemInfo;

static void meminfo_calc(MemInfo *m) {
    m->bufcache_kb = m->buffers_kb + m->cached_kb;
    m->app_kb = (m->total_kb > m->free_kb + m->bufcache_kb)
              ? m->total_kb - m->free_kb - m->bufcache_kb : 0;
    m->swap_used_kb = (m->swap_total_kb > m->swap_free_kb)
                    ? m->swap_total_kb - m->swap_free_kb : 0;
}

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_EQ(name, got, expected) do { \
    tests_run++; \
    if ((got) == (expected)) { \
        tests_passed++; \
    } else { \
        printf("FAIL: %s: expected %llu, got %llu\n", \
               name, (unsigned long long)(expected), (unsigned long long)(got)); \
    } \
} while(0)

static void test_typical_system(void) {
    /* Typical DGX Spark: 128G total, 44G free, 81G buf/cache */
    MemInfo m = {0};
    m.total_kb      = 127601388;
    m.free_kb       = 43931308;
    m.buffers_kb    = 1218328;
    m.cached_kb     = 80000000;
    m.swap_total_kb = 16777216;
    m.swap_free_kb  = 16777200;
    meminfo_calc(&m);

    ASSERT_EQ("typical: bufcache", m.bufcache_kb, 81218328ULL);
    ASSERT_EQ("typical: app", m.app_kb, 127601388ULL - 43931308ULL - 81218328ULL);
    ASSERT_EQ("typical: swap_used", m.swap_used_kb, 16ULL);
}

static void test_low_memory(void) {
    /* System under pressure: little free, lots of app usage */
    MemInfo m = {0};
    m.total_kb      = 127601388;
    m.free_kb       = 500000;
    m.buffers_kb    = 200000;
    m.cached_kb     = 2000000;
    m.swap_total_kb = 16777216;
    m.swap_free_kb  = 8000000;
    meminfo_calc(&m);

    ASSERT_EQ("lowmem: bufcache", m.bufcache_kb, 2200000ULL);
    ASSERT_EQ("lowmem: app", m.app_kb, 127601388ULL - 500000ULL - 2200000ULL);
    ASSERT_EQ("lowmem: swap_used", m.swap_used_kb, 16777216ULL - 8000000ULL);
}

static void test_no_swap(void) {
    MemInfo m = {0};
    m.total_kb      = 8000000;
    m.free_kb       = 2000000;
    m.buffers_kb    = 500000;
    m.cached_kb     = 1500000;
    m.swap_total_kb = 0;
    m.swap_free_kb  = 0;
    meminfo_calc(&m);

    ASSERT_EQ("noswap: bufcache", m.bufcache_kb, 2000000ULL);
    ASSERT_EQ("noswap: app", m.app_kb, 4000000ULL);
    ASSERT_EQ("noswap: swap_used", m.swap_used_kb, 0ULL);
}

static void test_mostly_cache(void) {
    /* Lots of cache, very little app usage — the original bug scenario */
    MemInfo m = {0};
    m.total_kb      = 127601388;
    m.free_kb       = 43000000;
    m.buffers_kb    = 1000000;
    m.cached_kb     = 80000000;
    m.swap_total_kb = 16777216;
    m.swap_free_kb  = 16777216;
    meminfo_calc(&m);

    ASSERT_EQ("cache-heavy: bufcache", m.bufcache_kb, 81000000ULL);
    ASSERT_EQ("cache-heavy: app", m.app_kb, 127601388ULL - 43000000ULL - 81000000ULL);
    /* app should be small relative to total */
    ASSERT_EQ("cache-heavy: app < 5%", m.app_kb < m.total_kb / 20, 1ULL);
    ASSERT_EQ("cache-heavy: swap_used", m.swap_used_kb, 0ULL);
}

static void test_bufcache_exceeds_used(void) {
    /* Edge case: free + bufcache > total (shouldn't happen but guard against underflow) */
    MemInfo m = {0};
    m.total_kb   = 8000000;
    m.free_kb    = 5000000;
    m.buffers_kb = 2000000;
    m.cached_kb  = 3000000;
    meminfo_calc(&m);

    ASSERT_EQ("underflow: app clamped to 0", m.app_kb, 0ULL);
}

static void test_zero_total(void) {
    /* Edge case: all zeros (e.g. /proc/meminfo unreadable) */
    MemInfo m = {0};
    meminfo_calc(&m);

    ASSERT_EQ("zero: app", m.app_kb, 0ULL);
    ASSERT_EQ("zero: bufcache", m.bufcache_kb, 0ULL);
    ASSERT_EQ("zero: swap_used", m.swap_used_kb, 0ULL);
}

static void test_hugepages_scenario(void) {
    /* Simulates post-HugePages adjustment: avail overwritten, swap zeroed */
    MemInfo m = {0};
    m.total_kb      = 127601388;
    m.free_kb       = 1000000;
    m.avail_kb      = 50000000;  /* overwritten by hugepages logic */
    m.buffers_kb    = 500000;
    m.cached_kb     = 1000000;
    m.swap_total_kb = 16777216;
    m.swap_free_kb  = 16777216;  /* zeroed by hugepages logic */
    meminfo_calc(&m);

    ASSERT_EQ("hugepages: bufcache", m.bufcache_kb, 1500000ULL);
    ASSERT_EQ("hugepages: app", m.app_kb, 127601388ULL - 1000000ULL - 1500000ULL);
    ASSERT_EQ("hugepages: swap_used", m.swap_used_kb, 0ULL);
}

int main(void) {
    test_typical_system();
    test_low_memory();
    test_no_swap();
    test_mostly_cache();
    test_bufcache_exceeds_used();
    test_zero_total();
    test_hugepages_scenario();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
