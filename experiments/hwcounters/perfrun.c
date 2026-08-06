/* perfrun: run a command under inherited perf_event counters and report
 * hardware counters, cgroup CPU-bandwidth throttling deltas, and rusage.
 *
 * Standalone experiment harness for issue #355 step 18. Not part of the
 * library build. Counters are opened on self with inherit=1 and
 * enable_on_exec=1, so they aggregate over the child and every thread it
 * creates -- the same mechanism `perf stat -- cmd` uses.
 *
 *   cc -O2 -o perfrun perfrun.c
 *   ./perfrun <set> <cmd> [args...]      set = core | mem | tlb | none
 *
 * Counter lines go to stderr prefixed PERFRUN; child stdout is untouched.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define HWC(c, o, r) ((uint64_t)(c) | ((uint64_t)(o) << 8) | ((uint64_t)(r) << 16))

typedef struct {
    const char *name;
    uint32_t type;
    uint64_t config;
    int fd;
} Ev;

static Ev core_set[] = {
    {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1},
    {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1},
    {"stalled-cycles-backend", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND, -1},
    {"stalled-cycles-frontend", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND, -1},
    {"task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, -1},
    {"context-switches", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, -1},
    {"cpu-migrations", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, -1},
    {"page-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, -1},
};

static Ev mem_set[] = {
    {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1},
    {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1},
    {"cache-references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, -1},
    {"cache-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, -1},
    {"LLC-load-misses", PERF_TYPE_HW_CACHE,
     HWC(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS), -1},
    {"task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, -1},
};

static Ev tlb_set[] = {
    {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1},
    {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1},
    {"L1-dcache-load-misses", PERF_TYPE_HW_CACHE,
     HWC(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS), -1},
    {"dTLB-load-misses", PERF_TYPE_HW_CACHE,
     HWC(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS), -1},
    {"task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, -1},
};

static long perf_open(struct perf_event_attr *a, pid_t pid, int cpu, int grp,
                      unsigned long flags)
{
    return syscall(__NR_perf_event_open, a, pid, cpu, grp, flags);
}

/* Read one u64 field out of a cgroup stat file ("<key> <value>" lines). */
static long long stat_field(const char *path, const char *key)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char k[128];
    long long v;
    long long out = -1;
    while (fscanf(f, "%127s %lld", k, &v) == 2) {
        if (strcmp(k, key) == 0) { out = v; break; }
    }
    fclose(f);
    return out;
}

static const char *cpu_stat_path(void)
{
    if (access("/sys/fs/cgroup/cpu.stat", R_OK) == 0) return "/sys/fs/cgroup/cpu.stat";
    return "/sys/fs/cgroup/cpu/cpu.stat";
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: perfrun <core|mem|tlb|none> <cmd> [args...]\n");
        return 2;
    }
    Ev *set = NULL;
    int n = 0;
    if (strcmp(argv[1], "core") == 0) { set = core_set; n = sizeof core_set / sizeof *core_set; }
    else if (strcmp(argv[1], "mem") == 0) { set = mem_set; n = sizeof mem_set / sizeof *mem_set; }
    else if (strcmp(argv[1], "tlb") == 0) { set = tlb_set; n = sizeof tlb_set / sizeof *tlb_set; }
    else if (strcmp(argv[1], "none") != 0) { fprintf(stderr, "bad set %s\n", argv[1]); return 2; }

    /* Kernel counters are only legible below paranoid 2; try full first and
     * fall back to user-only so a hardened host still yields a reading. */
    int excl_kernel = 0;
    int opened = 0;
    for (int i = 0; i < n; i++) {
        struct perf_event_attr a;
        memset(&a, 0, sizeof a);
        a.size = sizeof a;
        a.type = set[i].type;
        a.config = set[i].config;
        a.disabled = 1;
        a.inherit = 1;
        a.enable_on_exec = 1;
        a.exclude_hv = 1;
        a.exclude_kernel = excl_kernel;
        a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        long fd = perf_open(&a, 0, -1, -1, 0);
        if (fd < 0 && !excl_kernel && (errno == EACCES || errno == EPERM)) {
            excl_kernel = 1;
            a.exclude_kernel = 1;
            fd = perf_open(&a, 0, -1, -1, 0);
        }
        if (fd < 0) {
            fprintf(stderr, "PERFRUN\topen_fail\t%s\terrno=%d\t%s\n",
                    set[i].name, errno, strerror(errno));
            set[i].fd = -1;
            continue;
        }
        set[i].fd = (int)fd;
        opened++;
    }
    if (n) fprintf(stderr, "PERFRUN\topened\t%d/%d\texclude_kernel=%d\n", opened, n, excl_kernel);

    const char *cs = cpu_stat_path();
    long long thr0 = stat_field(cs, "nr_throttled");
    long long tt0 = stat_field(cs, "throttled_time");
    if (tt0 < 0) tt0 = stat_field(cs, "throttled_usec");
    long long np0 = stat_field(cs, "nr_periods");
    double w0 = now_s();

    pid_t pid = fork();
    if (pid == 0) {
        execvp(argv[2], &argv[2]);
        perror("execvp");
        _exit(127);
    }
    int status = 0;
    struct rusage ru;
    wait4(pid, &status, 0, &ru);

    double wall = now_s() - w0;
    long long thr1 = stat_field(cs, "nr_throttled");
    long long tt1 = stat_field(cs, "throttled_time");
    if (tt1 < 0) tt1 = stat_field(cs, "throttled_usec");
    long long np1 = stat_field(cs, "nr_periods");

    for (int i = 0; i < n; i++) {
        if (set[i].fd < 0) continue;
        uint64_t v[3] = {0, 0, 0};
        if (read(set[i].fd, v, sizeof v) != (ssize_t)sizeof v) {
            fprintf(stderr, "PERFRUN\tread_fail\t%s\terrno=%d\n", set[i].name, errno);
            continue;
        }
        /* Multiplexed events run for a fraction of the window; scale up. */
        double scaled = v[2] ? (double)v[0] * (double)v[1] / (double)v[2] : 0.0;
        fprintf(stderr, "PERFRUN\tcounter\t%s\t%.0f\traw=%llu\tenabled=%llu\trunning=%llu\n",
                set[i].name, scaled, (unsigned long long)v[0],
                (unsigned long long)v[1], (unsigned long long)v[2]);
        close(set[i].fd);
    }

    double utime = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6;
    double stime = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
    fprintf(stderr, "PERFRUN\twall_s\t%.3f\n", wall);
    fprintf(stderr, "PERFRUN\tutime_s\t%.3f\n", utime);
    fprintf(stderr, "PERFRUN\tstime_s\t%.3f\n", stime);
    fprintf(stderr, "PERFRUN\tcpu_per_wall\t%.3f\n", wall > 0 ? (utime + stime) / wall : 0.0);
    fprintf(stderr, "PERFRUN\tnr_periods_delta\t%lld\n", np1 - np0);
    fprintf(stderr, "PERFRUN\tnr_throttled_delta\t%lld\n", thr1 - thr0);
    fprintf(stderr, "PERFRUN\tthrottled_time_delta\t%lld\n", tt1 - tt0);
    fprintf(stderr, "PERFRUN\tmaxrss_kb\t%ld\n", ru.ru_maxrss);
    fprintf(stderr, "PERFRUN\texit\t%d\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
