// SPDX-License-Identifier: GPL-2.0
/*
 * shmem swap-in loop on a large swap-cache folio built from reused slots.
 * file z fills the first cluster of a 4m brd swap device, so file x's 64k
 * folio goes out as one order-4 entry at slots 512..527. the reader swaps
 * x back in; the punch thread, fifo on the same cpu and woken by a timer
 * early in that read, punches x and swaps out pages 1..16 of file y into
 * the freed slots. if it lands between the reader's lookup of x's entry
 * and its swap cache insert, a 64k folio for y's slots stays in the swap
 * cache, and a fault on y page 2 retries forever in shmem_get_folio_gfp():
 * it spins in the kernel, survives sigkill, and a munmap() of its mapping
 * waits in d state (hung task).
 *
 * usage: shmemloop SWAPDEV XDIR YDIR [SECONDS]
 *   SWAPDEV: an mkswap'ed 4m brd device, not in use
 *   XDIR: a tmpfs mounted huge=within_size, YDIR: one mounted huge=never
 * (shmemloop --fault FILE is the faulting child it starts each round)
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/swap.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define PAGE		4096
#define FOLIO		(64 * 1024)
#define Z_PAGES		511	/* cluster 0 minus the swap header */
#define CALIBRATE	8	/* rounds that time an undisturbed read */

static const char *prog, *swapdev;
static char x_path[256], y_path[256];
static int x_fd, y_fd, z_fd;
static pthread_barrier_t start, done;
static long long t_go, delay, read_ns;
static _Atomic int faulting;
static const char *ok = "[+]", *warn = "[~]", *err = "[-]";

static void __attribute__((noreturn)) die(const char *msg)
{
	fprintf(stderr, "%s shmemloop: %s: %s\n", err, msg, strerror(errno));
	exit(2);
}

static long long now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void pin(int cpu)
{
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	if (sched_setaffinity(0, sizeof(set), &set))
		die("sched_setaffinity");
}

static int open_new(const char *path)
{
	int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);

	if (fd < 0)
		die(path);
	return fd;
}

/* non-zero data: zero-filled pages are not written to swap */
static void fill(int fd, off_t off, size_t len, int c)
{
	static char buf[FOLIO];

	memset(buf, c, len);
	if (pwrite(fd, buf, len, off) != (ssize_t)len)
		die("pwrite");
}

/* swap out the pages of fd in [off, off + len) */
static void pageout(int fd, off_t off, size_t len)
{
	volatile char *p = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, off);
	size_t i;

	if (p == MAP_FAILED)
		die("mmap");
	for (i = 0; i < len; i += PAGE)
		(void)p[i];
	if (madvise((void *)p, len, MADV_PAGEOUT))
		die("madvise(MADV_PAGEOUT)");
	munmap((void *)p, len);
}

static long swap_cached_kb(void)
{
	char line[128];
	long kb = -1;
	FILE *f = fopen("/proc/meminfo", "r");

	if (!f)
		die("/proc/meminfo");
	while (fgets(line, sizeof(line), f))
		if (!strncmp(line, "SwapCached:", 11))
			kb = atol(line + 11);
	fclose(f);
	return kb;
}

/* reader: x page 0, which swaps x's 64k folio back in */
static void *reader(void *arg)
{
	char buf[PAGE];

	pin(0);
	for (;;) {
		pthread_barrier_wait(&start);
		while (now_ns() < t_go)
			;
		if (pread(x_fd, buf, PAGE, 0) != PAGE)
			die("pread X");
		read_ns = now_ns() - t_go;
		pthread_barrier_wait(&done);
	}
	return arg;
}

/* punch: preempts the reader, frees x's slots and reuses them for y */
static void *puncher(void *arg)
{
	struct sched_param sp = { .sched_priority = 1 };
	struct timespec ts;

	pin(0);
	if (sched_setscheduler(0, SCHED_FIFO, &sp))
		die("sched_setscheduler");
	for (;;) {
		pthread_barrier_wait(&start);
		ts.tv_sec = (t_go + delay) / 1000000000LL;
		ts.tv_nsec = (t_go + delay) % 1000000000LL;
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
		if (fallocate(x_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, 0, FOLIO))
			die("punch X");
		fill(y_fd, PAGE, FOLIO, 0x59);
		pageout(y_fd, PAGE, FOLIO);
		pthread_barrier_wait(&done);
	}
	return arg;
}

/* swap on, z's pages fill cluster 0 */
static void swap_setup(void)
{
	int i;

	if (swapon(swapdev, 0))
		die("swapon");
	for (i = 0; i < Z_PAGES; i++)
		fill(z_fd, (off_t)i * PAGE, PAGE, 0x5a);
	pageout(z_fd, 0, Z_PAGES * PAGE);
}

static void *unmapper(void *p)
{
	int i;

	for (i = 0; i < 100 && faulting; i++)
		usleep(1000);
	if (faulting)
		munmap(p, 3 * PAGE);	/* waits for the fault's vma lock */
	return NULL;
}

/* the faulting child (shmemloop --fault FILE): read page 2 of FILE, munmap() it from a thread */
static int fault(const char *path)
{
	int fd = open(path, O_RDONLY);
	volatile char *p;
	pthread_t t;

	if (fd < 0)
		die(path);
	p = mmap(NULL, 3 * PAGE, PROT_READ, MAP_SHARED, fd, 0);
	if (p == MAP_FAILED)
		die("mmap Y");
	faulting = 1;
	errno = pthread_create(&t, NULL, unmapper, (void *)p);
	if (errno)
		die("pthread_create");
	(void)p[2 * PAGE];
	faulting = 0;
	pthread_join(t, NULL);
	return 0;
}

/*
 * fault on y page 2 in a child, a fresh exec of this program (we have
 * threads, so a forked copy should not start one); returns its pid if the
 * fault has not returned in 2 s
 */
static pid_t victim(void)
{
	char *argv[] = { (char *)prog, "--fault", y_path, NULL };
	int i, status;
	pid_t pid;

	errno = posix_spawnp(&pid, prog, NULL, NULL, argv, environ);
	if (errno)
		die("posix_spawnp");
	for (i = 0; i < 200; i++) {
		if (waitpid(pid, &status, WNOHANG) == pid) {
			/* the fault returned: exit 0, or a signal after the munmap() */
			if (WIFEXITED(status) && WEXITSTATUS(status)) {
				fprintf(stderr, "%s shmemloop: the faulting child failed\n", err);
				exit(2);
			}
			return 0;
		}
		usleep(10000);
	}
	return pid;
}

/* fields 3 (state), 14 (utime) and 15 (stime) of dir/stat */
static int task_stat(const char *dir, char *state, unsigned long *utime, unsigned long *stime)
{
	char path[128];
	FILE *f;
	int n;

	snprintf(path, sizeof(path), "%s/stat", dir);
	f = fopen(path, "r");
	if (!f)
		return -1;
	n = fscanf(f, "%*d %*s %c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu",
		   state, utime, stime);
	fclose(f);
	return n == 3 ? 0 : -1;
}

static int sigkill_pending(const char *dir)
{
	char path[128], line[128];
	unsigned long long mask = 0;
	FILE *f;

	snprintf(path, sizeof(path), "%s/status", dir);
	f = fopen(path, "r");
	if (!f)
		return 0;
	while (fgets(line, sizeof(line), f))
		if (!strncmp(line, "SigPnd:", 7))
			mask = strtoull(line + 7, NULL, 16);
	fclose(f);
	return !!(mask & (1ULL << (SIGKILL - 1)));
}

/* a thread's state and the function names of its kernel stack, on one line */
static void show_thread(const char *dir)
{
	char path[128], line[256], state = '?', *p;
	unsigned long utime, stime;
	FILE *f;
	int i = 0;

	task_stat(dir, &state, &utime, &stime);
	printf("    thread %s: state %c, kernel stack:", strrchr(dir, '/') + 1, state);
	snprintf(path, sizeof(path), "%s/stack", dir);
	f = fopen(path, "r");
	while (f && fgets(line, sizeof(line), f) && i++ < 14) {
		p = strchr(line, ' ');	/* "[<0>] name+0x1/0x2" */
		if (p)
			printf(" %.*s", (int)strcspn(p + 1, "+\n"), p + 1);
	}
	printf("\n");
	if (f)
		fclose(f);
}

/*
 * the fault never returned: sigkill the child and show that it keeps
 * running in the kernel (it shares our cpu, so its stack is not stale)
 */
static int show_stuck(pid_t pid)
{
	unsigned long u0, s0, u1, s1;
	char dir[32], tdir[64], state;
	struct dirent *de;
	DIR *d;

	snprintf(dir, sizeof(dir), "/proc/%d", pid);
	if (task_stat(dir, &state, &u0, &s0))
		return 0;
	kill(pid, SIGKILL);
	sleep(10);
	if (task_stat(dir, &state, &u1, &s1)) {
		printf("%s shmemloop: pid %d exited after SIGKILL\n", warn, pid);
		return 0;
	}
	printf("shmemloop: 10 s after SIGKILL, pid %d: state %c, user time %lu -> %lu, system time %lu -> %lu ticks, SIGKILL %s\n",
	       pid, state, u0, u1, s0, s1, sigkill_pending(dir) ? "pending" : "not pending");
	snprintf(tdir, sizeof(tdir), "%s/task", dir);
	d = opendir(tdir);
	while (d && (de = readdir(d)))
		if (de->d_name[0] != '.') {
			snprintf(tdir, sizeof(tdir), "%s/task/%.15s", dir, de->d_name);
			show_thread(tdir);
		}
	if (d)
		closedir(d);
	if (state != 'R' || u1 != u0 || s1 < s0 + 100)
		return 0;
	printf("LB-UNKILLABLE pid=%d: still running in the kernel 10 s after SIGKILL\n", pid);
	return 1;
}

/* one race: the reader's swap-in of x against the punch, punch_delay ns later */
static void race(long long punch_delay)
{
	x_fd = open_new(x_path);
	y_fd = open_new(y_path);
	fill(x_fd, 0, FOLIO, 0x58);
	pageout(x_fd, 0, FOLIO);	/* one order-4 entry: slots 512..527 */
	delay = punch_delay;
	t_go = now_ns() + 1000000;
	pthread_barrier_wait(&start);
	pthread_barrier_wait(&done);
}

int main(int argc, char **argv)
{
	int limit = argc > 4 ? atoi(argv[4]) : 600;
	long long t0, read_min = 0;
	char z_path[256];
	pthread_t t1, t2;
	double secs;
	pid_t pid;
	int n;

	if (argc == 3 && !strcmp(argv[1], "--fault"))
		return fault(argv[2]);
	if (argc < 4) {
		fprintf(stderr, "usage: %s SWAPDEV XDIR YDIR [SECONDS]\n", argv[0]);
		return 2;
	}
	setvbuf(stdout, NULL, _IONBF, 0);
	if (isatty(STDOUT_FILENO)) {
		ok = "\033[32m[+]\033[0m";
		warn = "\033[33m[~]\033[0m";
		err = "\033[31m[-]\033[0m";
	}
	prog = argv[0];
	swapdev = argv[1];
	snprintf(x_path, sizeof(x_path), "%s/x", argv[2]);
	snprintf(y_path, sizeof(y_path), "%s/y", argv[3]);
	snprintf(z_path, sizeof(z_path), "%s/z", argv[3]);
	z_fd = open_new(z_path);
	pin(1);
	swap_setup();
	pthread_barrier_init(&start, NULL, 3);
	pthread_barrier_init(&done, NULL, 3);
	pthread_create(&t1, NULL, reader, NULL);
	pthread_create(&t2, NULL, puncher, NULL);
	srand(getpid());
	t0 = now_ns();
	for (n = 1;; n++) {
		/*
		 * the first rounds time an undisturbed read (the punch comes
		 * 10 ms later); then punch at a random point of its first
		 * quarter, where the folio is allocated, before the 64k is read
		 */
		if (n <= CALIBRATE) {
			race(10000000);
			if (!read_min || read_ns < read_min)
				read_min = read_ns;
		} else {
			race(rand() % (read_min / 4 + 1));
		}
		if (n == CALIBRATE)
			printf("shmemloop: an undisturbed read of X takes %lld us\n", read_min / 1000);

		pid = victim();
		if (pid) {
			secs = (now_ns() - t0) / 1e9;
			printf("shmemloop: round %d, after 2 s: the fault on Y page 2 has not returned; sending SIGKILL\n", n);
			if (!show_stuck(pid)) {
				printf("%s shmemloop: the fault was not shown to survive SIGKILL\n", warn);
				return 3;
			}
			printf("%s shmemloop: bug: the fault on Y page 2 never returned and survived SIGKILL (round %d, %.1f s)\n",
			       ok, n, secs);
			printf("LB-RESULT bug rounds=%d seconds=%.1f\n", n, secs);
			return 0;
		}
		close(x_fd);
		close(y_fd);
		unlink(x_path);
		unlink(y_path);
		/* a folio left in the swap cache spoils the slot layout: start over */
		if (swap_cached_kb() > 0) {
			if (swapoff(swapdev))
				die("swapoff");
			swap_setup();
		}
		secs = (now_ns() - t0) / 1e9;
		if (secs > limit) {
			printf("%s shmemloop: no bug in %d rounds, %.1f s\n", ok, n, secs);
			printf("LB-RESULT none rounds=%d seconds=%.1f\n", n, secs);
			return 1;
		}
		if (n % 1000 == 0)
			printf("shmemloop: %d rounds, %.0f s\n", n, secs);
	}
}
