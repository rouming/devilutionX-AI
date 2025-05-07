#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>

#include <string>

#include "utils/ring.h"
#include "utils/shared.h"

namespace devilution {

/*
 * Here we define all variables which should be shared
 */
namespace shared {
    struct ring_queue input_queue;
    struct ring_queue events_queue;
    struct player_state player;
    uint64_t game_ticks;
    uint64_t game_saves;
    uint64_t game_loads;
}

/**
 * share_diablo_state() - function finds the first mapped region with
 * a valid path pointing to `this` executable binary and 'rw-p'
 * permissions, indicating that this region corresponds to the segment
 * that includes sections such as .init_*, .data, etc. The next
 * adjacent region should be anonymous and corresponds to .bss, also
 * with 'rw-p' permissions. These two regions make up the application
 * data, which we can safely remap to a @mshared_path file, allowing a
 * third-party application to attach to it.
 */
void shared::share_diablo_state(const std::string &mshared_path)
{
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    exe_path[len] = '\0';

    FILE *fp = fopen("/proc/self/maps", "r");
    if (!fp) {
        perror("fopen");
        exit(1);
    }

    uintptr_t start = 0, end = 0;
    char line[4096];
    bool found = false;

    while (fgets(line, sizeof(line), fp)) {
        uintptr_t s, e;
        char perms[5];
        char path[256];
        int n;

        path[0] = '\0';
        perms[0] = '\0';

        n = sscanf(line, "%lx-%lx %4s %*s %*s %*s %[^\n]", &s, &e, perms, path);
        if (n < 3) {
            // Wow, can't parse maps?
            fprintf(stderr, "Can't parse maps\n");
            fclose(fp);
            exit(1);
        }
        if (!found && !strcmp(perms, "rw-p") && !strcmp(path, exe_path)) {
            start = s;
            end = e;
            found = true;
        } else if (found) {
            if (strcmp(perms, "rw-p") || path[0] != '\0' || s != end) {
                // Expect contiguous regions
                fprintf(stderr, "Can't find correct mapped region\n");
                fclose(fp);
                exit(1);
            }
            end = e;
            break;
        }
    }
    fclose(fp);

    if (!start || !end || end <= start) {
        fprintf(stderr, "Could not find suitable contiguous mappings\n");
        exit(1);
    }

    int fd = open(mshared_path.c_str(), O_CREAT | O_RDWR,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    if (fd < 0) {
        perror("open: ");
        exit(1);
    }

    size_t length = end - start;
    if (ftruncate(fd, length) < 0) {
        perror("ftruncate");
        close(fd);
        exit(1);
    }

    // Write current memory content to the file
    if (write(fd, (void *)start, length) != (ssize_t)length) {
        perror("write");
        close(fd);
        exit(1);
    }

    // Remap the entire region as shared
    void *new_map = mmap((void *)start, length, PROT_READ | PROT_WRITE,
                         MAP_FIXED | MAP_SHARED, fd, 0);
    if (new_map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        exit(1);
    }

    printf("Remapped region %lx-%lx (length %lx) as MAP_SHARED\n",
           start, end, length);
    close(fd);
}

}
