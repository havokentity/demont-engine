# Parallel-agent coordination

When multiple Claude Code agents run concurrently in worktrees on the
same host machine, they share the GPU and `/tmp/`. Without
coordination, simultaneous `demont` smoke runs can saturate the GPU
and freeze the desktop.

## Smoke-test serialization

Before invoking the `demont` binary (interactive OR `--smoke-frames`),
acquire an atomic lock via `mkdir`. macOS doesn't ship `flock(1)`, so
we use `mkdir` on a lockdir — POSIX-guaranteed atomic, no install
needed, released on shell exit via `trap`.

```bash
# Acquire (poll until directory creation succeeds = atomic mutex)
LOCK=/tmp/demont_test.lockdir
while ! mkdir "$LOCK" 2>/dev/null; do sleep 0.5; done
trap 'rmdir "$LOCK"' EXIT

# Diagnostic logging (optional but recommended)
echo "$(date -Iseconds) <branch> acquired lock" >> /tmp/demont_test_log.txt

# Run demont
build/mac-release/src/app/demont \
    --smoke-frames=N --r-backend=metal \
    --smoke-exec=PATH/TO/fixture.cfg \
    --smoke-capture-out=captures/your_capture.png

echo "$(date -Iseconds) <branch> released lock" >> /tmp/demont_test_log.txt
# trap fires on shell exit and releases the lock
```

Concurrent BUILDS (`cmake --build`) are fine — only serialize the
`demont` invocation itself. The poll interval (0.5 s) is small
relative to typical smoke-test duration (1-5 s) so the contention
window is brief.

If you can't be bothered with the trap, the minimum-viable form is:
```bash
while ! mkdir /tmp/demont_test.lockdir 2>/dev/null; do sleep 0.5; done
build/.../demont --smoke-frames=N ...
rmdir /tmp/demont_test.lockdir
```
But the trap form is safer — it releases the lock even if demont
crashes or the script is SIGINT'd.

## Vulkan / NRD work on Mac

NRD (NVIDIA Real-time Denoisers) integration (#50) targets Vulkan
only. Mac uses Metal. Vulkan PC-specific testing is the user's job
on their Windows box — agents working on Vulkan-only features should
land the PR with a clear "needs PC validation" note rather than block
on testing they can't perform.

## File ownership in parallel batches

When multiple agents are working a single batch of issues, each issue
body lists "files OWNED" and "files NOT touched." Strict adherence
prevents merge conflicts. If you find you need a file outside your
ownership, STOP and document it in the PR body rather than touch it
— the orchestrator will mediate.

## Integration branch

The orchestrator maintains `integration/parallel-batch-N` as a
periodically-rebased branch containing every in-flight feature, for
user testing. Feature PRs target `main`; the orchestrator merges
your branch into the integration branch automatically as you push.
You don't need to interact with it.
