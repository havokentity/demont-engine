# Parallel-agent coordination

When multiple Claude Code agents run concurrently in worktrees on the
same host machine, they share the GPU and `/tmp/`. Without
coordination, simultaneous `demont` smoke runs can saturate the GPU
and freeze the desktop.

## Smoke-test serialization

Before invoking the `demont` binary (interactive OR `--smoke-frames`),
acquire a kernel-level lock via `flock` against
`/tmp/demont_test.lock`:

```bash
flock /tmp/demont_test.lock build/mac-release/src/app/demont \
    --smoke-frames=N --r-backend=metal \
    --smoke-exec=PATH/TO/fixture.cfg \
    --smoke-capture-out=captures/your_capture.png
```

`flock` is POSIX, FIFO-ordered by the kernel, no busy-wait. The lock
is released the moment your `demont` exits. Concurrent builds
(`cmake --build`) are fine — only serialize the `demont` invocation
itself.

For diagnostic visibility, log entry/exit:

```bash
echo "$(date -Iseconds) <branch> starting smoke" >> /tmp/demont_test_log.txt
flock /tmp/demont_test.lock build/.../demont --smoke-frames=N ...
echo "$(date -Iseconds) <branch> done" >> /tmp/demont_test_log.txt
```

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
