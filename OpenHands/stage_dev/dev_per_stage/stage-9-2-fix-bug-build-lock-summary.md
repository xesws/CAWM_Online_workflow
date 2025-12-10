# Stage 9-2: Fix Image Build Race Condition

## Bug Description

After fixing the image **pull** race condition in Stage 9-1, parallel AWM tests still failed with:

```
#27 ERROR: image "ghcr.io/openhands/runtime:oh_v0.62.0_gwwvpuwt73mfjwrg_fh5avyztd8j2uwr9": already exists
------
ERROR: failed to build: failed to solve: image "ghcr.io/openhands/runtime:oh_v0.62.0_gwwvpuwt73mfjwrg_fh5avyztd8j2uwr9": already exists
```

## Root Cause

Stage 9-1 only fixed the **pull** race condition in `image_exists()`, but the **build** race condition in `_build_sandbox_image()` was not addressed.

**Race condition flow:**
1. Process A checks image doesn't exist, starts `docker buildx build`
2. Process B also checks image doesn't exist, also starts `docker buildx build`
3. Process A finishes first, image loaded successfully
4. Process B finishes, tries to load image but it already exists, error occurs

## Solution

Add the same file lock mechanism to `_build_sandbox_image()` as was added to `image_exists()`.

## Code Changes

### Modified file: `openhands/runtime/utils/runtime_build.py`

**Before** (lines 363-391):
```python
def _build_sandbox_image(
    build_folder: Path,
    runtime_builder: RuntimeBuilder,
    runtime_image_repo: str,
    source_tag: str,
    lock_tag: str,
    versioned_tag: str | None,
    platform: str | None = None,
    extra_build_args: list[str] | None = None,
) -> str:
    """Build and tag the sandbox image."""
    names = [
        f'{runtime_image_repo}:{source_tag}',
        f'{runtime_image_repo}:{lock_tag}',
    ]
    if versioned_tag is not None:
        names.append(f'{runtime_image_repo}:{versioned_tag}')
    names = [name for name in names if not runtime_builder.image_exists(name, False)]

    image_name = runtime_builder.build(
        path=str(build_folder),
        tags=names,
        platform=platform,
        extra_build_args=extra_build_args,
    )
    if not image_name:
        raise AgentRuntimeBuildError(f'Build failed for image {names}')

    return image_name
```

**After:**
```python
def _build_sandbox_image(
    build_folder: Path,
    runtime_builder: RuntimeBuilder,
    runtime_image_repo: str,
    source_tag: str,
    lock_tag: str,
    versioned_tag: str | None,
    platform: str | None = None,
    extra_build_args: list[str] | None = None,
) -> str:
    """Build and tag the sandbox image.

    Uses file-based locking to prevent race conditions when multiple processes
    try to build the same image simultaneously.
    """
    from openhands.runtime.utils.image_lock import ImageLock

    primary_image_name = f'{runtime_image_repo}:{source_tag}'

    # Use file lock to prevent multiple processes building the same image
    lock = ImageLock(primary_image_name)
    if not lock.acquire(timeout=600.0):  # 10 minutes for build operations
        logger.warning(
            f'Could not acquire lock for building image {primary_image_name}, proceeding anyway'
        )

    try:
        # Double-check: another process may have built the image while we waited
        if runtime_builder.image_exists(primary_image_name, pull_from_repo=False):
            logger.info(
                f'Image {primary_image_name} already exists (built by another process)'
            )
            return primary_image_name

        # Build the image with all tags that do not yet exist
        names = [
            f'{runtime_image_repo}:{source_tag}',
            f'{runtime_image_repo}:{lock_tag}',
        ]
        if versioned_tag is not None:
            names.append(f'{runtime_image_repo}:{versioned_tag}')
        names = [
            name for name in names if not runtime_builder.image_exists(name, False)
        ]

        if not names:
            logger.info('All image tags already exist, skipping build')
            return primary_image_name

        image_name = runtime_builder.build(
            path=str(build_folder),
            tags=names,
            platform=platform,
            extra_build_args=extra_build_args,
        )
        if not image_name:
            raise AgentRuntimeBuildError(f'Build failed for image {names}')

        return image_name
    finally:
        lock.release()
```

## Key Changes

1. **Import ImageLock**: Reuse the lock class created in Stage 9-1
2. **Acquire lock**: Get file lock before building, timeout set to 10 minutes (builds take longer than pulls)
3. **Double-check**: After acquiring lock, check if image already exists (another process may have finished)
4. **Early return**: If all tags already exist, return immediately without building
5. **Finally release**: Ensure lock is released in all cases

## Stage 9 Complete Protection Summary

| Operation | File | Function | Stage |
|-----------|------|----------|-------|
| Image pull | `docker.py` | `image_exists()` | Stage 9-1 |
| Image build | `runtime_build.py` | `_build_sandbox_image()` | Stage 9-2 |

## Test Verification

After fix, run in two terminals simultaneously:
```bash
poetry run python evaluation/awm/tests/test_awm_fast.py --limit 5 --llm-config llm.kimi-k2
```

Expected behavior:
- Process A acquires lock, starts building runtime image
- Process B waits for lock
- Process A finishes build, releases lock
- Process B acquires lock, double-check finds image already exists, returns immediately
- Both processes continue normally, no more "already exists" errors
