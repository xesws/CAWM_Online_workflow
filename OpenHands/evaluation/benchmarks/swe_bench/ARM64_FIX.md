# SWE-bench ARM64 Docker Image Fix

## Issue

When running SWE-bench evaluation on ARM64 Macs (Apple Silicon), the evaluation fails with 404 errors when trying to pull Docker images.

### Root Cause

1. The SWE-bench harness (v4.0.4) automatically detects ARM64 architecture via `platform.machine()`
2. It attempts to use ARM64-specific Docker images with naming pattern: `swebench/sweb.eval.arm64.*_1776_*:latest`
3. These ARM64 images **do not exist** on Docker Hub - only x86_64 images are available
4. This causes Docker pull failures with 404 "Not Found" errors

### Error Example
```
Error building image astropy__astropy-7746: 404 Client Error for
http+docker://localhost/v1.51/images/create?tag=latest&fromImage=swebench%2Fsweb.eval.arm64.astropy_1776_astropy-7746:
Not Found ("pull access denied for swebench/sweb.eval.arm64.astropy_1776_astropy-7746,
repository does not exist or may require 'docker login'")
```

## Solution Applied

Modified the SWE-bench harness to force x86_64 architecture on ARM64 Macs, allowing the images to run via emulation (Rosetta 2).

### File Modified
```
/Users/tangyiq/Library/Caches/pypoetry/virtualenvs/openhands-ai-RReIwyui-py3.12/lib/python3.12/site-packages/swebench/harness/test_spec/test_spec.py
```

### Changes (Lines 219-223)

**Before:**
```python
if platform.machine() in {"aarch64", "arm64"}:
    # use arm64 unless explicitly specified
    arch = "arm64" if instance_id not in USE_X86 else "x86_64"
else:
    arch = "x86_64"
```

**After:**
```python
# Force x86_64 architecture on ARM64 Macs since ARM64 Docker images
# with the _1776_ naming convention don't exist on Docker Hub
# The x86_64 images will run via emulation (Rosetta 2 on macOS)
arch = "x86_64"
```

### Backup Location
Original file backed up to:
```
/Users/tangyiq/Library/Caches/pypoetry/virtualenvs/openhands-ai-RReIwyui-py3.12/lib/python3.12/site-packages/swebench/harness/test_spec/test_spec.py.backup
```

## Verification

Test run with single instance (`astropy__astropy-7746`):
- **Before fix**: Failed with 404 Docker image error
- **After fix**: Completed successfully in 41 seconds with no errors

```
Total instances: 1
Instances submitted: 1
Instances completed: 1
Instances incomplete: 0
Instances with errors: 0  ✓
```

## Important Notes

1. **Performance**: x86_64 images run via Rosetta 2 emulation on ARM64 Macs. Performance impact is minimal for evaluation purposes.

2. **Persistence**: This fix modifies the installed swebench package. If you reinstall or upgrade swebench, you'll need to reapply this fix.

3. **Alternative Solutions**:
   - Wait for SWE-bench to publish ARM64 Docker images (unlikely to help immediately)
   - Use Docker buildx with platform specification (more complex)
   - Run evaluation on x86_64 hardware (not practical for local development)

## Reapplying After Package Update

If you need to reinstall swebench, reapply this fix:

```bash
# 1. Locate the file
FILE="/Users/tangyiq/Library/Caches/pypoetry/virtualenvs/openhands-ai-RReIwyui-py3.12/lib/python3.12/site-packages/swebench/harness/test_spec/test_spec.py"

# 2. Create backup
cp "$FILE" "$FILE.backup"

# 3. Apply the fix (edit lines 219-223 as shown above)
```

## Date Applied
November 14, 2025

## Related Issues
- SWE-bench harness version: 4.0.4
- Platform: macOS 15.7.2 (ARM64)
- Python: 3.12
