# Stage 9: 实现总结 - 修复AWM并行运行时Docker镜像冲突

## 问题描述

当多个AWM测试进程并行运行时（如在多个terminal窗口运行 `test_awm_fast.py`），处理相同instance时会出现"image already exists"报错。

**根本原因**：多个进程同时检查到镜像不存在，然后同时尝试拉取同一个镜像，导致竞态条件冲突。

## 解决方案

在 `docker.py` 的 `image_exists()` 方法中添加文件锁机制，确保镜像拉取操作的原子性。

## 代码修改

### 1. 新建文件：`openhands/runtime/utils/image_lock.py`

创建 `ImageLock` 类，提供基于文件的进程间锁机制：

```python
class ImageLock:
    """针对特定Docker镜像的文件锁"""

    def __init__(self, image_name: str, lock_dir: Optional[str] = None):
        # 使用MD5哈希生成安全的锁文件名
        sanitized_name = hashlib.md5(image_name.encode()).hexdigest()
        self.lock_file_path = os.path.join(lock_dir, f'image_{sanitized_name}.lock')

    def acquire(self, timeout: float = 300.0) -> bool:
        # 使用 fcntl.flock() 获取排他锁
        # 默认超时5分钟（镜像拉取可能较慢）

    def release(self) -> None:
        # 释放锁并清理锁文件
```

**特性**：
- 使用 `fcntl.flock()` 实现跨进程文件锁
- 支持超时机制，防止死锁
- 支持上下文管理器（with语句）
- 锁文件存储在 `/tmp/openhands_image_locks/`

### 2. 修改文件：`openhands/runtime/builder/docker.py`

修改 `image_exists()` 方法（第251-335行），添加锁机制：

**修改前的流程**：
```
检查本地镜像 → 不存在则拉取
```

**修改后的流程**：
```
检查本地镜像（快速路径，无锁）
    ↓ 不存在
获取文件锁
    ↓
再次检查本地镜像（double-check）
    ↓ 仍不存在
执行拉取
    ↓
释放锁（finally块）
```

**关键代码改动**：

```python
def image_exists(self, image_name: str, pull_from_repo: bool = True) -> bool:
    # 快速路径：先检查本地镜像（无需锁）
    try:
        self.docker_client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        pass

    if not pull_from_repo:
        return False

    # 慢速路径：需要拉取，使用文件锁
    from openhands.runtime.utils.image_lock import ImageLock

    lock = ImageLock(image_name)
    if not lock.acquire(timeout=300.0):
        logger.warning(f'Could not acquire lock for image {image_name}, proceeding anyway')

    try:
        # Double-check：其他进程可能已经拉取完成
        try:
            self.docker_client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            pass

        # 执行拉取...
        return True
    finally:
        lock.release()
```

## 设计考量

### 为什么使用文件锁？

1. **跨进程同步**：多个独立的Python进程需要协调
2. **无需额外依赖**：使用标准库 `fcntl`（Unix系统内置）
3. **遵循现有模式**：与codebase中的 `port_lock.py` 风格一致

### 为什么使用 double-check 模式？

当进程B等待进程A释放锁时，进程A可能已经完成了镜像拉取。获取锁后再次检查可以避免重复拉取。

### 为什么锁获取失败时仍继续执行？

作为降级策略，即使锁机制失败（如文件系统问题），程序仍尝试继续执行，而不是直接报错退出。

## 测试验证

修复后在两个terminal同时运行：
```bash
# Terminal 1
poetry run python evaluation/awm/tests/test_awm_fast.py --limit 5 --llm-config llm.kimi-k2

# Terminal 2
poetry run python evaluation/awm/tests/test_awm_fast.py --limit 5 --llm-config llm.kimi-k2
```

**预期行为**：
- 第一个进程获得锁并拉取镜像
- 第二个进程等待锁释放
- 锁释放后，第二个进程发现镜像已存在，直接使用
- 不再出现"image already exists"报错

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `openhands/runtime/utils/image_lock.py` | 新建 | ImageLock类实现 |
| `openhands/runtime/builder/docker.py` | 修改 | image_exists()方法添加锁（防止pull竞态） |
| `openhands/runtime/utils/runtime_build.py` | 修改 | _build_sandbox_image()函数添加锁（防止build竞态） |

---

## 补充修复：镜像构建（build）竞态

### 问题

初始修复只覆盖了镜像**拉取（pull）**的竞态，但**构建（build）**仍存在竞态：

```
ERROR: failed to build: failed to solve: image "ghcr.io/openhands/runtime:oh_v0.62.0_xxx": already exists
```

### 修复

在 `_build_sandbox_image()` 函数中添加同样的锁机制：

```python
def _build_sandbox_image(...) -> str:
    from openhands.runtime.utils.image_lock import ImageLock

    primary_image_name = f'{runtime_image_repo}:{source_tag}'

    lock = ImageLock(primary_image_name)
    if not lock.acquire(timeout=600.0):  # 10分钟超时
        logger.warning(f'Could not acquire lock...')

    try:
        # Double-check: 其他进程可能已构建完成
        if runtime_builder.image_exists(primary_image_name, pull_from_repo=False):
            return primary_image_name

        # 构建镜像...
    finally:
        lock.release()
```

### 完整保护

现在两种场景都有锁保护：

| 操作 | 位置 | 状态 |
|------|------|------|
| 镜像拉取 (pull) | `docker.py` → `image_exists()` | ✅ 已修复 |
| 镜像构建 (build) | `runtime_build.py` → `_build_sandbox_image()` | ✅ 已修复 |
