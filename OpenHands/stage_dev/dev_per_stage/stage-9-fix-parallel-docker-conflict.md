# Stage 9: 修复AWM并行运行时Docker镜像冲突问题

## 问题描述

当多个AWM测试进程并行运行时（如在多个terminal窗口运行 `test_awm_fast.py`），处理相同instance时会出现"image already exists"报错。

**根本原因**：多个进程同时检查到镜像不存在，然后同时尝试拉取同一个镜像，导致冲突。

**期望行为**：
- 镜像已存在 → 所有进程直接使用
- 镜像不存在 → 只有一个进程拉取，其他进程等待后直接使用

## 实现方案：镜像操作文件锁

### 核心思路

在 `docker.py` 的 `image_exists()` 方法中添加文件锁，确保镜像拉取操作的原子性：

1. 快速路径：先检查本地镜像是否存在，存在则直接返回（无需锁）
2. 慢速路径：需要拉取时，获取文件锁 → 再次检查（double-check）→ 执行拉取

### 需要修改的文件

#### 1. 新建文件：`openhands/runtime/utils/image_lock.py`

创建 `ImageLock` 类，参考现有的 `port_lock.py` 模式：

```python
"""镜像操作文件锁，防止并行拉取/构建时的竞态条件"""

import hashlib
import os
import tempfile
import time
from typing import Optional

from openhands.core.logger import openhands_logger as logger

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


class ImageLock:
    """针对特定Docker镜像的文件锁"""

    def __init__(self, image_name: str, lock_dir: Optional[str] = None):
        self.image_name = image_name
        self.lock_dir = lock_dir or os.path.join(
            tempfile.gettempdir(), 'openhands_image_locks'
        )
        # 使用MD5哈希生成安全的文件名
        sanitized_name = hashlib.md5(image_name.encode()).hexdigest()
        self.lock_file_path = os.path.join(self.lock_dir, f'image_{sanitized_name}.lock')
        self.lock_fd: Optional[int] = None
        self._locked = False
        os.makedirs(self.lock_dir, exist_ok=True)

    def acquire(self, timeout: float = 300.0) -> bool:
        """获取锁，默认超时5分钟（镜像拉取可能较慢）"""
        if self._locked:
            return True

        try:
            self.lock_fd = os.open(
                self.lock_file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC
            )

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    if HAS_FCNTL:
                        fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._locked = True
                    # 写入调试信息
                    os.write(self.lock_fd, f'{self.image_name}\n{os.getpid()}\n'.encode())
                    logger.debug(f'Acquired lock for image {self.image_name}')
                    return True
                except (OSError, IOError):
                    time.sleep(0.5)

            if self.lock_fd:
                os.close(self.lock_fd)
                self.lock_fd = None
            return False

        except Exception as e:
            logger.debug(f'Failed to acquire lock for image {self.image_name}: {e}')
            return False

    def release(self) -> None:
        """释放锁"""
        if self.lock_fd is not None:
            try:
                if HAS_FCNTL:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                try:
                    os.unlink(self.lock_file_path)
                except FileNotFoundError:
                    pass
                logger.debug(f'Released lock for image {self.image_name}')
            except Exception as e:
                logger.warning(f'Error releasing lock: {e}')
            finally:
                self.lock_fd = None
                self._locked = False

    def __enter__(self) -> 'ImageLock':
        if not self.acquire():
            raise OSError(f'Could not acquire lock for image {self.image_name}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
```

#### 2. 修改文件：`openhands/runtime/builder/docker.py`

在 `image_exists()` 方法（第251-307行）中添加锁机制：

**修改前** (第251-307行):
```python
def image_exists(self, image_name: str, pull_from_repo: bool = True) -> bool:
    # ... 当前实现直接检查并拉取
```

**修改后**:
```python
def image_exists(self, image_name: str, pull_from_repo: bool = True) -> bool:
    """检查镜像是否存在，必要时从registry拉取（带文件锁防止并行冲突）"""
    if not image_name:
        logger.error(f'Invalid image name: `{image_name}`')
        return False

    # 快速路径：先检查本地镜像（无需锁）
    try:
        logger.debug(f'Checking if image exists locally: {image_name}')
        self.docker_client.images.get(image_name)
        logger.debug('Image found locally.')
        return True
    except docker.errors.ImageNotFound:
        pass

    if not pull_from_repo:
        logger.debug(f'Image {image_name} not found locally')
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
            logger.debug('Image found locally (pulled by another process).')
            return True
        except docker.errors.ImageNotFound:
            pass

        # 执行拉取
        try:
            logger.debug('Pulling image, please wait...')
            # ... 现有的拉取逻辑 ...
            return True
        except docker.errors.ImageNotFound:
            return False
        except Exception as e:
            logger.debug(f'Image could not be pulled: {e}')
            return False
    finally:
        lock.release()
```

### 实现步骤

1. **创建 `image_lock.py`**：新建 `openhands/runtime/utils/image_lock.py`
2. **修改 `docker.py`**：在 `image_exists()` 方法中：
   - 添加 import 语句
   - 保留快速路径（本地检查）
   - 在拉取逻辑前获取锁
   - 获取锁后执行 double-check
   - 在 finally 块中释放锁

### 方案优点

1. **从根本上解决竞态**：确保只有一个进程执行拉取
2. **高效**：镜像已存在时无需获取锁
3. **遵循现有模式**：与 `port_lock.py` 风格一致
4. **容错**：锁获取失败时仍可继续执行（降级处理）

### 测试验证

修复后在两个terminal同时运行：
```bash
poetry run python evaluation/awm/tests/test_awm_fast.py --limit 5 --llm-config llm.kimi-k2
```

预期结果：
- 第一个进程拉取镜像，第二个进程等待
- 镜像拉取完成后，两个进程都能正常使用
- 不再出现"image already exists"报错
