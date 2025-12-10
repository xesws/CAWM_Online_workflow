# Stage 7: Docker External Storage Configuration

> 任务: 将 Docker Desktop 的数据存储位置移动到外置移动硬盘，解决本机存储空间不足问题
> 状态: **已完成** ✅
> 创建日期: 2025-11-30
> 完成日期: 2025-11-30

---

## 1. 背景

### 1.1 问题

在运行 AWM 测试时，Docker 会创建大量镜像和容器，导致本机存储空间快速耗尽。之前的测试中，Docker.raw 文件曾达到 926GB，占满了整个磁盘。

### 1.2 解决方案

将 Docker 数据目录移动到外置移动硬盘 (OWC Envoy Ultra)，从而：
- 避免本机存储空间爆炸
- 外置硬盘容量更大，可以支持更多测试

---

## 2. 环境信息

| 项目 | 值 |
|------|---|
| 外置硬盘 | OWC Envoy Ultra |
| 挂载路径 | `/Volumes/OWC Envoy Ultra` |
| 文件系统 | APFS (支持 symlink) |
| 连接方式 | 始终连接 |
| 接口 | Thunderbolt 5 (速度 > 6000MB/s) |

---

## 3. 方案选择

### 推荐方案: Symlink 方法

**原因:**
1. Docker Desktop GUI 的 "Disk image location" 设置有已知 bug，会导致 "cross-device link" 错误 ([GitHub Issue #7310](https://github.com/docker/for-mac/issues/7310))
2. Symlink 方法是社区验证的最可靠方案 ([DEV Community Guide](https://dev.to/felipebelluco/how-to-move-docker-data-to-an-external-drive-on-macos-4h79))
3. APFS 格式完全支持 symlink

---

## 4. 执行步骤

### 前置条件
- [ ] Docker Desktop 已退出
- [ ] 之前的 Docker.raw 已删除
- [ ] 外置硬盘已挂载

### Step 1: 确认 Docker 数据目录位置

```bash
# 检查当前 Docker Desktop 版本
# 版本 4.3.0+: ~/Library/Group Containers/group.com.docker
# 旧版本: ~/Library/Containers/com.docker.docker/Data/

ls -la ~/Library/Group\ Containers/ | grep docker
ls -la ~/Library/Containers/ | grep docker
```

### Step 2: 在外置硬盘上创建 Docker 数据目录

```bash
# 创建目标目录 (注意路径中的空格需要转义)
mkdir -p "/Volumes/OWC Envoy Ultra/DockerData"
```

### Step 3: 移动现有数据 (如果有)

```bash
# 如果存在 group.com.docker 目录，移动它
# 由于之前删除了 Docker.raw，这个目录可能是空的或不存在

# 检查是否存在
ls -la ~/Library/Group\ Containers/group.com.docker/ 2>/dev/null

# 如果存在且有数据，移动到外置硬盘
mv ~/Library/Group\ Containers/group.com.docker "/Volumes/OWC Envoy Ultra/DockerData/"
```

### Step 4: 创建符号链接

```bash
# 创建 symlink 指向外置硬盘
ln -s "/Volumes/OWC Envoy Ultra/DockerData/group.com.docker" ~/Library/Group\ Containers/group.com.docker
```

### Step 5: 验证 symlink

```bash
# 确认 symlink 正确
ls -la ~/Library/Group\ Containers/ | grep docker
# 应该显示: group.com.docker -> /Volumes/OWC Envoy Ultra/DockerData/group.com.docker
```

### Step 6: 启动 Docker Desktop 并验证

1. 启动 Docker Desktop
2. 等待 Docker 启动完成
3. 运行测试命令:

```bash
docker run hello-world
```

4. 检查数据是否写入外置硬盘:

```bash
ls -la "/Volumes/OWC Envoy Ultra/DockerData/group.com.docker/"
du -sh "/Volumes/OWC Envoy Ultra/DockerData/"
```

---

## 5. 重要注意事项

### 5.1 风险提示

| 风险 | 说明 | 缓解措施 |
|------|------|----------|
| 硬盘断开 | 如果外置硬盘断开，Docker 会失败 | 确保硬盘始终连接，设置自动挂载 |
| 性能差异 | 外置硬盘速度可能略低于内置 SSD | OWC Envoy Ultra 是 Thunderbolt 5 SSD，性能足够 |
| 路径空格 | `/Volumes/OWC Envoy Ultra` 包含空格 | 所有命令中使用引号包裹路径 |

### 5.2 如何回滚

如果需要恢复到内置存储：

```bash
# 1. 退出 Docker Desktop
# 2. 删除 symlink
rm ~/Library/Group\ Containers/group.com.docker

# 3. 移动数据回内置存储
mv "/Volumes/OWC Envoy Ultra/DockerData/group.com.docker" ~/Library/Group\ Containers/

# 4. 重启 Docker Desktop
```

---

## 6. 后续建议

### 6.1 设置硬盘自动挂载 (可选)

确保 OWC Envoy Ultra 在系统启动时自动挂载，避免 Docker 启动失败。

### 6.2 监控磁盘空间

```bash
# 定期检查 Docker 占用空间
du -sh "/Volumes/OWC Envoy Ultra/DockerData/"

# 清理不需要的镜像和容器
docker system prune -a --volumes
```

---

## 7. 与其他 Stage 的关系

```
Stage 1: Infrastructure (single_inference.py, single_evaluation.py)
    ↓
Stage 2: Pipeline + Buffer
    ↓
Stage 3: Induction + Memory
    ↓
Stage 4: Online Loop
    ↓
Stage 5: Log Handler
    ↓
Stage 6: Workflow Injection
    ↓
Stage 6.1: Evaluation Permission Fix (Bug Fix)
    ↓
Stage 7: Docker External Storage (本阶段) ← 运维优化
```

**依赖关系**:
- Stage 7 是运维配置，不依赖其他 Stage 的代码
- 所有需要运行 Docker 的 AWM 测试都会受益于这个配置

---

## 8. 参考资料

- [How to Move Docker Data to an External Drive on MacOS](https://dev.to/felipebelluco/how-to-move-docker-data-to-an-external-drive-on-macos-4h79)
- [Docker for Mac Issue #7310 - cross-device link error](https://github.com/docker/for-mac/issues/7310)
- [OWC Envoy Ultra Product Page](https://eshop.macsales.com/shop/owc-envoy-ultra)

---

## 9. 验收标准

| 项目 | 预期结果 |
|------|----------|
| Symlink 创建 | `ls -la` 显示正确的符号链接 |
| Docker 启动 | Docker Desktop 正常启动 |
| 数据写入位置 | `du -sh` 显示数据写入外置硬盘 |
| hello-world 测试 | `docker run hello-world` 成功 |
| AWM 测试 | AWM 测试正常运行，数据存储在外置硬盘 |
