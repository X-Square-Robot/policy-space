# 安装与发布

Policy Space 的协议运行时按语义化版本发布。生产评测应使用不可变的 Git 标签、提交哈希或 CI 生成的 wheel；可编辑安装只用于本地开发。

## 本地开发

```bash
git clone <repository-url> policy-space
cd policy-space
pip install -e ".[test]"
pytest -q
```

## 从不可变版本安装

正式版本使用 `v<major>.<minor>.<patch>` 标签，例如：

```bash
pip install "policy-space @ git+ssh://<host>/<namespace>/policy-space.git@v0.3.0"
```

需要完全固定构建输入时，将标签替换为 40 位提交哈希。ManaEnv 的生产依赖采用这种方式，避免分支后续变化影响评测。

## 从 CI wheel 安装

每次 CI 构建都会生成 `dist/` artifact。下载后执行：

```bash
pip install dist/policy_space-0.3.0-py3-none-any.whl
```

仅需协议和契约定义的客户端，可从同一版本安装轻量子包：

```bash
pip install "policy-space-protocol @ git+ssh://<host>/<namespace>/policy-space.git@v0.3.0#subdirectory=packages/protocol"
```

## 发布检查

发布前应完成：

```bash
pytest -q
python -m build
```

确认测试和 wheel 安装检查通过后，再创建与 `pyproject.toml` 版本一致的标签。破坏协议兼容性的修改提升主版本；新增兼容能力提升次版本；兼容性修复提升补丁版本。

## 导出公开仓库

内部仓库保留完整开发历史和既有固定提交。对外发布时，从已验证的标签导出源码树，再在新的空仓库中创建首个提交，避免把内部开发记录带入公开历史：

```bash
mkdir -p <empty-directory>
git archive --format=tar --prefix=policy-space/ v0.3.0 \
  | tar -xf - -C <empty-directory>
cd <empty-directory>/policy-space
git init
git add .
git commit -m "Initial public release: Policy Space v0.3.0"
```

随后为新仓库设置公开远端并推送。公开快照应从 release tag 生成，不从开发分支或内部仓库镜像历史。

[English](release.md)
