# GitHub 仓库连接指南

## 方法一：使用 Cursor 的 Git 集成（推荐）

### 步骤 1: 在 GitHub 上创建新仓库

1. 登录 [GitHub](https://github.com)
2. 点击右上角的 **+** 号，选择 **New repository**
3. 填写仓库信息：
   - **Repository name**: `us_stock_predict` (或你喜欢的名字)
   - **Description**: 美股预测系统 / US Stock Prediction System
   - **Visibility**: 选择 **Public** (公开) 或 **Private** (私有)
   - **不要**勾选 "Initialize this repository with a README"（因为本地已有代码）
4. 点击 **Create repository**

### 步骤 2: 在 Cursor 中连接 GitHub

#### 方式 A: 使用 Cursor 的源代码管理面板

1. 在 Cursor 左侧边栏，点击 **源代码管理** 图标（或按 `Ctrl+Shift+G`）
2. 你会看到所有未提交的文件
3. 点击右上角的 **...** 菜单
4. 选择 **Remote** → **Add Remote**
5. 输入远程仓库名称：`origin`
6. 输入 GitHub 仓库 URL（例如：`https://github.com/你的用户名/us_stock_predict.git`）

#### 方式 B: 使用终端命令（更直接）

在 Cursor 的终端中运行以下命令（替换为你的 GitHub 用户名和仓库名）：

```bash
# 添加远程仓库
git remote add origin https://github.com/你的用户名/us_stock_predict.git

# 或者使用 SSH（如果你配置了 SSH 密钥）
git remote add origin git@github.com:你的用户名/us_stock_predict.git
```

### 步骤 3: 提交并推送代码

```bash
# 1. 添加所有文件到暂存区
git add .

# 2. 创建初始提交
git commit -m "Initial commit: US Stock Prediction System"

# 3. 设置主分支名称（如果还没有）
git branch -M main

# 4. 推送到 GitHub
git push -u origin main
```

## 方法二：使用 GitHub CLI（如果已安装）

```bash
# 1. 登录 GitHub CLI
gh auth login

# 2. 创建并推送仓库
gh repo create us_stock_predict --public --source=. --remote=origin --push
```

## 常见问题

### 问题 1: 需要身份验证

如果推送时要求输入用户名和密码，建议：

1. **使用 Personal Access Token (推荐)**:
   - 在 GitHub 设置中生成 Token
   - Settings → Developer settings → Personal access tokens → Tokens (classic)
   - 生成新 token，勾选 `repo` 权限
   - 推送时，用户名输入你的 GitHub 用户名，密码输入 token

2. **使用 SSH 密钥（更安全）**:
   ```bash
   # 生成 SSH 密钥（如果还没有）
   ssh-keygen -t ed25519 -C "your_email@example.com"
   
   # 复制公钥到剪贴板
   cat ~/.ssh/id_ed25519.pub
   
   # 在 GitHub 上添加 SSH 密钥
   # Settings → SSH and GPG keys → New SSH key
   
   # 使用 SSH URL 添加远程仓库
   git remote set-url origin git@github.com:你的用户名/us_stock_predict.git
   ```

### 问题 2: 文件太大无法推送

如果某些文件太大（如模型文件），确保 `.gitignore` 已正确配置：

```bash
# 检查哪些大文件被跟踪
git ls-files | xargs ls -lh | sort -k5 -hr | head -20

# 如果大文件已被提交，需要从 Git 历史中移除
git rm --cached results/models/*.joblib
git commit -m "Remove large model files"
```

### 问题 3: 推送被拒绝

如果提示 "Updates were rejected"，可能是因为：
- 远程仓库有初始提交（README 等）
- 解决方式：
  ```bash
  git pull origin main --allow-unrelated-histories
  # 解决可能的冲突后
  git push -u origin main
  ```

## 验证连接

推送成功后，在浏览器中打开你的 GitHub 仓库，应该能看到所有代码文件。

## 后续操作

### 日常提交和推送

```bash
# 1. 查看更改
git status

# 2. 添加更改的文件
git add .

# 3. 提交更改
git commit -m "描述你的更改"

# 4. 推送到 GitHub
git push
```

### 在 Cursor 中使用 Git

Cursor 提供了可视化的 Git 操作：
- **源代码管理面板**: 查看更改、暂存文件、提交
- **分支管理**: 创建和切换分支
- **合并冲突**: 可视化解决冲突

## 注意事项

⚠️ **重要**: 确保 `.gitignore` 文件已正确配置，避免推送敏感信息和大文件！

