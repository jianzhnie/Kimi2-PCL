---
alwaysApply: true
scene: git_message
---

请遵循以下规则来生成 Git 提交消息：

### 1. 格式规范
提交消息应遵循 Conventional Commits 规范，格式如下：
`<type>(<scope>): <subject>`

### 2. 类型 (Type)
- `feat`: 新功能 (feature)
- `fix`: 修补 bug
- `docs`: 文档变更 (documentation)
- `style`: 格式化 (不影响代码运行的变动)
- `refactor`: 重构 (即不是新增功能，也不是修改 bug 的代码变动)
- `perf`: 性能优化 (performance)
- `test`: 增加测试
- `chore`: 构建过程或辅助工具的变动
- `revert`: 回滚到上一个版本

### 3. 范围 (Scope) - 可选
说明 commit 影响的范围，例如 `scripts`, `utils`, `core` 等。

### 4. 主题 (Subject)
- 使用小写字母开头。
- 尽量简洁，控制在 50 个字符以内。
- 结尾不加句号。
- 描述应当清晰地表达此次更改的目的。

### 5. 正文 (Body) - 可选
如果更改较为复杂，可以在正文中详细说明修改的原因以及主要逻辑。

### 示例
- `feat(scripts): add ckpt_convert_hf2mcore.sh for weight conversion`
- `fix(utils): update moe structure detection logic`
- `style: format code for better readability`
