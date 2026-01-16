# Character Map Visualization

一个基于 React Flow 的人物关系图可视化编辑器。

## 安装依赖

```bash
# 进入项目目录
cd visualization/app

# 安装所有依赖
npm install
```

### 核心依赖

| 依赖包 | 版本 | 说明 |
|--------|------|------|
| `@xyflow/react` | ^12.10.0 | React Flow 图形编辑库 |
| `dagre` | ^0.8.5 | 图形自动布局算法 |
| `react-icons` | latest | 图标库 |
| `tailwindcss` | latest | CSS 框架 |
| `@tailwindcss/vite` | latest | Tailwind Vite 插件 |

### 手动安装命令

如果需要单独安装依赖，可以使用以下命令：

```bash
# 安装 React Flow 和自动布局
npm install @xyflow/react dagre

# 安装类型定义
npm install -D @types/dagre

# 安装图标库
npm install react-icons

# 安装 Tailwind CSS
npm install tailwindcss @tailwindcss/vite
```

## 运行项目

```bash
# 开发模式
npm run dev

# 构建生产版本
npm run build

# 预览生产版本
npm run preview
```

## 功能特性

- 🎨 漫画风格的节点和边设计
- ➕ 添加新角色节点
- 🔄 自动布局（基于 Dagre 算法）
- ✏️ 编辑角色属性（名称、重要性、颜色）
- 🔗 编辑关系标签
- 🗑️ 删除节点或连接
- 🖱️ 拖拽创建连接

## 项目结构

```
visualization/app/
├── src/
│   ├── components/
│   │   └── CharacterMapModal.tsx  # 人物关系图组件
│   ├── App.tsx                     # 主应用入口
│   ├── index.css                   # 全局样式 + Tailwind 配置
│   └── main.tsx                    # React 入口
├── vite.config.ts                  # Vite 配置
├── package.json                    # 项目依赖
└── README.md                       # 本文档
```
