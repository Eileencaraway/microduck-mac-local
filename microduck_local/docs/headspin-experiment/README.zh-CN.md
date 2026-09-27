# Microduck 头转实验

[English README](README.md)

这个分支在 [Microduck Lab](https://github.com/jonathanhawkins/microduck-lab)
中新增了两个实验性强化学习动作：

- `headspin`：进入头倒立后，在头部支撑状态下学习固定方向、肉眼可见的旋转。
- `headspin_launch`：从站立开始建立绕竖直轴的动力，转换到头部支撑，并持续快速旋转。一圈即算成功，第二到第四圈获得逐渐更高的奖励，不再要求特定收尾姿势。

## 当前状态

**这是仿真研究原型。** 仓库包含新的动作定义、训练课程、模型迁移逻辑、测试和确定性评估工具，但不包含经过实体机器人验证的控制器。

现有结果尚不能证明已经学会稳定完整的头转：

| 模型 | 确定性评估条件 | 结果 |
| --- | --- | --- |
| `teach-headspin-f95723` | 3 次站立起点 + 3 次倒立起点 | 0/6 连续完整一圈；最佳物理代理角度约 225° |
| `teach-headspin_launch-e743c8-s4` | 6 次静止站立起点、零旋转辅助 | 0/6 完成启动、旋转和收尾；躯干/头部最佳代理旋转约 0.38/0.54 圈 |
| `teach-headspin_launch-9406bd` | 20 次静止站立起点、零旋转辅助 | 11/20 超过连续一圈，1/20 超过两圈；头部和躯干共同旋转最佳约 2.11 圈 |

这些结果使用接触和角速度积分作为仿真代理，仍需观看回放确认。撑地启动实验还暴露了一个测量问题：50 Hz 足部接触力矩代理有时会在起跳角动量为正时累计成负值，从而把启动质量清零。因此该指标仍是待解决问题，不能据此宣布已经实现足部主导启动。

## 修改内容

实现保持 Microduck 共享的 61 维观测、14 维动作接口不变。

- `behaviors/headstand.py`：两个动作、奖励、阶段机、训练课程、报告和成功代理。
- `behaviors/core.py`：允许动作声明需要继承的前置模型。
- `behaviors/env.py`：启动阶段变化后刷新任务指令槽。
- `training_init.py`：把 donor 模型迁移到六个任务指令输入，同时保证迁移瞬间的动作输出不变。
- `train_behavior.py`：记录 donor、随机种子和课程环境参数，并执行模型输入初始化。
- `viz_server.py`：从网页训练时寻找已经完成的前置模型。
- `scripts/eval_headspin*.py`：运行确定性、无辅助评估。
- `tests/test_headspin*.py`：约束奖励、课程、观测和有限圈数阶段机。

更详细的设计记录见 [headspin.md](../headspin.md) 和
[headspin-launch.md](../headspin-launch.md)。

## 复现源码环境

本实验基于 Microduck Lab 提交 `54989df` 开发。分支
`headspin-prototype-2026-09-23` 保留了产生本次实验的旧版源码，避免后续上游升级破坏可复现性。

```bash
git clone --branch headspin-prototype-2026-09-23 \
  https://github.com/Eileencaraway/microduck-mac-local.git
cd microduck-mac-local
./scripts/setup.sh
```

按照父项目 README 启动实验室，在 Teach 面板中输入 `headspin` 或
`headspin_launch`。前置训练目录需要包含 `model.zip`、
`vecnormalize.pkl` 和 `policy.onnx`；没有可用 donor 时，网页会返回明确错误。

本地 `runs/` 目录有意被 Git 忽略。文档中的运行名称用于标识作者电脑上的实验记录，不代表仓库已经附带对应模型。

## 测试

在 `microduck_local/` 中运行：

```bash
uv run --with pytest pytest -q \
  tests/test_headspin.py \
  tests/test_headspin_launch.py \
  tests/test_behaviors.py \
  tests/test_lab.py
```

引入反向课程时，相关测试集合共有 212 项通过。这个数字对应本原型分支和当时的依赖环境，不代表运行了上游全部测试。

## 评估训练模型

持续头转：

```bash
.venv/bin/python scripts/eval_headspin.py \
  --policy runs/NAME/policy.onnx \
  --seeds 6 --seed0 100 \
  --out /tmp/headspin-eval.json
```

撑地启动有限圈数头转：

```bash
.venv/bin/python scripts/eval_headspin_launch.py \
  --policy runs/NAME/policy.onnx \
  --seeds 6 --seed0 300 \
  --out /tmp/headspin-launch-eval.json
```

不能根据奖励曲线上升判断成功。宣称动作完成前，必须使用新的站立起点种子运行确定性 ONNX 评估，并观看实际回放。

## 限制与使用范围

- 模型来自本地 CPU MuJoCo 原型栈，域随机化少于官方 GPU 训练栈。
- `headspin_launch` 在既有指令槽中使用仿真生成的角动量、接触、阶段和进度信号；实体机器人需要相应估计器和运行时支持。
- 早期课程中的倒立起点和初始旋转是训练辅助。只有最后一段的静止站立、零初速环境能用于判断自主启动。
- 两圈目标造成过大的探索跨度，因此当前先降为一圈；一圈加收尾仍未稳定完成。
- 不要把当前策略直接部署到实体机器人。应先移植到官方
  [`microduck_rl`](https://github.com/pollen-robotics/microduck_rl)，使用其 sim-to-real 随机化重新训练，再进行安全验证。

## 仓库内容规范

这个分支只包含源码、测试和文档，不上传虚拟环境、访问令牌、浏览器状态、本地训练目录、原始 checkpoint 或自动生成的录屏。选定最终模型和运行方式并完成评估后，可另行在 Hugging Face Hub 发布 ONNX 和 Model Card。

## 许可证与致谢

代码沿用父仓库的 Apache-2.0 许可证。项目基于 Jonathan Hawkins 的
Microduck Lab，以及 Pollen Robotics 开源的 Microduck 和 `microduck_rl`。具体修改文件由 Git 历史记录。

开发和文档整理使用了 OpenAI Codex 辅助。实验判断、结果解释和公开发布由仓库所有者负责。
