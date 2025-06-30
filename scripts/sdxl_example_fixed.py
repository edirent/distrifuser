import matplotlib.pyplot as plt
import numpy as np

# 用户提供的数据
data = {
    "GPU 4": {"Computation": 16621.397685, "Communication": 2462.493629},
    "GPU 2": {"Computation": 12901.426359, "Communication": 0},
    "GPU 1": {"Computation": 14076.947861, "Communication": 0},
}

fig, ax = plt.subplots(figsize=(12, 6))

# Y轴的类别和位置
gpus = ["GPU 1", "GPU 2", "GPU 4"]
y_pos = np.arange(len(gpus)) * 15  # 增大GPU之间的间距

colors = {"Computation": "skyblue", "Communication": "salmon"}

# 为每个GPU绘制条形
for i, gpu in enumerate(gpus):
    # 假设计算从t=0开始
    # 通信紧随计算之后
    computation_duration = data[gpu].get("Computation", 0)
    communication_duration = data[gpu].get("Communication", 0)

    # 绘制计算条
    if computation_duration > 0:
        ax.barh(y=y_pos[i], width=computation_duration, left=0, height=8,
                align='center', color=colors["Computation"], edgecolor='gray')

    # 绘制通信条
    if communication_duration > 0:
        ax.barh(y=y_pos[i], width=communication_duration, left=computation_duration, height=8,
                align='center', color=colors["Communication"], edgecolor='gray')


# 设置坐标轴和标题
ax.set_yticks(y_pos)
ax.set_yticklabels(gpus)
ax.invert_yaxis()  # 让标签从上到下阅读
ax.set_xlabel('Timeline (ms)')
ax.set_title('Gantt Chart of GPU Computation and Communication')
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# 创建图例
patches = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
legend_labels = list(colors.keys())
ax.legend(patches, legend_labels, loc='lower right')

plt.tight_layout()

plt.savefig("gpu_gantt_chart.png")
plt.show()
