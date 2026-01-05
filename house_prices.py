# 示例：使用 Pandas 和 Matplotlib 进行基础数据分析
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------- 设置中文字体 start --------------------------
plt.rcParams['font.sans-serif'] = [
    # Windows 优先
    'SimHei', 'Microsoft YaHei',
    # macOS 优先
    'PingFang SC', 'Heiti TC',
    # Linux 优先
    'WenQuanYi Micro Hei', 'DejaVu Sans'
]
# 修复负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 设置中文字体 end --------------------------

# 1. 读取数据
data = pd.read_csv('house_prices.csv')
print("数据前5行：")
print(data.head())

# 2. 查看数据基本信息
print("\n数据信息：")
print(data.info())

# 3. 绘制房屋面积与价格的散点图
plt.figure(figsize=(10, 6))
plt.scatter(data['面积'], data['价格'], alpha=0.5)
plt.title('房屋面积 vs 价格')
plt.xlabel('面积 (平方米)')
plt.ylabel('价格 (万元)')
plt.grid(True)
plt.show()