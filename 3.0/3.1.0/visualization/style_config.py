from matplotlib import pyplot as plt
import seaborn as sns

class PlotStyleConfigurator:
    """绘图样式配置器"""
    
    def __init__(self):
        self._configure_matplotlib()
    
    def _configure_matplotlib(self):
        """配置Matplotlib样式"""
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.unicode_minus'] = False
    
    def configure_plot_style(self, ax):
        """配置绘图样式"""
        sns.set_style("white")
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_position(('data', 0))
        ax.grid(False)

        ax.tick_params(
            axis='both',
            which='both',
            bottom=True,    # 显示底部刻度
            left=True,      # 显示左侧刻度
            top=False,      # 不显示顶部刻度
            right=False,    # 不显示右侧刻度
            labelbottom=True,  # 显示底部刻度标签
            labelleft=True     # 显示左侧刻度标签
        )
        
        # 确保刻度线样式
        ax.tick_params(
            axis='both',
            which='major',
            direction='out',
            length=6,
            width=1,
            colors='black'
        )