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
        ax.grid(False)