from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

#from .style_config import PlotStyleConfigurator
from .style_config import PlotStyleConfigurator

# ==================== 展示层 ====================
class PlotDataPreparer:
    """绘图数据准备器"""
    
    @staticmethod
    def prepare_plot_data(pltdata, pltdata_errorbars):
        """准备绘图数据"""
        pltdata = PlotDataPreparer.vectorized_trailing_zero_processing(pltdata)
        pltdata_melted = pltdata.melt(id_vars=['Days'], var_name='Group', value_name='Survival Rate')
        pltdata_errorbars_melted = pltdata_errorbars.melt(id_vars=['Days'], var_name='Group', value_name='se')
        
        pltdata_melted = pltdata_melted.merge(
            pltdata_errorbars_melted, on=['Days', 'Group'], how='left'
        )
        
        return pltdata_melted
    
    @staticmethod
    def vectorized_trailing_zero_processing(df):
        """向量化处理末尾零值"""
        df_processed = df.copy()
    
        for col in df.columns:
            col_data = df[col]
            non_zero_mask = (col_data != 0) & (col_data.notna())
            
            if non_zero_mask.any():
                last_non_zero_idx = non_zero_mask[::-1].idxmax()
                trailing_indices = col_data.index[col_data.index > last_non_zero_idx]
                
                if len(trailing_indices) > 1:
                    df_processed.loc[trailing_indices[1:], col] = np.nan
        
        return df_processed

class SurvivalCurvePlotter:
    """生存曲线绘图器"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.style_configurator = PlotStyleConfigurator()
        self.data_preparer = PlotDataPreparer()
    
    def plot_survival_curve(self, survival_df, error_df, gender="unknown", sheet_name=""):
        """绘制生存曲线"""
        pltdata_melted = self.data_preparer.prepare_plot_data(survival_df, error_df)
        
        fig = plt.figure(figsize=(6, 6))
        fig.canvas.manager.set_window_title(self.file_path)
        
        ax = plt.gca()
        self.style_configurator.configure_plot_style(ax)
        
        self._plot_errorbars(pltdata_melted)
        
        sns.lineplot(data=pltdata_melted, x='Days', y='Survival Rate', 
                     hue='Group', marker='o')
        
        self._force_ticks_visible(ax)

        self._configure_chart(ax, sheet_name, gender, survival_df)
        
        self._add_file_info_annotation(fig, sheet_name)
        
        plt.show()
        plt.close()
        
    def _force_ticks_visible(self, ax):
        """强制显示刻度线"""
        # 移除 seaborn 可能设置的任何隐藏
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
    def _plot_errorbars(self, pltdata_melted):
        """绘制误差线"""
        for group in pltdata_melted['Group'].unique():
            group_data = pltdata_melted[pltdata_melted['Group'] == group]
            error=group_data['se']
            plt.errorbar(group_data['Days'], group_data['Survival Rate'], 
                        yerr=[np.zeros_like(error), error], fmt='none', capsize=3, 
                        ecolor='black', elinewidth=1)
    
    def _configure_chart(self, ax, sheet_name, gender, pltdata):
        """配置图表属性"""
        plt.title(f'Survival Curve - {sheet_name}')
        plt.xlabel('Days')
        plt.ylabel('Survival Rate')
        plt.ylim(0, 1.01)
        plt.xlim(0, pltdata['Days'].max() + 1)
        plt.legend(loc='lower left', frameon=False)
    
    def _add_file_info_annotation(self, fig, sheet_name):
        """添加文件信息注释"""
        fig.text(0.02, 0.98,  
                f"{self.file_path}\nSheet: {sheet_name}\n",
                fontsize=5,
                verticalalignment='top',
                horizontalalignment='left',
                transform=fig.transFigure,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0),
                fontfamily='SimHei'
                )