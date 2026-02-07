import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy import stats
import os
import re
import warnings
warnings.filterwarnings("ignore")

# ==================== 数据模型层 ====================
class ExcelFileData:
    """Excel文件数据模型"""
    def __init__(self):
        self.sheet_nums = 0
        self.sheet_names = []
        self.data = []
        self.file_type = ""
        self.gender = "unknown"

class SheetAnalysisData:
    """Sheet分析数据模型"""
    def __init__(self):
        self.df = None
        self.sheet_name = ""
        self.gender = "unknown"
        self.per_group_names = []
        self.big_group_nums = 0
        self.per_group_nums = []

# ==================== 数据访问层 ====================
class FileReader:
    """文件读取器"""
    
    @staticmethod
    def read_file(file_path):
        """读取文件"""
        if not os.path.exists(file_path):
            raise ValueError("File not found")
        
        if file_path.endswith('.xlsx'):
            excel_file = pd.ExcelFile(file_path)
            return ('.xlsx', excel_file)
        elif file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
            return ('.xls', data)
        elif file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            return ('.csv', data)
        else:
            raise ValueError("Unsupported file format")
    
    @staticmethod
    def remove_empty_rows_and_cols(df):
        """去除空行空列"""
        print("去除前：", df.head(10))
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        df = df.reset_index(drop=True)
        print("去除后：", df.head(10))
        return df

# ==================== 业务逻辑层 ====================
class GenderDetector:
    """性别检测器"""
    
    def __init__(self):
        self.pattern_male = re.compile(r'male', re.IGNORECASE)
        self.pattern_female = re.compile(r'female', re.IGNORECASE)
    
    def detect_from_dataframe(self, df):
        """从DataFrame中检测性别"""
        gender = "unknown"
        # 直接匹配
        if df.isin(['male']).any().any():
            #gender = "male"
            has_male = True
        elif df.isin(['female']).any().any(): 
            #gender = "female"
            has_female = True
        
        # 正则表达式匹配符号
        if df.applymap(lambda x: bool(re.search(r'♂', str(x)))).any().any():
            has_male = True
        elif df.applymap(lambda x: bool(re.search(r'♀', str(x)))).any().any():
            #gender = "female"
            has_female = True
        
        # 模糊匹配文本
        has_male = df.applymap(lambda x: bool(re.search(self.pattern_male, str(x)))).any().any()
        has_female = df.applymap(lambda x: bool(re.search(self.pattern_female, str(x)))).any().any()
        
        if has_male and not has_female:
            gender = "male"
        elif has_female and not has_male:
            gender = "female"
        
        return gender

class ExcelDataPreprocessor:
    """Excel数据预处理器"""
    
    def __init__(self, gender_detector=None):
        self.gender_detector = gender_detector or GenderDetector()
    
    def preprocess_excel_file(self, excel_file):
        """预处理Excel文件"""
        '''
        sheet_names = excel_file.sheet_names
        data = []
        
        for sheet_name in sheet_names:
            df = excel_file.parse(sheet_name=sheet_name, header=0)
            df = FileReader.remove_empty_rows_and_cols(df)
            data.append(df)
        
        return len(sheet_names), sheet_names, data
        '''
        sheet_names = excel_file.sheet_names
        data = []
        valid_sheets = []  # 存储有效的sheet名称
        
        for sheet_name in sheet_names:
            df = excel_file.parse(sheet_name=sheet_name, header=0)
            
            # 检测是否为空sheet
            if self.is_empty_sheet(df):
                print(f"跳过空sheet: '{sheet_name}'")
                continue
            
            # 清理数据
            df = FileReader.remove_empty_rows_and_cols(df)
            
            # 再次检查清理后是否为空
            if df.empty:
                print(f"清理后仍为空，跳过sheet: '{sheet_name}'")
                continue
                
            data.append(df)
            valid_sheets.append(sheet_name)
        
        return len(valid_sheets), valid_sheets, data
    
    def process_sheet_data(self, df, sheet_name):
        """处理单个sheet的数据"""
        # 检测性别
        gender = self.gender_detector.detect_from_dataframe(df)
        
        # 数据清洗
        df = df.drop(df.columns[0], axis=1)
        df = df.reset_index(drop=True)
        df = df.fillna(0)
        
        # 提取分组信息
        group_names = df.iloc[0, :]
        df.iat[0, 0] = 0
        
        # 计算分组
        per_group_nums, per_group_names = self._calculate_group_info(group_names)
        big_group_nums = len(per_group_nums)
        
        # 进一步处理数据
        df = df.iloc[1:, :]
        df.iat[0, 0] = "Days"
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        
        return df, gender, per_group_names, big_group_nums, per_group_nums
    
    def _calculate_group_info(self, group_names):
        """计算分组信息"""
        per_group_nums = []
        per_group_names = []
        k = 1
        print("原始分组名：",group_names.tolist())
        for j in range(1, len(group_names)):
            if group_names[j] != 0:
                per_group_names.append(group_names[j])
                per_group_nums.append(k)
                k = 1
            else:
                k += 1
            if j == len(group_names) - 1:
                per_group_nums.append(k)
        
        per_group_nums = per_group_nums[1:]
        print("分组名：",per_group_names)
        return per_group_nums, per_group_names

    def is_empty_sheet(self, df):
        '''检查sheet是否为空'''
        # 基础检查：是否为空DataFrame
        if df.empty:
            return True
        
        # 检查是否全为NaN或空值
        if df.isna().all().all():
            return True
        
        # 检查是否只有表头但没有数据行
        if len(df) == 0:
            return True
        
        return False

class SurvivalCalculator:
    """生存率计算器"""
    
    def calculate_survival_data(self, df, per_group_names, big_group_nums, per_group_nums):
        """计算生存数据"""
        df = df.reset_index(drop=True)
        print("作图准备：\n", df.head(10))
        
        # 类型转换
        try:
            df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
        except:
            print("Data conversion error: Non-numeric values found.")
            return None, None

        #df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
        
        # 计算生存率
        sumt = -df.iloc[:, 1:].sum(axis=0)
        sum1 = sumt.copy()
        df_sur = df.copy()
        
        for i in range(len(df)):
            df.iloc[i, 1:] = sum1 + df.iloc[i, 1:]
            df_sur.iloc[i, 1:] = df.iloc[i, 1:] / sumt
            sum1 = df.iloc[i, 1:]
        
        # 分组计算
        group_data, pltdata, pltdata_errorbars = self._calculate_group_statistics(
            df_sur, per_group_names, big_group_nums, per_group_nums
        )
        
        # 构建绘图数据
        for i in range(big_group_nums):
            pltdata[per_group_names[i]] = group_data[i]['means']
            pltdata_errorbars[per_group_names[i]] = group_data[i]['SE']
        
        # 清理数据
        pltdata = self._clean_plot_data(pltdata)
        
        return pltdata, pltdata_errorbars
    
    def _calculate_group_statistics(self, df_sur, per_group_names, big_group_nums, per_group_nums):
        """计算分组统计"""
        group_data = []
        pltdata = pd.DataFrame({'Days': df_sur['Days']})
        pltdata_errorbars = pd.DataFrame({'Days': df_sur['Days']})
        
        for j in range(big_group_nums):
            i = sum(per_group_nums[:j]) + 1
            group_df = df_sur.iloc[:, [0] + list(range(i, i + per_group_nums[j]))]
            group_df['means'] = group_df.iloc[:, 1:].mean(axis=1)
            group_df["SE"] = group_df.iloc[:, 1:-1].std(axis=1) / np.sqrt(per_group_nums[j])
            group_data.append(group_df)
        
        return group_data, pltdata, pltdata_errorbars
    
    def _clean_plot_data(self, pltdata):
        """清理绘图数据"""
        pltdata['testzero'] = pltdata.iloc[:, 1:].mean(axis=1)
        pltdata = pltdata[pltdata['testzero'] != 0]
        pltdata = pltdata.drop(columns=['testzero'])
        
        # 添加最后一行零值
        zerorow = [pltdata.iloc[-1, 0] + 2] + [0] * (pltdata.shape[1] - 1)
        zerorow = pd.DataFrame([zerorow], columns=pltdata.columns)
        pltdata = pd.concat([pltdata, zerorow], ignore_index=True)
        
        return pltdata

# ==================== 展示层 ====================
class PlotDataPreparer:
    """绘图数据准备器"""
    
    @staticmethod
    def prepare_plot_data(pltdata, pltdata_errorbars):
        """准备绘图数据"""
        pltdata_melted = pltdata.melt(id_vars=['Days'], var_name='Group', value_name='Survival Rate')
        pltdata_errorbars_melted = pltdata_errorbars.melt(id_vars=['Days'], var_name='Group', value_name='se')
        
        pltdata_melted = pltdata_melted.merge(
            pltdata_errorbars_melted, on=['Days', 'Group'], how='left'
        )
        
        return pltdata_melted

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

class SurvivalCurvePlotter:
    """生存曲线绘图器"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.style_configurator = PlotStyleConfigurator()
        self.data_preparer = PlotDataPreparer()
        self.survival_calculator = SurvivalCalculator()
    
    def plot_survival_curve(self, df, gender="unknown", sheet_name="", 
                           per_group_names=[], big_group_nums=0, per_group_nums=0):
        """绘制生存曲线"""
        # 计算生存数据
        pltdata, pltdata_errorbars = self.survival_calculator.calculate_survival_data(
            df, per_group_names, big_group_nums, per_group_nums
        )
        
        # 调整误差条数据
        pltdata_errorbars = pltdata_errorbars.iloc[:len(pltdata['Days']), :]
        
        # 准备绘图数据
        pltdata_melted = self.data_preparer.prepare_plot_data(pltdata, pltdata_errorbars)
        
        # 创建图形
        fig = plt.figure(figsize=(6, 6))
        fig.canvas.manager.set_window_title(self.file_path)
        
        ax = plt.gca()
        self.style_configurator.configure_plot_style(ax)
        
        # 绘制误差线
        self._plot_errorbars(pltdata_melted)
        
        # 绘制生存曲线
        sns.lineplot(data=pltdata_melted, x='Days', y='Survival Rate', 
                     hue='Group', marker='o')
        
        # 配置图表
        self._configure_chart(ax, sheet_name, gender, pltdata)
        
        # 添加注释
        self._add_file_info_annotation(fig, sheet_name)
        
        # 显示图形
        plt.show()
        plt.close()
    
    def _plot_errorbars(self, pltdata_melted):
        """绘制误差线"""
        for group in pltdata_melted['Group'].unique():
            group_data = pltdata_melted[pltdata_melted['Group'] == group]
            plt.errorbar(group_data['Days'], group_data['Survival Rate'], 
                        yerr=group_data['se'], fmt='none', capsize=3, 
                        ecolor='black', elinewidth=1)
    
    def _configure_chart(self, ax, sheet_name, gender, pltdata):
        """配置图表属性"""
        plt.title(f'Survival Curve - {sheet_name} ({gender})')
        plt.xlabel('Days')
        plt.ylabel('Survival Rate')
        plt.ylim(0, 1.05)
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

# ==================== 服务层 ====================
class SurvivalAnalysisService:
    """生存分析服务"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_reader = FileReader()
        self.data_preprocessor = ExcelDataPreprocessor()
        self.plotter = SurvivalCurvePlotter(file_path)
    
    def analyze_excel_file(self):
        """分析Excel文件"""
        # 读取文件
        file_type, data = self.file_reader.read_file(self.file_path)
        
        if file_type == '.xlsx':
            # 预处理Excel文件
            sheet_nums, sheet_names, data_list = self.data_preprocessor.preprocess_excel_file(data)
            
            # 处理每个sheet
            for i in range(sheet_nums):
                df = data_list[i]
                
                # 处理sheet数据
                processed_data = self.data_preprocessor.process_sheet_data(df, sheet_names[i])
                df_processed, gender, per_group_names, big_group_nums, per_group_nums = processed_data
                
                # 绘制生存曲线
                self.plotter.plot_survival_curve(
                    df_processed, gender, sheet_names[i],
                    per_group_names, big_group_nums, per_group_nums
                )
        
        elif file_type == '.xls':
            # 处理.xls文件（保持原有逻辑）
            pass
        elif file_type == '.csv':
            # 处理.csv文件（保持原有逻辑）
            pass

# ==================== 应用层 ====================
class SurvivalAnalysisApp:
    """生存分析应用程序"""
    
    def __init__(self):
        self.file_path = ""
        self.service = None
    
    def set_file_path(self, file_path):
        """设置文件路径"""
        self.file_path = file_path
    
    def run(self):
        """运行应用程序"""
        print(f"使用的路径: {self.file_path}")
        
        # 创建分析服务
        self.service = SurvivalAnalysisService(self.file_path)
        
        try:
            # 执行分析
            self.service.analyze_excel_file()
        except Exception as e:
            print(f"分析过程中发生错误: {str(e)}")
            raise
    
    def run_with_default_file(self):
        """使用默认文件运行"""
        self.file_path = "E:\\科研项目\\lifespan\\test.xlsx"
        self.run()

# ==================== 主程序 ====================
def main():
    """主函数"""
    app = SurvivalAnalysisApp()
    
    #mode=input("请选择模式 (1-样例文件, 2-交互输入): ")
    # 使用默认文件
    #app.run_with_default_file()
    
    # 或者使用交互式输入
    user_input_path = input("请输入路径: ")
    file_path = os.path.abspath(user_input_path)
    app.set_file_path(file_path)
    app.run()

if __name__ == "__main__":
    main()