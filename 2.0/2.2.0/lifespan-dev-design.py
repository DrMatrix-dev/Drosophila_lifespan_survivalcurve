import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        self.survival_data = {}
        self.group_structure = {}  # 新增：存储分组结构信息

class SheetAnalysisData:
    """Sheet分析数据模型"""
    def __init__(self):
        self.df = None
        self.sheet_name = ""
        self.gender = "unknown"
        self.per_group_names = []
        self.big_group_nums = 0
        self.per_group_nums = []
        self.survival_df = None
        self.error_df = None
        self.raw_survival_df = None  # 新增：原始生存率数据（包含所有小组）

# ==================== 数据访问层 ====================
class FileIO:
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
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        df = df.reset_index(drop=True)
        return df
    
    @staticmethod
    def save_plot(fig, output_path, dpi=300, bbox_inches='tight'):
        """保存绘图"""
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

    @staticmethod
    def save_all_survival_data(excel_file_data, output_path="survival_data_output.xlsx"):
        """一次性保存所有sheet的生存率数据到Excel文件"""
        if not excel_file_data.survival_data:
            print("没有生存率数据可保存")
            return
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, sheet_data in excel_file_data.survival_data.items():
                    survival_df = sheet_data.get('survival_df', pd.DataFrame())
                    error_df = sheet_data.get('error_df', pd.DataFrame())
                    raw_survival_df = sheet_data.get('raw_survival_df', pd.DataFrame())
                    per_group_names = sheet_data.get('per_group_names', [])
                    per_group_nums = sheet_data.get('per_group_nums', [])
                    
                    if not survival_df.empty and not error_df.empty and not raw_survival_df.empty:
                        # 创建新的DataFrame
                        df_to_save = pd.DataFrame()
                        df_to_save['Days'] = raw_survival_df['Days']
                        
                        # 记录当前列索引
                        current_col_index = 1
                        
                        # 遍历每个大组
                        start_idx = 1  # 从第1列开始（跳过Days列）
                        for group_idx in range(len(per_group_names)):
                            group_name = per_group_names[group_idx]
                            group_size = per_group_nums[group_idx]
                            
                            # 添加该大组内各个小组的原始数据
                            for subgroup_idx in range(group_size):
                                subgroup_col_idx = start_idx + subgroup_idx
                                if subgroup_col_idx < len(raw_survival_df.columns):
                                    subgroup_col_name = f"{group_name} {subgroup_idx+1}"
                                    df_to_save[subgroup_col_name] = raw_survival_df.iloc[:, subgroup_col_idx]
                            
                            # 移动起始索引
                            start_idx += group_size
                            
                            # 添加该大组的mean值
                            mean_col_name = f"{group_name}_mean"
                            if group_name in survival_df.columns:
                                df_to_save[mean_col_name] = survival_df[group_name]
                            
                            # 添加该大组的SE值
                            se_col_name = f"{group_name}_SE"
                            if group_name in error_df.columns:
                                df_to_save[se_col_name] = error_df[group_name]
                        
                        # 保存到Excel
                        df_to_save.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                        
                        # 打印列结构信息
                        print(f"  - Sheet '{sheet_name}' 列结构:")
                        cols = list(df_to_save.columns)
                        print(f"    列数: {len(cols)}")
                        print(f"    列顺序: {', '.join(cols[:min(10, len(cols))])}..." if len(cols) > 10 else f"    列顺序: {', '.join(cols)}")
                
                print(f"\n✓ 所有生存率数据已保存到: {output_path}")
                print(f"✓ 共保存了 {len(excel_file_data.survival_data)} 个sheet的数据")
                
        except Exception as e:
            print(f"保存文件时出错: {e}")
            # 尝试创建新文件
            try:
                with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
                    for sheet_name, sheet_data in excel_file_data.survival_data.items():
                        survival_df = sheet_data.get('survival_df', pd.DataFrame())
                        error_df = sheet_data.get('error_df', pd.DataFrame())
                        raw_survival_df = sheet_data.get('raw_survival_df', pd.DataFrame())
                        per_group_names = sheet_data.get('per_group_names', [])
                        per_group_nums = sheet_data.get('per_group_nums', [])
                        
                        if not survival_df.empty and not error_df.empty and not raw_survival_df.empty:
                            df_to_save = pd.DataFrame()
                            df_to_save['Days'] = raw_survival_df['Days']
                            
                            start_idx = 1
                            for group_idx in range(len(per_group_names)):
                                group_name = per_group_names[group_idx]
                                group_size = per_group_nums[group_idx]
                                
                                for subgroup_idx in range(group_size):
                                    subgroup_col_idx = start_idx + subgroup_idx
                                    if subgroup_col_idx < len(raw_survival_df.columns):
                                        subgroup_col_name = f"{group_name} {subgroup_idx+1}"
                                        df_to_save[subgroup_col_name] = raw_survival_df.iloc[:, subgroup_col_idx]
                                
                                start_idx += group_size
                                
                                mean_col_name = f"{group_name}_mean"
                                if group_name in survival_df.columns:
                                    df_to_save[mean_col_name] = survival_df[group_name]
                                
                                se_col_name = f"{group_name}_SE"
                                if group_name in error_df.columns:
                                    df_to_save[se_col_name] = error_df[group_name]
                            
                            df_to_save.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                print(f"数据已保存到: {output_path}")
            except Exception as e2:
                print(f"创建新文件时也出错: {e2}")

    @staticmethod
    def _clean_filepath(file_path):
        path = file_path.strip()
        
        if len(path) < 2:
            return path
        
        if (path[0] == '"' and path[-1] == '"'):
            return path[1:-1]
        elif (path[0] == "'" and path[-1] == "'"):
            return path[1:-1]
        
        return path

# ==================== 业务逻辑层 ====================
class GenderDetector:
    """性别检测器"""
    
    def __init__(self):
        self.pattern_male = re.compile(r'male', re.IGNORECASE)
        self.pattern_female = re.compile(r'female', re.IGNORECASE)
    
    def detect_from_dataframe(self, df):
        """从DataFrame中检测性别"""
        gender = "unknown"
        has_male = False
        has_female = False

        if df.isin(['male']).any().any():
            has_male = True
        if df.isin(['female']).any().any(): 
            has_female = True

        if df.isin(['♂']).any().any():
            has_male = True
        if df.isin(['♀']).any().any(): 
            has_female = True
        
        if df.applymap(lambda x: bool(re.search(r'♂', str(x)))).any().any():
            has_male = True
        if df.applymap(lambda x: bool(re.search(r'♀', str(x)))).any().any():
            has_female = True
        
        if has_male==True and has_female==False:
            gender = "male"
        if has_female==True and has_male==False:
            gender = "female"
        
        return gender

class ExcelDataPreprocessor:
    """Excel数据预处理器"""
    
    def __init__(self, gender_detector=None):
        self.gender_detector = gender_detector or GenderDetector()
    
    def preprocess_excel_file(self, excel_file):
        """预处理Excel文件"""
        sheet_names = excel_file.sheet_names
        data = []
        valid_sheets = []
        
        for sheet_name in sheet_names:
            df = excel_file.parse(sheet_name=sheet_name, header=None)
            
            if self.is_empty_sheet(df):
                print(f"跳过空sheet: '{sheet_name}'")
                continue
            
            df = FileIO.remove_empty_rows_and_cols(df)
            
            if df.empty:
                print(f"清理后仍为空，跳过sheet: '{sheet_name}'")
                continue
                
            data.append(df)
            valid_sheets.append(sheet_name)
        
        return len(valid_sheets), valid_sheets, data
    
    def process_sheet_data(self, df, sheet_name):
        """处理单个sheet的数据"""
        gender = self.gender_detector.detect_from_dataframe(df)
        
        df = df.drop(df.columns[0], axis=1)
        df = df.reset_index(drop=True)
        df.columns = pd.RangeIndex(start=0, stop=len(df.columns), step=1)
        df = df.fillna(0)
        
        group_names = df.iloc[0, :]
        df.iat[0, 0] = 0
        
        per_group_nums, per_group_names = self._calculate_group_info(group_names)
        big_group_nums = len(per_group_nums)
        
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
        return per_group_nums, per_group_names

    def is_empty_sheet(self, df):
        '''检查sheet是否为空'''
        if df.empty:
            return True
        
        if df.isna().all().all():
            return True
        
        if len(df) == 0:
            return True
        
        return False

class SurvivalCalculator:
    """生存率计算器"""
    
    def calculate_survival_data(self, df, per_group_names, big_group_nums, per_group_nums, sheet_name=""):
        """计算生存数据"""
        df = df.reset_index(drop=True)
        print(f"\n处理Sheet '{sheet_name}' 的作图准备数据：")
        print(df.head(10))
        
        try:
            df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
        except:
            print("Data conversion error: Non-numeric values found.")
            return None, None, None
        
        # 保存原始df用于后续计算各小组数据
        df_original = df.copy()
        
        sumt = -df.iloc[:, 1:].sum(axis=0)
        sum1 = sumt.copy()
        df_sur = df.copy()
        
        for i in range(len(df)):
            df.iloc[i, 1:] = sum1 + df.iloc[i, 1:]
            df_sur.iloc[i, 1:] = df.iloc[i, 1:] / sumt
            sum1 = df.iloc[i, 1:]
        
        # 计算各小组原始生存率数据
        raw_survival_df = self._calculate_raw_survival_data(df_original, per_group_nums)
        
        # 计算分组统计（mean和SE）
        group_data, pltdata, pltdata_errorbars = self._calculate_group_statistics(
            df_sur, per_group_names, big_group_nums, per_group_nums
        )
        
        # 构建mean和SE的DataFrame
        survival_df = pd.DataFrame({'Days': df_sur['Days']})
        error_df = pd.DataFrame({'Days': df_sur['Days']})
        
        for i in range(big_group_nums):
            survival_df[per_group_names[i]] = group_data[i]['means']
            error_df[per_group_names[i]] = group_data[i]['SE']
        
        # 清理数据
        survival_df = self._clean_plot_data(survival_df)
        error_df = error_df.iloc[:len(survival_df['Days']), :]
        raw_survival_df = raw_survival_df.iloc[:len(survival_df['Days']), :]
        
        print(f"\nSheet '{sheet_name}' 数据预览：")
        print("1. 各小组原始生存率数据（前5列）：")
        print(raw_survival_df.iloc[:, :min(6, len(raw_survival_df.columns))].head())
        
        print(f"\n2. 大组mean值数据：")
        print(survival_df.head())
        
        print(f"\n3. 大组SE值数据：")
        print(error_df.head())
        
        return survival_df, error_df, raw_survival_df
    
    def _calculate_raw_survival_data(self, df, per_group_nums):
        """计算各小组原始生存率数据"""
        # 计算每个小组的原始生存率
        sumt = -df.iloc[:, 1:].sum(axis=0)
        sum1 = sumt.copy()
        raw_survival = df.copy()
        
        for i in range(len(df)):
            df.iloc[i, 1:] = sum1 + df.iloc[i, 1:]
            raw_survival.iloc[i, 1:] = df.iloc[i, 1:] / sumt
            sum1 = df.iloc[i, 1:]
        
        # 清理数据
        raw_survival = self._clean_plot_data(raw_survival)
        
        return raw_survival
    
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
        
        self._configure_chart(ax, sheet_name, gender, survival_df)
        
        self._add_file_info_annotation(fig, sheet_name)
        
        plt.show()
        plt.close()
    
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

# ==================== 服务层 ====================
class SurvivalAnalysisService:
    """生存分析服务"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_reader = FileIO()
        self.data_preprocessor = ExcelDataPreprocessor()
        self.survival_calculator = SurvivalCalculator()
        self.plotter = SurvivalCurvePlotter(file_path)
        self.excel_file_data = ExcelFileData()
    
    def analyze_excel_file(self):
        """分析Excel文件"""
        file_type, data = self.file_reader.read_file(self.file_path)
        
        if file_type == '.xlsx':
            self.excel_file_data.sheet_nums, self.excel_file_data.sheet_names, data_list = \
                self.data_preprocessor.preprocess_excel_file(data)
            self.excel_file_data.file_type = file_type
            
            print(f"找到 {self.excel_file_data.sheet_nums} 个有效sheet")
            
            for i in range(self.excel_file_data.sheet_nums):
                sheet_name = self.excel_file_data.sheet_names[i]
                df = data_list[i]
                
                print(f"\n{'='*50}")
                print(f"处理sheet: {sheet_name}")
                print(f"{'='*50}")
                
                processed_data = self.data_preprocessor.process_sheet_data(df, sheet_name)
                df_processed, gender, per_group_names, big_group_nums, per_group_nums = processed_data
                
                # 计算生存数据（现在返回三个值）
                survival_df, error_df, raw_survival_df = self.survival_calculator.calculate_survival_data(
                    df_processed, per_group_names, big_group_nums, per_group_nums, sheet_name
                )
                
                if survival_df is not None and error_df is not None and raw_survival_df is not None:
                    # 存储到统一的数据结构中
                    self.excel_file_data.survival_data[sheet_name] = {
                        'survival_df': survival_df,
                        'error_df': error_df,
                        'raw_survival_df': raw_survival_df,  # 新增：原始各小组数据
                        'gender': gender,
                        'per_group_names': per_group_names,
                        'per_group_nums': per_group_nums,
                        'big_group_nums': big_group_nums
                    }
                    
                    # 绘制生存曲线（只使用mean数据）
                    self.plotter.plot_survival_curve(survival_df, error_df, gender, sheet_name)
            
            # 保存所有数据
            self.save_all_survival_data()
        
        elif file_type == '.xls':
            pass
        elif file_type == '.csv':
            pass
    
    def save_all_survival_data(self, output_path=None):
        """保存所有sheet的生存率数据"""
        if not output_path:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_dir = os.path.dirname(self.file_path)
            output_path = os.path.join(output_dir, f"{base_name}_survival_data.xlsx")
        
        print(f"\n{'='*50}")
        print("保存所有数据到Excel文件...")
        FileIO.save_all_survival_data(self.excel_file_data, output_path)
        
        # 显示详细的分组信息
        print(f"\n{'='*50}")
        print("数据输出格式说明：")
        print("每个sheet的输出列顺序为：")
        print("1. Days列")
        
        for sheet_name, sheet_data in self.excel_file_data.survival_data.items():
            per_group_names = sheet_data.get('per_group_names', [])
            per_group_nums = sheet_data.get('per_group_nums', [])
            
            print(f"\nSheet '{sheet_name}' 的分组结构：")
            for i, (group_name, group_size) in enumerate(zip(per_group_names, per_group_nums)):
                print(f"  大组{i+1}: {group_name} (包含{group_size}个小组)")
                print(f"    → 输出列: ", end="")
                for j in range(group_size):
                    print(f"{group_name}_subgroup{j+1}", end=", ")
                print(f"{group_name}_mean, {group_name}_SE")

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
        
        self.service = SurvivalAnalysisService(self.file_path)
        
        try:
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
    
    user_input_path = input("请输入Excel文件路径: ")
    file_path = FileIO._clean_filepath(user_input_path)
    file_path = os.path.abspath(file_path)
    app.set_file_path(file_path)
    app.run()

if __name__ == "__main__":
    main()