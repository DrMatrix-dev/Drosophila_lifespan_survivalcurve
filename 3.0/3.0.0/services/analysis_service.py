import os

#from models.data_models import ExcelFileData, SheetAnalysisData
#from IO.file_IO import FileIO
#from core.data_processor import ExcelDataPreprocessor
#from core.survival_calculator import SurvivalCalculator
#from visualization.plotter import SurvivalCurvePlotter
from models import ExcelFileData
from IO import FileIO
from core import (
    ExcelDataPreprocessor, 
    SurvivalCalculator
)
from visualization import SurvivalCurvePlotter

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