import pandas as pd
#from .gender_detector import GenderDetector
#from IO.file_IO import FileIO
# 使用相对导入（现在在同一个包内）
from .gender_detector import GenderDetector
# 或者使用绝对导入
from IO import FileIO

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
