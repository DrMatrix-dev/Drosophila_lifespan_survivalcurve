#matplotlib.use('Agg')  # 非交互式后端
import os
import sys
from pathlib import Path
import matplotlib
# 强制使用交互式后端
matplotlib.use('TkAgg')  # 或 'Qt5Agg', 'WXAgg'

# 然后再导入 pyplot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from services import SurvivalAnalysisService
from IO import FileIO

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