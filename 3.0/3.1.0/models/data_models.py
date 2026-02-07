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
        self.survival_compare_results = {}  # 新增：存储生存分析比较结果

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

