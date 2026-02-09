import re
#import pandas as pd

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