import pandas as pd
import numpy as np


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