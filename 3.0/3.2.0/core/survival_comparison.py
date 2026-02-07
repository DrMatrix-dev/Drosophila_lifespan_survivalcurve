import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests

#from .whole_to_individual_transfomer import SurvivalDataTransformer
from IO.file_IO import FileIO

class SurvivalStatistics:
    
    def __init__(self):
        #self.DataTransformer = SurvivalDataTransformer()
        pass

    def perform_logrank_test(self, df,  per_group_names, big_group_nums, per_group_nums, per_group_type, sheet_name=""):
        #per_group_nums未使用，后续考虑删去
        df = df.reset_index(drop=True)
        df_individual = self.data_tranformer(df, per_group_names, big_group_nums, per_group_nums, per_group_type)
        test_results = self.logranktest(df_individual)
        return test_results
        #pass
        # test_results是个列表
        # 这个设计弃用 返回：元组，顺序为 (groupname，exp_type, treat_type, p_value，p_adj, adj_flag), 如果adj_flag为True，那么进行了校正

    def logranktest(self, df_individual):
        exp_types = sorted(df_individual['Exp_type'].unique())
        print(f"发现 {len(exp_types)} 个实验: {exp_types}")

        test_results = []
        for exp in exp_types:
             # 获取该实验的数据
            exp_df = df_individual[df_individual['Exp_type'] == exp]
            
            # 检查是否有对照组
            if 'con' not in exp_df['Treat_type'].values:
                print(f"  警告: 实验 {exp} 没有对照组 (con)，跳过")
                continue

            # 分离对照组
            control_df = exp_df[exp_df['Treat_type'] == 'con']
            control_group_name = control_df['Group'].iloc[0] if not control_df.empty else "unknown"
            control_count = len(control_df)

            # 获取治疗组个数（排除con）
            treatment_types = [t for t in exp_df['Treat_type'].unique() if t != 'con']
            print(f"  实验 {exp} 包含对照组和 {len(treatment_types)} 个治疗组: {treatment_types}")
            
            # 收集所有比较的原始p值
            p_values_raw = []
            comparison_info = []

            for treat in treatment_types:
                treat_df = exp_df[exp_df['Treat_type'] == treat]
                treat_count = len(treat_df)

                # 根据treat类型获取group名称
                treat_group_name = treat_df['Group'].iloc[0] if not treat_df.empty else "unknown"
                
                print(f"  比较: con (n={control_count}) vs {treat} (n={treat_count})")
                
                # 执行logrank检验
                result = logrank_test(
                    control_df['Days'],
                    treat_df['Days'],
                    event_observed_A=control_df['Event'],
                    event_observed_B=treat_df['Event']
                )
                
                p_values_raw.append(result.p_value)
                comparison_info.append({
                    'exp': exp,
                    'control_group_name': control_group_name,
                    'treat_group_name': treat_group_name,
                    'treat': treat,
                    'p_raw': result.p_value,
                    #'statistic': result.test_statistic,
                    'n_treat': treat_count,
                })
                
                print(f"    原始p值: {result.p_value:.6f}")
            
            # ========== 多重比较校正 ==========
            n_comparisons = len(p_values_raw)
            
            if n_comparisons == 1:
                print(f"\n  单次比较，无需多重比较校正")
                correction_method = "None"
                
                for info in comparison_info:
                    test_results.append({
                        'exp_type': exp,
                        'control_group': info['control_group_name'],
                        'treatment_group': info['treat_group_name'],
                        'treat': info['treat'],
                        'p_value': info['p_raw'],
                        'p_adj': info['p_raw'],
                        'correction_method': correction_method,
                        'significant': info['p_raw'] < 0.05,
                        #'statistic': info['statistic'],
                        'n_control': control_count,
                        'n_treatment': info['n_treat']
                    })
            
            else:
                print(f"\n  多重比较校正 (共{n_comparisons}次比较)")
                method = 'bonferroni'  # 可以选择其他方法，如 'fdr_bh'
                reject, p_adj, _, _ = multipletests(
                        p_values_raw, 
                        alpha=0.05, 
                        method=method
                )
                #更新test_results
                for i, info in enumerate(comparison_info):
                    test_results.append({
                        'exp_type': exp,
                        'control_group': info['control_group_name'],
                        'treatment_group': info['treat_group_name'],
                        'treat': info['treat'],
                        'p_value': info['p_raw'],
                        'p_adj': p_adj[i],
                        'correction_method': method,
                        'significant': reject[i],
                        #'statistic': info['statistic'],
                        'n_control': control_count,
                        'n_treatment': info['n_treat']
                    })
                    #print(f"    比较: con vs {info['treat']} - 原始p值: {info['p_raw']:.6f}, 校正后p值: {p_adj[i]:.6f}, {'显著' if reject[i] else '不显著'}")

        print("Logrank test results:", test_results)
        print("\n\n")
        #breakpoint()
        return test_results
                

    def data_tranformer(self, df, per_group_names, big_group_nums, per_group_nums, per_group_type):
        '''将每组的生存数据转换为个体数据'''
        all_group_individual_result = []
        for j in range(big_group_nums):
            print("进度：",j)
            i = sum(per_group_nums[:j]) + 1
            group_df = df.iloc[:, [0] + list(range(i, i + per_group_nums[j]))]
            print("data_tranformer每组信息: \n", group_df)
            #breakpoint()
            group_df["per_day_sum"] = -group_df.iloc[:, 1:].sum(axis=1)
            #print("data_tranformer每天死亡数加和: \n", group_df)
            #breakpoint()

            # 转换为个体数据
            per_group_individual_result = []
            for row in group_df.itertuples(index=True, name='Pandas'):
                #print("每行数据: \n")
                #print(f"Index: {row.Index}, Days: {row.Days}, per_day_sum: {row.per_day_sum}")
                for _ in range(int(row.per_day_sum)):
                    per_group_individual_result.append({
                        "Days": row.Days,
                        "Event": 1,
                        "Group": per_group_names[j],
                        "Exp_type": per_group_type[j].split('_')[0],
                        "Treat_type": per_group_type[j].split('_')[1]
                    })
            all_group_individual_result.extend(per_group_individual_result)

            #print("per_group_individual_result: \n", per_group_individual_result)
            #breakpoint()
        
        all_group_individual_result = pd.DataFrame(all_group_individual_result)
        print("转换完成的个体数据: \n", all_group_individual_result)

        #保存个体数据(开发人员专用)
        #FileIO.save_individual_data(all_group_individual_result)
        #breakpoint()

        return all_group_individual_result
        #breakpoint()
        #print("转换完成的个体数据: \n", all_group_individual_result)
