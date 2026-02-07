import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re
import warnings
warnings.filterwarnings("ignore")

file_path = ""
# 智能读取数据，返回文件类型和文件。
# 如果是excel文件，返回excel文件对象；如果是csv，返回DataFrame对象。
def import_file(file_path):
    #判断文件是否存在
    if not os.path.exists(file_path):
        raise ValueError("File not found")

    # 根据文件扩展名选择合适的读取方法
    if file_path.endswith('.xlsx'):
        #data = pd.read_excel(file_path, sheet_name=0, header=0)
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


#预处理excel文件，将各个表格拆分到data列表中
def pretreat_excelfile(excel_file):
    sheet_names = excel_file.sheet_names
    #print("nums:", len(sheet_names))
    #print("Available sheets:", sheet_names)
    data=[]
    for i in range(len(sheet_names)):
        data.append(remove_empty_rows_and_cols(excel_file.parse(sheet_name=sheet_names[i], header=0)))
        #print(data[i].head(10))
    return (len(sheet_names), sheet_names, data)

#去除全空行列
def remove_empty_rows_and_cols(df):
    print("去除前：",df.head(10))
    # 删除全空的行
    df = df.dropna(how='all')
    # 删除全空的列
    df = df.dropna(axis=1, how='all')
    
    # 重建索引
    df = df.reset_index(drop=True)
    print("去除后：",df.head(10))
    return df

# 搜索性别
# 在文件名中查找
# 在表格中使用直接搜索male或female，或者使用正则表达式搜索♂和♀
def search_gender(df):
    gender="unknown"
    pattern_male = re.compile(r'male', re.IGNORECASE)
    pattern_female = re.compile(r'female', re.IGNORECASE)
    #直接匹配
    if(df.isin(['male']).any().any()):
        gender="male"
    elif(df.isin(['female']).any().any()): 
        gender="female"

    #正则表达式匹配
    if df.applymap(lambda x: bool(re.search(r'♂', str(x)))).any().any():
        gender="male"
    elif df.applymap(lambda x: bool(re.search(r'♀', str(x)))).any().any():
        gender="female"

    has_male=df.applymap(lambda x: bool(re.search(pattern_male, str(x)))).any().any()
    print(has_male)
    has_female=df.applymap(lambda x: bool(re.search(pattern_female, str(x)))).any().any()
    print(has_female)
    if has_male and not has_female:
        gender="male"
    elif has_female:
        gender="female"

    #文件名中查找
    #if ("male" in file_path.lower() and "female" not in file_path.lower()) or ("♂" in file_path and "♀" not in file_path):
    #    gender="male"
    #elif ("female" in file_path.lower() and "male" not in file_path.lower()) or ("♀" in file_path and "♂" not in file_path):
    #    gender="female"

    return gender

#处理预处理后的excel文件，作图
def treat_exceldata(excelfile_pretreated):
    sheet_nums, sheet_names, data = excelfile_pretreated
    for i in range(sheet_nums):
        print(f"Processing sheet: {sheet_names[i]}")
        df = data[i]
        # 对每个sheet数据的处理逻辑
        #用于测试 
        print(df.head(10))
        gender=search_gender(df)
        df = df.drop(df.columns[0], axis=1)  # 删除第一列
        df = df.reset_index(drop=True)
        df = df.fillna(0)  # 缺失值替换为0
        group_names = df.iloc[0,:]

        df.iat[0,0] = 0
        print(group_names)
        #rprint(df.head(10))
        #breakpoint()
        # 智能分组

        # 每个小组内的重复数
        per_group_nums=[]
        per_group_names=[]
        k=1
        #group_names_temp=[]
        #group_nums_temp=[]
        for j in range(1,len(group_names)):
            if group_names[j] != 0:
                per_group_names.append(group_names[j])
                per_group_nums.append(k)
                #per_group_nums.append(k)
                #per_group_names.append(j)
                k=1 
            else:
                k+=1
            if j == len(group_names)-1:
                per_group_nums.append(k)

        per_group_nums=per_group_nums[1:]
        print(per_group_names)
        print(per_group_nums)
        #breakpoint()
        #print(per_group_names)

        # 大组数，表示有多少个对照组+实验组
        big_group_nums=len(per_group_nums)
        #print(big_group_nums)
        
        #df = df.drop(index=0) #删除第一行
        df=df.iloc[1: , :]
        #print(df.head(10))
        #breakpoint()
        #print(df.iloc[0,0])
        #breakpoint()
        df.iat[0,0] = "Days"
        #print(df.head(10))
        #breakpoint()
        #设置列名
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        #print(df.head(10))
        #breakpoint()
        #print(df.iloc[0,:])
        #print(per_group_nums)
        #print(df.head(10))
        #print("process done.")
        print(per_group_names)
        #breakpoint()
        flag = plot_survival_curve(df, gender, sheet_names[i], per_group_names, big_group_nums, per_group_nums)
        if flag == -1:
            print(f"Skipping sheet: {sheet_names[i]} due to data conversion error.")
            continue
        # 测试用，只绘制一幅图
        #break


# 作图
def plot_survival_curve(df, gender="unknown", sheet_name="", per_group_names=[], big_group_nums=0, per_group_nums=0):
    #print("Plotting survival curve...")
    df = df.reset_index(drop=True)
    print("作图准备：\n",df.head(10))

    # 注意：格子内一定不能有非数值型数据，否则会报错。这里进行了类型转换，但对字符串型数据无法处理
    # 使用前一定要先检查空格子内是否有多余的空格或回车
    try:
        df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
    except ValueError as e:
        print("Error converting data to float. Please check for non-numeric values in the data.")
        print(e)
        return -1

    #breakpoint()
    sumt = -df.iloc[:, 1:].sum(axis=0)
    print("sumt:", sumt)
    sum1 = sumt.copy()
    df_sur = df.copy()
    for i in range(len(df)):
        # 更新当前行的数值列
        df.iloc[i, 1:] = sum1 + df.iloc[i, 1:]
        # 计算生存率
        df_sur.iloc[i, 1:] = df.iloc[i, 1:] / sumt
        # 更新累积值
        sum1 = df.iloc[i, 1:]
    #print(df_sur)
    #breakpoint()
    group_data=[]
    pltdata=pd.DataFrame({'Days': df['Days']})
    pltdata_errorbars=pd.DataFrame({'Days': df['Days']})
    #print(pltdata)
    #breakpoint()
    #----------------------
    '''
    i=1
    for j in range(big_group_nums):
        group_df = df_sur.iloc[:, [0] + list(range(i, i+per_group_nums[j]))]
        group_df['means'] = group_df.iloc[:, 1:].mean(axis=1)
        group_df["SE"] = group_df.iloc[:, 1:-1].std(axis=1) / np.sqrt(per_group_nums[j])
        #pltdata[group_names[i-1]] = group_df['means']
        #pltdata[group_names[]]
        group_data.append(group_df)
        i += per_group_nums[j]
        #print(group_df.head(10))
        #break
    print(pltdata)
    '''
    for j in range(big_group_nums):
        i = sum(per_group_nums[:j]) + 1
        group_df = df_sur.iloc[:, [0] + list(range(i, i+per_group_nums[j]))]
        group_df['means'] = group_df.iloc[:, 1:].mean(axis=1)
        group_df["SE"] = group_df.iloc[:, 1:-1].std(axis=1) / np.sqrt(per_group_nums[j])
        group_data.append(group_df)
        #print(group_df.head(10))
    #breakpoint()
    #----------------------

    #breakpoint()
    print(per_group_names)
    #breakpoint()
    for i in range(big_group_nums):
        #print(per_group_names[i])
        pltdata[per_group_names[i]] = group_data[i]['means']
        pltdata_errorbars[per_group_names[i]] = group_data[i]['SE']
    #print(pltdata)
    #breakpoint()
    pltdata['testzero'] = pltdata.iloc[:, 1:].mean(axis=1)

    #去除pltdata零值行
    pltdata = pltdata[pltdata['testzero'] != 0]
    pltdata = pltdata.drop(columns=['testzero'])
    #print(pltdata)
    #breakpoint()
    #print(pltdata_errorbars.head(10))

    #添加最后一行，Days值为最后一行Days值+2，其余值为0
    zerorow=[pltdata.iloc[-1,0]+2,]
    for i in range(1, pltdata.shape[1]):
        zerorow.append(0)
    #将zerorow添加到pltdata末尾
    zerorow = pd.DataFrame([zerorow], columns=pltdata.columns)
    pltdata = pd.concat([pltdata, zerorow], ignore_index=True)
    #print(pltdata)
    #breakpoint()
    #保持pltdata_errorbars行数与pltdata一致
    pltdata_errorbars = pltdata_errorbars.iloc[:len(pltdata['Days']),:]
    #print(pltdata_errorbars)
    #print(pltdata.shape[0])


    pltdata_melted = pltdata.melt(id_vars=['Days'], var_name='Group', value_name='Survival Rate')
    #print(pltdata_melted)
    pltdata_errorbars_melted = pltdata_errorbars.melt(id_vars=['Days'], var_name='Group', value_name='se')
    #print(pltdata_errorbars_melted)

    # 合并pltdata_melted和pltdata_errorbars_melted，便于绘图
    pltdata_melted = pltdata_melted.merge(pltdata_errorbars_melted, on=['Days', 'Group'], how='left')
    #print(pltdata_melted)

    #plt.figure(figsize=(6, 6))
    fig = plt.figure(figsize=(6, 6))
    fig.canvas.manager.set_window_title(file_path) 
    sns.set_style("white")
    #plt.rcParams['font.sans-serif'] = ['SimHei'] # 用黑体显示中文
    #plt.rcParams['axes.unicode_minus'] = False # 正常显示负号


    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.unicode_minus'] = False

    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.grid(False)

    # 添加误差线
    for group in pltdata_melted['Group'].unique():  
        group_data = pltdata_melted[pltdata_melted['Group'] == group]
        plt.errorbar(group_data['Days'], group_data['Survival Rate'], yerr=group_data['se'], fmt='none', capsize=3, ecolor='black', elinewidth=1)


    sns.lineplot(data=pltdata_melted, x='Days', y='Survival Rate', hue='Group', marker='o')

    plt.title(f'Survival Curve - {sheet_name} ({gender})')
    plt.xlabel('Days')
    plt.ylabel('Survival Rate')
    plt.ylim(0, 1.05)
    plt.xlim(0, pltdata['Days'].max()+1)
    plt.legend(loc='lower left', frameon=False)
    #plt.tight_layout()

    # 在图形窗口的左上角添加注释
    #plt.rcParams['font.sans-serif'] = ['SimHei'] # 用黑体显示中文
    #plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
    fig.text(0.02, 0.98,  
            file_path+f"\nSheet: {sheet_name}\n",
            fontsize=5,
            verticalalignment='top',
            horizontalalignment='left',
            transform=fig.transFigure,  # 关键：使用图形坐标系
            bbox=dict(boxstyle='round', facecolor='white', alpha=0),
            fontfamily='SimHei'
            )
        
    #plt.savefig(f"e:\\{sheet_name}_survival_curve.png", dpi=300)
    plt.show()
    plt.close()

    #print(pltdata)
    #pass

#主函数
def main():
    global file_path
    #file_path = "E:\\科研项目\\lifespan\\1.24\\Hml-Rel (1).xlsx"
    #file_path = "E:\\科研项目\\lifespan\\1.24\\Hml-yv and Relish-i ♂ mSD 29℃ 1st-3rd.xlsx"
    file_path= "E:\\科研项目\\lifespan\\全自动寿命分析软件\\测试数据\\Dh44-TrpA1 - 副本.xlsx"
    #user_input_path = input("请输入路径: ")
    # 转换为绝对路径
    #file_path = os.path.abspath(user_input_path)
    print(f"使用的路径: {file_path}")
    
    file_type, data = import_file(file_path)

    if file_type == '.xlsx':
        excelfile_pretreated = pretreat_excelfile(data)
        treat_exceldata(excelfile_pretreated)
    if file_type == '.xls':
        # 处理xls文件
        pass
    if file_type == '.csv':
        # 处理csv文件
        pass
    

if __name__ == "__main__":
    main()
