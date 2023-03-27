# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 21:56:01 2019

@author: 92156
"""

import xlrd
import pandas as pd
from pandas import DataFrame
wb = xlrd.open_workbook("0827cqbhebing.xlsx")
print(wb)
sheets = wb.sheet_names()
# print(sheets)
  
# 循环遍历所有sheet
df_28 = DataFrame()
for i in range(len(sheets)):

# skiprows=2 忽略前两行
    df = pd.read_excel(wb, sheet_name=i, skiprows=2, index=False, encoding='utf8')
    df_28 = df_28.append(df)
    

import os
import datetime
import xlrd
import xlsxwriter

TARPATH = 'C:\Users\92156\.spyder-py3'

#获取文件名
filename = []

#写入文件存储
records = []

#写入的目标文件：
tar_file = TARPATH +'0827cqbhebing.xlsx'

# print(tar_file)

#获取目录下的所有文件名
def get_filename(tar_path):
    for currentDir, _, includedFiles in os.walk(tar_path):
        if  currentDir.endswith('ex2'):
            for i in includedFiles:
                if i.endswith('xls') or i.endswith('xlsx'):
                    # print(i)
                    filename.append(tar_path + '/' + i)
        else:
            continue
    return filename


#获取excel文件的内容数据
def concat_and_insert(fdir,sheet_name = 'Sheet1',n = 2):
    if len(fdir)>0:
        for ai in fdir:
            # 读文件
            data = xlrd.open_workbook(ai)
            #第一个sheet页的名称;
            first_sheet = data.sheet_by_index(0).name
            print(ai,'>'*10,first_sheet)
            # 获取sheet页的名称
            sheet = data.sheet_by_name(sheet_name)
            #获取表的行数：
            nrows = sheet.nrows

            for i in range(nrows):
                # 跳过第一行
                if i < 2:
                    continue
                # print(sheet.row_values(i))
                records.append(sheet.row_values(i))
    return records


def insert_file(alist,tarfile):
    # 新建目标文件
    wh = xlsxwriter.Workbook(tarfile)
    wadd = wh.add_worksheet('total')
    if len(alist)>0:
        for row_num, row_data in enumerate(alist):
            wadd.write_row(row_num + 1, 0, row_data)
    wh.close()



if __name__ == '__main__':
    strat = datetime.datetime.now()
    print(strat)
    # time.sleep(2)
    get_filename(TARPATH)
    # print(filename)

    concat_and_insert(filename)
    # print(records)

    #写入文件
    insert_file(records,tar_file)

    end = datetime.datetime.now()
    print(end)
    print("持续时间{}".format(end-strat))
    print('ok')