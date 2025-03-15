import pandas as pd
import re
import json
# 读取并记录术语库
print("处理并记录术语库...")
df_term = pd.read_excel("legal_terms.xlsx")
terms = {}
for i, row in enumerate(df_term.iterrows()):
    # try:
    # print(row[1]['Weight'])
    if pd.notna(row[1]["Chinese_Term"]):
        weight = row[1]['Weight']
        content = []
        print(row[1]["Chinese_Term"])
        # 去除括号中的内容
        if "(" in row[1]["Chinese_Term"]:
            row[1]["Chinese_Term"] = re.sub(r'\([^)]*\)', '', row[1]["Chinese_Term"])
        # 将中文多个词分开并分别对应
        if ";" in row[1]["Chinese_Term"]:
            row[1]["Chinese_Term"] = row[1]["Chinese_Term"].replace(" ", "")
            cn_terms = row[1]["Chinese_Term"].split(";")
            content.append(row[1]["English_Term"])
            content.append(weight)
            for cn in cn_terms:
                terms[cn] = content
        else:
            content.append(row[1]["English_Term"])
            content.append(weight)
            terms[row[1]["Chinese_Term"]] = content
        content=[]
        # except Exception as e:
        #     print(f"读取并记录术语库出错：{e}!")

# 将结果保存为文本
with open("terms.txt", "w",encoding='utf-8') as f:
    json.dump(terms, f, ensure_ascii=False, indent=4)
print("术语库成功保存到terms.txt！")