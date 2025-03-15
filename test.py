import json
import time

import pandas as pd
import re

# from transformers import BertTokenizer, BertForMaskedLM
import spacy

from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity


def align_nouns(source_nouns, target_nouns):
    # # 加载模型
    model = SentenceTransformer(r'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    alignment = []
    source_embeddings = model.encode(source_nouns)
    target_embeddings = model.encode(target_nouns)

    for i, source_noun in enumerate(source_nouns):
        max_similarity = -1
        best_match = None
        for j, target_noun in enumerate(target_nouns):
            similarity = (source_embeddings[i] @ target_embeddings[j]) / (
                    (source_embeddings[i] @ source_embeddings[i]) ** 0.5 * (
                        target_embeddings[j] @ target_embeddings[j]) ** 0.5)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = target_noun
        if best_match and max_similarity > 0.2:  # 可以调整相似度阈值
            alignment.append((source_noun, best_match))

    # final_simi=alignment[0][0][1][1] * max_similarity
    # print(f"对齐后的术语为: {alignment}, 相似度: {max_similarity}, 权重:{alignment[0][0][1][1] }, 加权相似度为: {final_simi}\n")
    print(alignment)
    if not isinstance(alignment[0][0],str):
        print(f"对齐后的术语为: {alignment}, 相似度: {max_similarity}, 权重:{alignment[0][0][1][1] }\n")
        return max_similarity, alignment[0][0][1][1]
    else:
        print(f"对齐后的术语为: {alignment}, 相似度: {max_similarity}\n")
        return max_similarity
def terminology_consistency_score(term_path = "terms.txt", original_text="", translated_text=""):
    '''
    计算术语一致性得分
    :param term_path: 转换后的术语文本路径
    :param data_path: 原文与译文数据路径，需包含'原文'，'译文'，TODO 'Weight'
    :return: terms_score
    '''
    # 计算术语一致性得分

    # 1. 提取术语-术语库匹配 /依存句法分析（由于术语包含多种词性，暂不考虑）
    # 读取术语库
    with open(term_path,encoding="utf-8") as f:
        terms = json.load(f)
    # 表格读取需要评分的文本

    terms_score = 0
    terms_score_up = 0
    terms_score_down = 0

    # 逐个检查术语是否出现在句子中
    found_terms = [(term,value) for index,(term,value) in enumerate(terms.items()) if re.search(r'\b' + re.escape(term) + r'\b', original_text)]
    if found_terms:
        print(f"\n\n原文：{original_text} 查找的术语：{found_terms}")
        # 若出现，比较译文是否一致
        for item in found_terms:
            if re.search(r'\b' + re.escape(item[1][0]) + r'\b', translated_text):
                terms_score_up += 1 * item[1][1]
                terms_score_down += item[1][1]
                print(f"译文 {translated_text} 中查找到对应译文术语： {item[1][0]}！")
            else:
                print(f"译文 {translated_text} 中未查找到对应译文术语： {item[1][0]}, 开始进行译文术语提取...")
                # 查找译文术语
                found_eng_terms = [(term, value) for index, (term, value) in enumerate(terms.items()) if
                               re.search(r'\b' + re.escape(value[0]) + r'\b', translated_text)]
                # 术语对齐
                if found_eng_terms:
                    print(f"译文术语提取成功！提取术语为 {found_eng_terms}, 开始进行原文和译文术语对齐...\n")
                    temp, weight = align_nouns(found_terms,found_eng_terms)
                    terms_score_up += weight * temp
                    terms_score_down += weight
                else:
                    print("译文中未提取到术语！")

        if terms_score_down:
            print(terms_score_up,terms_score_down)
            terms_score = terms_score_up / terms_score_down
        print(f"术语一致性得分为：{terms_score}\n")

    return terms_score, found_terms

def extract_legal_releation(text, legal_releation_key, model):
    print(text)
    # 定义字典分别存储11种逻辑关系
    legal_releation_num = {"tiaojian": 0, "yiwu": 0, "jinzhi": 0, "yinguo": 0,
                              "shijian": 0, "xuanze": 0, "paichu": 0, "dijin": 0,
                              "jieshi": 0, "tidai": 0, "chufa": 0}
    # 加载spaCy的模型
    nlp = spacy.load('./venv/legal/bin/'+model+'/'+model+'-3.8.0')
    doc = nlp(text)

    print("依存句法分析中...")
    # 依存句法结合关键词提取逻辑关系
    for token in doc:
        if token.dep_ == "neg":
            legal_releation_num["jinzhi"] += 1
            print(f"语句中存在禁止关系，禁止关键词为：{token}！")
        if token.dep_ == "appos":
            legal_releation_num["jieshi"] += 1
            print(f"语句中存在解释关系，解释关键词为：{token}！")
    # print("依存句法分析完毕！\n")
    # 对于依存句法未抽取到的逻辑关系种类，进行关键词逻辑抽取
    print(f"开始进行关键词匹配...")
    for (key,value) in legal_releation_num.items():
        if not legal_releation_num[key]:
            # print(f"依存句法未提取到 {key} 逻辑关系，开始进行关键词匹配...")
            # pattern = '|'.join(re.escape(i) for i in legal_releation_key[key])
            if '...' not in legal_releation_key[key][0]:
                pattern = legal_releation_key[key][0]
            else:
                pattern =  legal_releation_key[key][0].split('...')[0] + '.*?' + legal_releation_key[key][0].split('...')[1]
            for i in legal_releation_key[key][1:]:
                i = i.strip()
                if '...' in i:
                    pattern += '|'+i.split('...')[0]+'.*?'+i.split('...')[1]
                else:
                    pattern += '|'+i

            num = re.findall(pattern, text)
            if num:
                legal_releation_num[key] += len(num)
                print(f"  提取到 {key} 逻辑关系 {len(num)} 条：{num}\n")
            # else:
            #     print(f"未提取到 {key} 逻辑关键词！\n")
    return legal_releation_num

# 逻辑一致性评分：通过句法依存关系匹配
def logical_consistency_score(original_text, translated_text, en_legal_releation_key,zh_legal_releation_key, legal_weight):
    result = 0
    print("提取译文逻辑关系中...\n")
    translated_num = extract_legal_releation(translated_text, en_legal_releation_key,model ="en_core_web_sm")
    print("提取原文逻辑关系中...\n")
    original_num = extract_legal_releation(original_text, zh_legal_releation_key, model = "zh_core_web_sm")
    print(f"原文和译文逻辑关系数量分别为：\n原文：{original_num}\n译文：{translated_num}\n")

    if translated_num.keys() == legal_weight.keys() and original_num.keys() == legal_weight.keys():
        trans_res = 0
        orig_res = 0
        for key in translated_num:
            trans_res += min(translated_num[key], original_num[key]) * legal_weight[key]
            orig_res += original_num[key] * legal_weight[key]
        if orig_res:
            result = trans_res / orig_res
    print(f"\n该条翻译的逻辑得分为: {result}\n")
    return result, all(v==0 for v in original_num.values()), all(v==0 for v in translated_num.values())
def bigmodel_ds(text):
    import dashscope
    DASHSCOPE_API_KEY = "sk-6c32bf3844e9439ea6a50cb9de638a5a"
    messages = [
        {
            "role": "user",
            "content": '''你是一个能够理解中英文语句的法律专家。你的任务是提取用户给出中英文句子中的表示法律效力的谓词，并将其分为规范强度动词和施为力度动词的对应等级，
                               其中谓词的定义是指法律条文或法律陈述中表达法律效果或法律关系的核心动词；规范强度动词指法律语言中能够表示约束力强度的动词，分为强约束、中约束和弱约束三个等级；施为力度动词施为动词指在法律语言中能起到表达法律效果、规定权利义务、界定法律关系的动词，分为规定义务，限制义务和赋予权力三个等级。
                               请以 '{"规范":{"强约束":["xx"],"中约束":["xx"],"弱约束":["xx"]}, "施为":{"规定义务":["xx"],"限制义务":["xx"],"赋予权力":["xx"]}}' 的字典形式返回，若句中没有表示法律效力的词，则返回空字典。**注意：1、请切勿输出非字典的其他任何文字；2、请从用户的输入中解析，不要输出未输入的内容**。
                               例如：用户输入语句: "(1) 通知银行收到买方开具的不可撤销信用证时，卖方必须开具信用证10%金额的履约担保。",输出 '{"规范":{"强约束":["不可撤销","必须"],"中约束":[],"弱约束":[]}, "施为":{"规定义务":["必须"],"限制义务":[],"赋予权力":[]}}'；
                               用户输入语句："合同货物的质量、性能、数量经双方确认，并签署本合同，条款如下：",输出 '{}'。
                               任务开始："''' + text
        }

    ]

    response = dashscope.Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=DASHSCOPE_API_KEY,
        model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
        messages=messages,
        # result_format参数不可以设置为"text"。
        result_format='message'
    )

    if "```json" in response.output.choices[0].message.content:
        result=response.output.choices[0].message.content.split("```json")[1].split("```")[0]
    elif "</think>" in response.output.choices[0].message.content:
        result = response.output.choices[0].message.content.split("</think>")[1].strip()
    else:
        result=response.output.choices[0].message.content
    # print(response.json()['message']['content'])
    # print(type(result),result)
    return result

def bigmodel_ds_test(text):
    import requests

    data = {
        "model": "deepseek-r1:7b",
        "messages": [
            {
                "role": "system",
                "content": '''你是一个能够理解中英文语句的法律专家。你的任务是提取用户给出中英文句子中的表示法律效力的谓词，并将其分为规范强度动词和施为力度动词的对应等级，
                           其中谓词的定义是指法律条文或法律陈述中表达法律效果或法律关系的核心动词；规范强度动词指法律语言中能够表示约束力强度的动词，分为强约束、中约束和弱约束三个等级；施为力度动词施为动词指在法律语言中能起到表达法律效果、规定权利义务、界定法律关系的动词，分为规定义务，限制义务和赋予权力三个等级。
                           请以 '{"规范":{"强约束":["xx"],"中约束":["xx"],"弱约束":["xx"]}, "施为":{"规定义务":["xx"],"限制义务":["xx"],"赋予权力":["xx"]}}' 的字典形式返回，若句中没有表示法律效力的词，则返回空字典。**注意：1、请切勿输出非字典的其他任何文字；2、请从用户的输入中解析，不要输出未输入的内容**。
                           例如：用户输入语句: "(1) 通知银行收到买方开具的不可撤销信用证时，卖方必须开具信用证10%金额的履约担保。",输出 '{"规范":{"强约束":["不可撤销","必须"],"中约束":[],"弱约束":[]}, "施为":{"规定义务":["必须"],"限制义务":[],"赋予权力":[]}}'；
                           用户输入语句："合同货物的质量、性能、数量经双方确认，并签署本合同，条款如下：",输出 '{}'。
                           任务开始："'''},
            {
                "role": "user",
                "content": text
            }
        ],
        "stream": False
    }
    response = requests.post(url="http://localhost:11434/api/chat", json=data)
    if "```json" in response.json()['message']['content']:
        result=response.json()['message']['content'].split("```json")[1].split("```")[0]
    elif "</think>" in response.json()['message']['content']:
        result = response.json()['message']['content'].split("</think>")[1].strip()
    else:
        result=response.json()['message']['content']
    # print(response.json()['message']['content'])
    # print(type(result),result)
    return result
# TODO 语句不存在逻辑关系或术语时不参与计算该项得分，而不是为0
# 法律效力等效性评分
def legal_effectiveness_score(original_text, translated_text, guifan_level_score):
    # 提取原文和译文效力词 施为/规范
    effect_flag = False
    effect_score_shiwei = 0
    effect_score_guifan = 0
    num_guifan =0
    num_shiwei = 0
    try:
        origin_effect_num = json.loads(bigmodel_ds(original_text).strip())
        trans_effect_num = json.loads(bigmodel_ds(translated_text).strip())
        num_shiwei = 0
        # 进行动词对齐-按照每一个等级进行动词对齐
        for i in origin_effect_num["施为"]:
            print("施为动词三个类别提取结果为: \n",i,origin_effect_num["施为"][i],trans_effect_num["施为"][i])
            if origin_effect_num["施为"][i] and trans_effect_num["施为"][i]:
                num_shiwei +=1
                print("对齐的施为动词及相似度为： ",)
                effect_flag = True
                effect_score_shiwei += align_nouns(origin_effect_num["施为"][i],trans_effect_num["施为"][i])
            elif origin_effect_num["施为"][i] or trans_effect_num["施为"][i]:
                effect_flag = True
                effect_score_shiwei = 0
        num_guifan = 0
        for i in origin_effect_num["规范"]:
            print("规范动词三个类别提取结果为: \n",i, origin_effect_num["规范"][i], trans_effect_num["规范"][i])
            if origin_effect_num["规范"][i] and trans_effect_num["规范"][i]:
                num_guifan +=1
                # 按照等级计算得分
                if i=="强约束":
                    effect_score_guifan += guifan_level_score['强约束']
                elif i=='中约束':
                    effect_score_guifan += guifan_level_score['中约束']
                elif i=='弱约束':
                    effect_score_guifan += guifan_level_score['弱约束']
                effect_flag = True
            elif origin_effect_num["规范"][i] or trans_effect_num["规范"][i]:
                effect_flag = True
                effect_score_guifan = 0

    except Exception as e:
        print(f"原文 {original_text} 或其译文返回格式不正确！",e)
    if num_guifan and num_shiwei:
        effect_score =(effect_score_guifan/num_guifan + effect_score_shiwei/num_shiwei) / 2
    elif num_shiwei:
        effect_score = effect_score_shiwei/num_shiwei
    elif num_guifan:
        effect_score = effect_score_guifan/num_guifan
    else:
        effect_score = 0
    print(f"\n该项效力一致性得分为{effect_score}\n")
    return effect_score, effect_flag
    # return cosine_similarity([original_embedding], [translated_embedding])[0][0]


# 通过BERT获取句子的向量表示
def get_bert_embeddings(text):
    # 加载BERT模型
    tokenizer = BertTokenizer.from_pretrained(
        r"D:/LYN/WorkFiles/NLP/序列标注模型/bert-base-cased/bert-base-cased/")  # bert-base-uncased
    model = BertForMaskedLM.from_pretrained(r"D:/LYN/WorkFiles/NLP/序列标注模型/bert-base-cased/bert-base-cased/")
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()



# 综合评分
def overall_score(original_text, translated_text):
    term_score,  = terminology_consistency_score(original_text, translated_text, terminology_dict)
    logic_score = logical_consistency_score(original_text, translated_text)
    legal_effectiveness_score_value = \
        ++++(original_text, translated_text)

    return (term_score + logic_score + legal_effectiveness_score_value) / 3


# ----------------------------------------------------------------------------------------------------------------

#数据准备
def all_score(original_text,translated_text):
    # 逻辑关系关键词 /或存储为文本形式 TODO 需考虑多个分开词的情况...
    en_legal_releation_key = {"tiaojian": ["in the event that","whereas","unless","if","provided that", "subject to","where", "in the absence of","on condition that",
                                           "should", "as long as","once","in case of","whether...or not","conditional upon"],
                              "yiwu": ["shall","must","obligated to","obligation","required to","duty","agree to","undertake to","responsible for","bear the liability",
                                       "guarantee","ensure","warrant","has the obligation"],
                              "jinzhi": ["shall not","must not", "may not", "prohibited","not allowed to","no right","not entitled","forbidden","without the consent of","not allowed","cannot","restriction on"],
                              "yinguo": ["because","caused by","owing to","due to","as a result","thus","consequently","as a result of","because of","by reason of","in consequence of","lead to","result in","thereby","therefore","hence"],
                              "shijian": ["before","after","upon","when","while","during","prior to","within","from the date","until","following","immediately after","subsequent to","commencing on"],
                              "xuanze": ["either...or","or","at the option of","may choose to","alternatively","whether...or","in lieu of","substitute","in place of"],
                              "paichu": ["except","unless","excluding","notwithstanding","except as otherwise","except","excluding","save for","other than","with the exception of","not including","except as provided","not applicable"],
                              "dijin": ["furthermore","in addition","also","moreover","and","further","additionally","in addition","what is more","besides","based on"],
                              "jieshi": ["including","such as","for example","namely","i.e.","that is to say","including but not limited to","for the avoidance of doubt","defined in","in particular","meaning","refer to"],
                              "tidai": ["instead","in lieu of","alternatively","or","instead of","replace","substitute","in place of","supersede","take over","supersede","prevail over"],
                              "chufa": ["upon the occurrence of","in the event of","if...then...","when","riggers","upon","when","in the event of","trigger","if and when","in the event that","in case","where...arises","give rise to","result in","activate","automatically","deemed"]}

    zh_legal_releation_key = {"tiaojian": ["若","如果","假如","倘若","一旦","只有在...时", "除非","如果...则","当...时","经...后","除...外", "符合...条件","一旦...就","在...情形下",
                                           "以...为条件","经...同意","需...的","经...批准","经...确认","发生...情形","满足...时","的情况下","情况下"],
                              "yiwu": ["应当","必须","需要","有义务","应","需","负有...的义务","应当按照","必须遵守","有责任","负有...义务","承担","保证","负责","不得拒绝","应当及时","承诺","须"],
                              "jinzhi": ["不得","禁止", "不可", "无权","不允许","不能","不应","严禁"],
                              "yinguo": ["因为","由于","因此","所以","导致","造成","结果","因而","致使","由此","基于","鉴于","因...原因","因...致使","因...造成","影响...实现","因","故"],
                              "shijian": ["在...之前","当...时","一旦","随后","之前","之后","同时","先后","在...期间","自...时起","在...之前","在...之后","届满","期限","到期","同时",
                                          "即时","立即","及时","提前","逾期","自...之日起","持续期间","经催告后","合理期限内","前","后","内","自...起"],
                              "xuanze": ["或者","要么","可选择","可以选择","也可以","有两种选择","可以...也可以","有权选择","或者...或者","优先","替代","替代履行","协商","协商不成","选择权","变更","调整","或","任选其一"],
                              "paichu": ["除非","例外","不包括","但","然而","不过","除非另有规定","不适用","排除","不包括","但是","除...外","另有约定","另有规定","除外","不在此限","不受影响","不承担","不视为","不适用","但书条款","特殊情形","例外情形","但根据"],
                              "dijin": ["进一步","此外","而且","不仅","进一步说","也","并且","还","同时","不仅...还","除...外","首先","其次","最后","在此基础上"],
                              "jieshi": ["即","也就是","具体而言","例如","比如","换句话说","具体来说","包括但不限于","也就是说","视为","推定","按照","根据","依据","参照","适用","推定为","指","定义为"],
                              "tidai": ["或者","替代","替代履行","代位行使","代为支付","取代","变更","调整","替换","替代物","转让","转委托","转租","转包","代替","置换","以...为准"],
                              "chufa": ["一旦","如果","在...的情况下","如果...则","发生...时","当...时","经...后","出现...情形","达到...标准","超过...期限","满足...条件","经催告","经通知","经请求","经批准","经同意","经确认","经催告后","经评估","经检验","自动","视为","到期未...则"]}

    legal_weight = {"tiaojian": 1, "yiwu": 0.9,"jinzhi": 0.9, "yinguo": 0.8,
                     "shijian": 0.7, "xuanze": 0.8, "paichu": 0.75, "dijin": 0.6,
                     "jieshi": 0.6, "tidai": 0.7, "chufa": 0.6}

    guifan_level_score ={'强约束':1,'中约束':0.8,'弱约束':0.6}



    term_score, found_terms = terminology_consistency_score(term_path = "terms.txt", original_text=original_text, translated_text=translated_text)
    if found_terms:
        # term_num += 1
        term_flag =True
    else: term_flag=False

    legal_score, origin_legal_flag, trans_legal_flag = logical_consistency_score(original_text, translated_text, en_legal_releation_key, zh_legal_releation_key, legal_weight)    # TODO 计算效力等效性得分
    if not origin_legal_flag or not trans_legal_flag:
        # legal_num += 1
        legal_flag=True
    else:
        legal_flag=False

    effect_score, effect_flag = legal_effectiveness_score(original_text, translated_text, guifan_level_score)
    # if effect_flag:
        # power_num+=1
    # 总分数计算
    score = (term_flag* term_score +legal_flag* legal_score +effect_flag * effect_score) / (term_flag+legal_flag+effect_flag)

    print(f"该条术语一致性得分为{term_flag* term_score}, 逻辑一致性得分为{legal_flag* legal_score}, 效力等效性得分为{effect_flag * effect_score}, 总得分为{score}")

    return score,term_flag,legal_flag,effect_flag

data_path = "测试语料1000条.xlsx"
df =pd.read_excel(data_path)
term_num = 0
legal_num =0
power_num = 0
time1 = time.time()
for i, row in enumerate(df.iterrows()):
    original_text = row[1]['中文']
    translated_text = row[1]['英文']
    score,term_flag,legal_flag,effect_flag = all_score(original_text,translated_text)
    if term_flag:term_num+=1
    if legal_flag:legal_num+=1
    if effect_flag:power_num+=1

time2 = time.time()
print(f"共{i + 1}条文本，其中{term_num}条中包含术语，{legal_num}条中包含逻辑关系，{power_num}条中包含效力动词，花费时间为 {time2 - time1}s.")