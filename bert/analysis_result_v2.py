import json
import os
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font='AppleGothic')

check_answer_dir = './check_answer/'
dev_json = check_answer_dir + 'dev.json'
exact_json = check_answer_dir + 'exact.json'
pattern1_json = check_answer_dir + 'pattern1.json'
pattern2_json = check_answer_dir + 'pattern2.json'
pattern3_json = check_answer_dir + 'pattern3.json'
json_files = [dev_json, exact_json, pattern1_json, pattern2_json, pattern3_json]
question_heads = ['who', 'when', 'where', 'what', 'why', 'how', '#other#']

analysis_result = {}
for json_file in json_files:
    question_head_counts = [0]*len(question_heads)
    other_counts = 0 # question_headsに入らなかった数をカウントする用
    with open(json_file, 'r') as f:
        data_dict = json.load(f)

    # data_dict:
    #     {
    #       "56dde1d966d3e219004dad8d": "Who upon arriving gave the original viking settlers a common identity?",
    #       "56dde27d9a695914005b9651": "What was the Norman religion?",
    #       ...
    #     }
    for q_id, question in data_dict.items():
        q_head = question.strip().split(' ')[0].lower()
        try:
            add_index = question_heads.index(q_head)     
            question_head_counts[add_index] += 1
        except ValueError:
            question_head_counts[-1] += 1 # #other#

    assert len(data_dict) == sum(question_head_counts)

    # question_heads: ['who', 'when', 'where', 'what', 'why', 'how', '#other#']
    # question_head_counts: [13, 10, 20, 11, 1, 20, 13]

    filekey = os.path.basename(json_file).split('.json')[0]
    analysis_result[filekey] = {}
    analysis_result[filekey]['total'] = len(data_dict)
    analysis_result[filekey]['each'] = {}
    for head, count in zip(question_heads, question_head_counts):
        analysis_result[filekey]['each'][head] = count

# print('analysis_result:')
# pprint(analysis_result)

df_dict = {}
for q_head in question_heads:
    df_dict[q_head] = []

patterns = []
for pattern, each_result in analysis_result.items():
    patterns.append(pattern)
    for q_head, count in each_result['each'].items():
        df_dict[q_head].append(count)

# --- pie chart plot ---
# for column_name, item in df.T.iteritems():
#     # type(item): <class 'pandas.core.series.Series'>ß
#     plt.figure()
#     item.plot.pie(subplots=True, autopct='%.1f%%')
#     plt.savefig(f"./{check_answer_dir}/img/{column_name}_pie.png")

patterns.append('incorrect')
for q_head in question_heads:
    df_dict[q_head].append(df_dict[q_head][0] - df_dict[q_head][1])

print('~ counts data frame ~\n')
df = pd.DataFrame(df_dict, index=patterns)
pprint(df)
print('-'*50)

df_ratio = df.copy()
for q_head in question_heads:
    df_ratio.at['exact', q_head] = 100*df.at['exact', q_head]/df.at['dev', q_head]
    df_ratio.at['pattern1', q_head] = 100*df.at['pattern1', q_head]/df.at['incorrect', q_head]
    df_ratio.at['pattern2', q_head] = 100*df.at['pattern2', q_head]/df.at['incorrect', q_head]
    df_ratio.at['pattern3', q_head] = 100*df.at['pattern3', q_head]/df.at['incorrect', q_head]

df_ratio = pd.concat([df_ratio, pd.DataFrame(df_ratio.mean(axis=1), columns=['mean'])], axis=1)
df_ratio = df_ratio.drop('dev', axis=0).drop('incorrect', axis=0)
print('~ ratio data frame ~\n')
pprint(df_ratio)
print('-'*50)

df_ratio2 = df_ratio.copy()
for q_head in question_heads:
    df_ratio2.at['exact', q_head] = 100*df_ratio.at['exact', q_head]/df_ratio.at['exact', 'mean']
    df_ratio2.at['pattern1', q_head] = 100*df_ratio.at['pattern1', q_head]/df_ratio.at['pattern1', 'mean']
    df_ratio2.at['pattern2', q_head] = 100*df_ratio.at['pattern2', q_head]/df_ratio.at['pattern2', 'mean']
    df_ratio2.at['pattern3', q_head] = 100*df_ratio.at['pattern3', q_head]/df_ratio.at['pattern3', 'mean']

df_ratio2 = df_ratio2.drop('mean', axis=1)
print('~ ratio data frame(mean div.) ~\n')
pprint(df_ratio2)
print('-'*50)
for column_name, item in df_ratio2.T.iteritems():
    plt.figure()
    item.plot.bar()
    if 'pattern' not in column_name:
        title = column_name + '/dev'
    else:
        title = column_name + '/incorrect'
    plt.title(title)
    plt.xlabel('question type')
    plt.ylabel('ratio')
    plt.hlines(100, -1, 7, 'red')
    plt.savefig(f"./{check_answer_dir}/img/{column_name}_bar.png")