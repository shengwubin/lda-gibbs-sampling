# coding=utf-8
"""邹博机器学习课程的LDA代码实现，参考文献为Parameter estimation for text analysis"""
import os
import os.path
import codecs
import jieba
import numpy as np


def calc_theta(alpha, n_d, nd_sum):
    """文献中的Eq.82，计算参数theta"""
    theta = np.divide(n_d + alpha, nd_sum + alpha + 0.0)
    return theta


def calc_phi(beta, n_t, nt_sum):
    """文献中的Eq.81，计算参数phi"""
    theta = np.divide(n_t + beta, nt_sum + beta + 0.0)
    return phi


def gibbs_sampling(doc_num, topic_number, term_num, z, m, i, nt, nd, nt_sum, nd_sum, term):
    not_finished = True
    while not_finished:
        pass


def lda(doc_num, topic_number, term_num, z, n_t, n_d, nt_sum, nd_sum, dic, doc):
    doc_num = len(z)
    for time in range(50):
        for m in range(doc_num):
            doc_length = len(z[m])
            for i in range(doc_length):
                term = dic[doc[m][i]]
                gibbs_sampling(z, m, i, nt, nd, nt_sum, nd_sum, term)
    theta = calc_theta(nd, nd_sum)
    phi = calc_phi(nt, nt_sum)
    return theta, phi


def load_stopwords():
    return []


def read_document(doc_num, stop_words, dic):
    doc = []
    dic_set = set()
    for i in range(doc_num):
        f = codecs.open('data/doc%d.txt' % (i + 1), 'r', 'utf-8')
        all_the_text = f.read()
        seg_list = jieba.cut(all_the_text)
        temp_list = []
        for word in seg_list:
            temp_list.append(word)
        doc.append(temp_list)
    for word_list in doc:
        dic_set = dic_set | set(word_list)
    for i, word in enumerate(dic_set):
        dic[word] = i
    return doc


def calc_summary(doc_num, topic_number, term_num, z, doc, nt, nd, nt_sum, nd_sum, dic):
    for i in range(term_num):
        for j in range(topic_number):
            nt[i][j] = 0
    for i in range(doc_num):
        for j in range(topic_number):
            nd[i][j] = 0
    for i in range(topic_number):
        nt_sum[i] = 0
    for i in range(doc_num):
        nd_sum[i] = 0
    for i, document in enumerate(doc):
        for j, word in enumerate(document):
            current_topic_number = z[i][j]
            current_word_number = dic[word]
            nt[current_word_number][current_topic_number] += 1
            nd[i][current_topic_number] += 1
            nt_sum[current_topic_number] += 1
            nd_sum[i] += 1


def init_topic(doc_num, topic_number, term_num, doc, nt, nd, nt_sum, nd_sum, dic):
    z = []
    for i, document in enumerate(doc):
        word_count = len(document)
        z.append([0 for t in range(word_count)])
    for i, document in enumerate(doc):
        word_count = len(document)
        topic_size = word_count / topic_number
        for j, word in enumerate(document):
            if topic_size != 0:
                if j / topic_size > topic_number - 1:
                    z[i][j] = topic_number - 1
                else:
                    z[i][j] = j / topic_size
            else:
                z[i][j] = 0
    calc_summary(doc_num, topic_number, term_num, z,
                 doc, nt, nd, nt_sum, nd_sum, dic)
    return z


def main():
    # 数据处理
    doc_num = 2  # 文档数
    topic_number = 4  # 主题数
    dic = {}
    stop_words = load_stopwords()
    doc = read_document(doc_num, stop_words, dic)
    term_num = len(dic.items())  # 词汇的数目
    # LDA
    # nt[w][t]：第term个词属于第t个主题的次数
    n_t = np.zeros((term_num, topic_number))
    #nt = [[0 for t in range(topic_number)] for term in range(term_num)]
    # nd[d][t]：第d个文档中出现第t个主题的次数
    n_d = np.zeros((doc_num, topic_number))
    #nd = [[0 for t in range(topic_number)] for d in range(doc_num)]
    # nt_sum[t]：第t个主题出现的次数（nt矩阵的第t列）
    nt_sum = np.zeros((topic_number, 1))
    #nt_sum = [0 for t in range(topic_number)]
    # nd_sum[t]：第d个文档的长度（nd矩阵的第d行）
    nd_sum = np.zeros((doc_num, 1))
    #nd_sum = [0 for d in range(doc_num)]
    z = init_topic(doc_num, topic_number, term_num,
                   doc, n_t, n_d, nt_sum, nd_sum, dic)
    theta, phi = lda(doc_num, topic_number, term_num, z,
                     n_t, n_d, nt_sum, nd_sum, dic, doc)
    show_result(theta, phi, dic)  # 输出每个文档的主题和每个主题的关键字
if __name__ == '__main__':
    main()
