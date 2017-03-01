# coding=utf-8
"""邹博机器学习课程的LDA代码实现，参考文献为Parameter estimation for text analysis"""
import os
import os.path
import codecs
import jieba
import numpy as np


def random_topic(topic_number, distribution=None):
    """产生服从多项分布的主题编号"""
    if distribution is not None:
        sample = np.random.multinomial(1, distribution, size=1)
    else:
        sample = np.random.multinomial(
            1, [1.0 / topic_number] * topic_number, size=1)
    return np.where(sample == 1)[1][0]


def calc_theta(alpha, n_d, nd_sum):
    """文献中的Eq.82，计算参数theta。"""
    theta = np.divide(n_d + alpha, nd_sum + alpha + 0.0)
    return theta


def calc_phi(beta, n_t, nt_sum):
    """文献中的Eq.81，计算参数phi。"""
    phi = np.divide(n_t + beta, nt_sum.T + beta + 0.0)
    return phi


def gibbs_sampling(alpha, beta, doc, dic, z_doc, n_t, n_d, nt_sum, nd_sum):
    """gibbs sampling算法实现"""
    sampling_time = 50
    while sampling_time > 0:
        for i, document in enumerate(doc):
            for j, word in enumerate(document):
                current_word_number = dic[word]
                current_topic_number = z_doc[i][j]
                n_t[current_word_number][current_topic_number] -= 1
                n_d[i][current_topic_number] -= 1
                nt_sum[current_topic_number] -= 1
                nd_sum[i] -= 1
                p_sampling = (n_t[current_word_number, :] + beta) * \
                    (n_d[i, :] + alpha) / (nt_sum.T + beta)
                p_sampling_norm = p_sampling / np.sum(p_sampling)
                current_topic_number = random_topic(
                    None, distribution=p_sampling_norm[0, :])
                z_doc[i][j] = current_topic_number
                n_t[current_word_number][current_topic_number] += 1
                n_d[i][current_topic_number] += 1
                nt_sum[current_topic_number] += 1
                nd_sum[i] += 1
        sampling_time -= 1


def lda(doc_num, topic_number, term_num, z_doc, n_t, n_d, nt_sum, nd_sum, dic, doc):
    """LDA核心"""
    alpha = 50.0 / topic_number
    beta = 0.01
    gibbs_sampling(alpha, beta, doc, dic, z_doc, n_t, n_d, nt_sum, nd_sum)
    theta = calc_theta(alpha, n_d, nd_sum)
    phi = calc_phi(beta, n_t, nt_sum)
    return theta, phi


def load_stopwords():
    """停止词十分重要，但是没有找到合适的词库，暂时为空。"""
    stop_words_file = codecs.open('data/stopword.txt', 'r', 'utf-8')
    stop_words_set = set(stop_words_file.read().split(
        '\r\n')) | set(['', '\r\n'])
    return stop_words_set


def read_document(doc_num, stop_words, dic):
    """读取文档，并生成词典。"""
    doc = []
    dic_set = set()
    for i in range(doc_num):
        current_file = codecs.open('data/doc%d.txt' % (i + 1), 'r', 'utf-8')
        all_the_text = current_file.read()
        seg_list = jieba.cut(all_the_text)
        temp_word_list = []
        for word in seg_list:
            if word not in stop_words:
                temp_word_list.append(word)
        doc.append(temp_word_list)
        dic_set = dic_set | set(doc[i])
    for i, word in enumerate(dic_set):
        dic[word] = i
    return doc


def init_topic(topic_number, doc, n_t, n_d, nt_sum, nd_sum, dic):
    """初始化主题分布"""
    z_doc = []
    for document in doc:
        word_count = len(document)
        z_doc.append([0 for t in range(word_count)])
    for i, document in enumerate(doc):
        for j, word in enumerate(document):
            current_topic_number = random_topic(topic_number)
            z_doc[i][j] = current_topic_number
            current_word_number = dic[word]
            n_t[current_word_number][current_topic_number] += 1
            n_d[i][current_topic_number] += 1
            nt_sum[current_topic_number] += 1
            nd_sum[i] += 1
    return z_doc


def show_result(theta, phi, dic):
    """打印计算结果"""
    dic_now = dict((value, key) for key, value in dic.iteritems())
    for i in range(phi.shape[1]):
        topic_keyword = dic_now[np.argmax(phi[:, i])]
        print 'topic%d:%s' % (i + 1, dic_now[np.argmax(phi[:, i])])
    for i in range(theta.shape[0]):
        distribution = [str(prob) for prob in theta[i, :]]
        print 'document%d:%s' % (i + 1, str(distribution))


def main():
    """LDA主程序"""
    # 数据处理
    doc_num = 2  # 文档数
    topic_number = 10  # 主题数
    dic = {}
    stop_words = load_stopwords()
    doc = read_document(doc_num, stop_words, dic)
    term_num = len(dic.items())  # 词汇的数目
    # LDA
    n_t = np.zeros((term_num, topic_number))  # nt[w][t]：第term个词属于第t个主题的次数
    n_d = np.zeros((doc_num, topic_number))  # nd[d][t]：第d个文档中出现第t个主题的次数
    nt_sum = np.zeros((topic_number, 1))  # nt_sum[t]：第t个主题出现的次数（nt矩阵的第t列）
    nd_sum = np.zeros((doc_num, 1))  # nd_sum[t]：第d个文档的长度（nd矩阵的第d行）
    z_doc = init_topic(topic_number, doc, n_t, n_d, nt_sum, nd_sum, dic)
    theta, phi = lda(doc_num, topic_number, term_num, z_doc,
                     n_t, n_d, nt_sum, nd_sum, dic, doc)
    show_result(theta, phi, dic)  # 输出每个文档的主题和每个主题的关键字
if __name__ == '__main__':
    main()
