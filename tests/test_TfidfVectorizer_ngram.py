"""
    main_module - TfidfVectorizer中文测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import sys
import unittest

from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import pysparnn  # noqa
print('pysparnn module path :{}'.format(pysparnn.__file__))  # 输出测试模块文件位置
from pysparnn import cluster_index as ci  # noqa


class TestTfidfVectorizer(unittest.TestCase):
    """TfidfVectorizer中文测试.

    Main methods:
        test_fit - 文本编码.
        test_search - 语义搜索.
    """
    @unittest.skip('debug')
    def test_fit(self):
        """文本编码.
        """
        print('{} test_fit {}'.format('-'*15, '-'*15))
        texts = [
            '温都尔站',
            '东乌广厦',
            '国电四郎',
            '阿尔善站',
            '朱日和基'
        ]
        tv = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
        tv.fit(texts)
        print(tv.get_feature_names())  # ['东乌', '乌广', '和基', '善站', '四郎', '国电', '尔善', '尔站', '广厦', '日和', '朱日', '温都', '电四', '都尔', '阿尔']
        features_vec = tv.transform(texts[:2])
        print(features_vec.shape)  # (2, 15)
        print(features_vec.toarray())
        """
        [[0.         0.         0.         0.         0.         0.
  0.         0.57735027 0.         0.         0.         0.57735027
  0.         0.57735027 0.        ]
 [0.57735027 0.57735027 0.         0.         0.         0.
  0.         0.         0.57735027 0.         0.         0.
  0.         0.         0.        ]]
        """

    # @unittest.skip('debug')
    def test_search(self):
        """语义搜索.
        """
        print('{} test_search {}'.format('-'*15, '-'*15))
        texts = [
            '温都尔站',
            '东乌广厦',
            '国电四郎',
            '阿尔善站',
            '朱日和基'
        ]
        # 文本向量化
        tv = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
        tv.fit(texts)
        features_vec = tv.transform(texts)
        # build the search index!
        cp = ci.MultiClusterIndex(features_vec, texts)
        search_texts = [
            '朱日和站',
            '温都尔站',
            '国电站'
        ]
        for text in search_texts:
            search_features_vec = tv.transform([text])
            search_result = cp.search(search_features_vec, k=2, k_clusters=2, return_distance=True)
            print('text:{}'.format(text))
            print('search_result:{}'.format(search_result))
        """
        text:朱日和站
        search_result:[[('0.18350341907227374', '朱日和基'), ('1.0', '温都尔站')]]
        text:温都尔站
        search_result:[[('-2.220446049250313e-16', '温都尔站'), ('1.0', '东乌广厦')]]
        text:国电站
        search_result:[[('0.42264973081037416', '国电四郎'), ('1.0', '温都尔站')]]
        """


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
