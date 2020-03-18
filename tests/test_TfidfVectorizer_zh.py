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
        tv = TfidfVectorizer(analyzer='char')
        tv.fit(texts)
        print(tv.get_feature_names())  # ['东', '乌', '厦', '和', '善', '四', '国', '基', '尔', '广', '日', '朱', '温', '电', '站', '郎', '都', '阿']
        features_vec = tv.transform(texts[:2])
        print(features_vec.shape)  # (2, 18)
        print(features_vec.toarray())
        """
        [[0.         0.         0.         0.         0.         0.
  0.         0.         0.44400208 0.         0.         0.
  0.55032913 0.         0.44400208 0.         0.55032913 0.        ]
 [0.5        0.5        0.5        0.         0.         0.
  0.         0.         0.         0.5        0.         0.
  0.         0.         0.         0.         0.         0.        ]]
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
        tv = TfidfVectorizer(analyzer='char')
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
        search_result:[[('0.21496262459568471', '朱日和基'), ('0.8125236116183452', '温都尔站')]]
        text:温都尔站
        search_result:[[('0.0', '温都尔站'), ('0.6057243079292995', '阿尔善站')]]
        text:国电站
        search_result:[[('0.3858110336573437', '国电四郎'), ('0.7799864067596534', '温都尔站')]]
        """


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
