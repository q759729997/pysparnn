"""
    main_module - TfidfVectorizer测试，测试时将对应方法的@unittest.skip注释掉.

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
    """TfidfVectorizer测试.

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
            'hello world',
            'oh hello there',
            'Play it',
            'Play it again Sam',
        ]
        tv = TfidfVectorizer()
        tv.fit(texts)
        print(tv.get_feature_names())  # ['again', 'hello', 'it', 'oh', 'play', 'sam', 'there', 'world']
        features_vec = tv.transform(texts[:2])
        print(features_vec.shape)  # (2, 8)
        print(features_vec)

    # @unittest.skip('debug')
    def test_search(self):
        """语义搜索.
        """
        print('{} test_search {}'.format('-'*15, '-'*15))
        texts = [
            'hello world',
            'oh hello there',
            'Play it',
            'Play it again Sam'
        ]
        # 文本向量化
        tv = TfidfVectorizer()
        tv.fit(texts)
        features_vec = tv.transform(texts)
        # build the search index!
        cp = ci.MultiClusterIndex(features_vec, texts)
        search_texts = [
            'Play it',
            'oh there',
            'Play it again Frank'
        ]
        for text in search_texts:
            search_features_vec = tv.transform([text])
            search_result = cp.search(search_features_vec, k=1, k_clusters=2, return_distance=True)
            print('text:{}'.format(text))
            print('search_result:{}'.format(search_result))
        """
        text:Play it
        search_result:[[('0.0', 'Play it')]]
        text:oh there
        search_result:[[('0.12656138024977548', 'oh hello there')]]
        text:Play it again Frank
        search_result:[[('0.16833831276420086', 'Play it again Sam')]]
        """


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
