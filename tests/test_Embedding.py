"""
    main_module - torchtext测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import sys
import unittest

import torch
from fastNLP.embeddings import StaticEmbedding
from fastNLP import Vocabulary

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import pysparnn  # noqa
print('pysparnn module path :{}'.format(pysparnn.__file__))  # 输出测试模块文件位置
from pysparnn import cluster_index as ci  # noqa


class TestTorchtext(unittest.TestCase):
    """torchtext测试.

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
        vocab = Vocabulary()
        for text in texts:
            vocab.add_word_lst(list(text))
        print(len(vocab))
        embed = StaticEmbedding(vocab, model_dir_or_name='./data/cn_char_fastnlp_100d.txt')
        texts_to_id = [[vocab.to_index(word) for word in list(text)] for text in ['朱日和', '东台变']]
        print(texts_to_id)  # [[16, 17, 18], [6, 1, 1]]
        words = torch.LongTensor(texts_to_id)  # 将文本转为index
        print(embed(words).size())  # torch.Size([2, 3, 100])

    @unittest.skip('debug')
    def test_search(self):
        """语义搜索.TypeError: expected dimension <= 2 array or matrix
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
        vocab = Vocabulary()
        for text in texts:
            vocab.add_word_lst(list(text))
        print(len(vocab))
        embed = StaticEmbedding(vocab, model_dir_or_name='./data/cn_char_fastnlp_100d.txt')
        texts_to_id = [[vocab.to_index(word) for word in list(text)] for text in texts]
        words = torch.LongTensor(texts_to_id)  # 将文本转为index
        features_vec = embed(words)
        print(features_vec.shape)
        # build the search index!
        cp = ci.MultiClusterIndex(features_vec.detach().numpy(), texts)
        search_texts = [
            '朱日和站',
            '温都尔站',
            '国电站'
        ]
        for text in search_texts:
            texts_to_id = [[vocab.to_index(word) for word in list(text)]]
            words = torch.LongTensor(texts_to_id)  # 将文本转为index
            features_vec = embed(words)
            search_features_vec = features_vec.detach().numpy()
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
