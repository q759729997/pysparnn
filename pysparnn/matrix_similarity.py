# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Defines a similarity search structure for doing similarity search"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import numpy as np
import scipy.sparse
import scipy.spatial.distance

class MatrixMetricSearch(object):
    """A sparse matrix representation out of features."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, sparse_matrix, records_data, is_similarity=True):
        """
        Args:
            records_features: List of features in the format of
               {feature_name1 -> value1, feature_name2->value2, ...}.
            records_data: Data to return when a doc is matched. Index of
                corresponds to records_features.
        """
        self.is_similarity = is_similarity
        self.matrix = sparse_matrix 
        self.records_data = np.array(records_data)

    @abc.abstractmethod
    def _transform_value(self, val):
        return

    @abc.abstractmethod
    def _similarity(self, a_matrix):
        return

    def nearest_search(self, sparse_matrix, k=1, min_threshold=None,
                       max_threshold=None):
        """Find the closest item(s) for each set of features in features_list.

        Args:
            features_list: A list where each element is a list of features
                to query.
            k: Return the k closest results.
            min_threshold: Return items equal or above the threshold.
            max_threshold: Return items equal or below the threshold.

        Returns:
            For each element in features_list, return the k-nearest items
            and their similarity scores
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]
        """

        sim_matrix = self._similarity(sparse_matrix).toarray()

        if min_threshold == None:
            min_threshold = -1 * float("inf")

        if max_threshold == None:
            max_threshold = float("inf")

        sim_filter = sim_matrix >= min_threshold
        sim_filter &= (sim_matrix <= max_threshold)

        ret = []
        for i in range(sim_matrix.shape[0]):
            # these arrays are the length of the sqrt(index)
            # replacing the for loop by matrix ops could speed things up

            index = sim_filter[i]
            scores = sim_matrix[i][index]
            records = self.records_data[index]

            arg_index = None
            if self.is_similarity:
                arg_index = np.argsort(scores)[-k:]
            else:
                arg_index = np.argsort(scores)[:k]

            curr_ret = zip(scores[arg_index], records[arg_index])

            ret.append(curr_ret)

        return ret

class CosineSimilarity(MatrixMetricSearch):
    """A matrix that implements cosine similarity search against it."""

    def __init__(self, records_features, records_data):
        super(CosineSimilarity, self).__init__(records_features, records_data)

        m_c = self.matrix.copy()
        m_c.data **= 2
        self.matrix_root_sum_square = \
                np.sqrt(np.asarray(m_c.sum(axis=1)).reshape(-1))

    def _transform_value(self, v):
        return v

    def _similarity(self, a_matrix):
        """Vectorised cosine similarity"""
        # what is the implmentation of transpose? can i change the order?
        dprod = a_matrix.dot(self.matrix.transpose()) * 1.0

        # do i need to copy?
        a_c = a_matrix.copy()
        a_c.data **= 2
        a_root_sum_square = np.asarray(a_c.sum(axis=1)).reshape(-1)
        a_root_sum_square = \
                a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = np.sqrt(a_root_sum_square)

        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return dprod.multiply(magnitude)

class UnitCosineSimilarity(MatrixMetricSearch):
    """A matrix that implements cosine similarity search against it. 
    Assumes unit-vectors and takes some shortucts:
      * Uses integers instead of floats
      * 1**2 == 1 so that operation can be skipped
    """

    def __init__(self, records_features, records_data):
        super(UnitCosineSimilarity, self).__init__(records_features, 
                                                   records_data)
        self.matrix_root_sum_square = \
                np.sqrt(np.asarray(self.matrix.sum(axis=1)).reshape(-1))

    def _transform_value(self, v):
        return 1

    def _similarity(self, a_matrix):
        """Vectorised cosine similarity"""
        # what is the implmentation of transpose? can i change the order?
        dprod = a_matrix.dot(self.matrix.transpose()) * 1.0

        a_root_sum_square = np.asarray(a_matrix.sum(axis=1)).reshape(-1)
        a_root_sum_square = \
                a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = np.sqrt(a_root_sum_square)

        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return dprod.multiply(magnitude)

class SlowEuclideanDistance(MatrixMetricSearch):
    """A matrix that implements euclidean distance search against it. 
    WARNING: This is not optimized.
    """

    def __init__(self, records_features, records_data):
        super(SlowEuclideanDistance, self).__init__(records_features, 
                                                    records_data)
        self.matrix = self.matrix.toarray()

    def _transform_value(self, v):
        return v

    def _similarity(self, a_matrix):
        """Euclidean distance"""
        # need to handle fipping argmin k to positive
        return scipy.sparse.csr_matrix(scipy.spatial.distance.cdist(
                a_matrix.toarray(), 
                self.matrix, 'euclidean'))