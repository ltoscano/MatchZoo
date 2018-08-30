"""Matchzoo point generator."""

from matchzoo import engine
from matchzoo import tasks
from matchzoo import datapack
from matchzoo import utils

import numpy as np
import typing
import operator


class PointGenerator(engine.BaseGenerator):
    """PointGenerator for Matchzoo.

    Ponit generator can be used for classification as well as ranking.

    Examples:
        >>> import pandas as pd
        >>> relation = [['qid0', 'did0', 1]]
        >>> left = [['qid0', [1, 2]]]
        >>> right = [['did0', [2, 3]]]
        >>> relation = pd.DataFrame(relation, 
        ...                         columns=['id_left', 'id_right', 'label'])
        >>> left = pd.DataFrame(left, columns=['id_left', 'text_left'])
        >>> left.set_index('id_left', inplace=True)
        >>> right = pd.DataFrame(right, columns=['id_right', 'text_right'])
        >>> right.set_index('id_right', inplace=True)
        >>> input = datapack.DataPack(relation=relation,
        ...                           left=left,
        ...                           right=right
        ... )
        >>> task = tasks.Classification()
        >>> generator = PointGenerator(input, task, 1, 'train', False)
        >>> x, y = generator[0]
        >>> x['text_left'].tolist()
        [[1, 2]]

    """

    def __init__(
        self,
        inputs: datapack.DataPack,
        task: engine.BaseTask=tasks.Classification(2),
        batch_size: int=32,
        stage: str='train',
        shuffle: bool=True
    ):
        """Construct the point generator.

        :param inputs: the output generated by :class:`DataPack`.
        :param task: the task is a instance of :class:`engine.BaseTask`.
        :param batch_size: number of instances in a batch.
        :param shuffle: whether to shuffle the instances while generating a
            batch.
        """
        self._relation = self._transform_relation(inputs)
        self._task = task
        self._left = inputs.left
        self._right = inputs.right
        super().__init__(batch_size, len(inputs.relation), stage, shuffle)

    def _transform_relation(self, inputs: datapack.DataPack) -> dict:
        """Obtain the transformed data from :class:`DataPack`.

        :param inputs: An instance of :class:`DataPack` to be transformed.
        :return: the output of all the transformed relation.
        """
        relation = inputs.relation
        out = {}
        for column in relation.columns:
            out[column] = np.asarray(relation[column])
        return out

    def _get_batch_of_transformed_samples(
        self,
        index_array: list
    ) -> typing.Tuple[dict, typing.Any]:
        """Get a batch of samples based on their ids.

        :param index_array: a list of instance ids.
        :return: A batch of transformed samples.
        """
        batch_x = {}
        batch_y = None

        columns = self._left.columns.values.tolist() + \
            self._right.columns.values.tolist() + ['ids']
        for column in columns:
            batch_x[column] = []
        
        # Create label field.
        if self.stage == 'train':
            if isinstance(self._task, tasks.Ranking):
                batch_y = map(self._task.output_dtype, self._relation['label'])
            elif isinstance(self._task, tasks.Classification):
                batch_y = np.zeros((len(index_array), self._task.num_classes))
                for idx, label in enumerate(self._relation['label'][index_array]):
                    label = self._task.output_dtype(label)
                    batch_y[idx, label] = 1
            else:
                msg = f"{self._task} is not a valid task type."
                msg += ":class:`Ranking` and :class:`Classification` expected."
                raise ValueError(msg)

        # Get batch of X.
        ids_left = self._relation['id_left'][index_array]
        ids_right = self._relation['id_right'][index_array]

        [batch_x['ids'].append(list(item)) for item in zip(ids_left, ids_right)]

        for column in self._left.columns:
            batch_x[column] = self._left.loc[ids_left, column]
        for column in self._right.columns:
            batch_x[column] = self._right.loc[ids_right, column]

        for key, val in batch_x.items():
            batch_x[key] = np.array(val)
        batch_x = utils.dotdict(batch_x)

        return (batch_x, batch_y)
