"""CDSSM (CLSM) Preprocessor."""

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack

import typing
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CDSSMPreprocessor(engine.BasePreprocessor):
    """
    CDSSM (CLSM) preprocessor.

    Example:
        >>> train_inputs = [
        ...     ("id0", "id1", "beijing", "Beijing is capital of China", 1),
        ...     ("id0", "id2", "beijing", "China is in east Asia", 0),
        ...     ("id0", "id3", "beijing", "Summer in Beijing is hot.", 1)
        ... ]
        >>> cdssm_preprocessor = CDSSMPreprocessor()
        >>> rv_train = cdssm_preprocessor.fit_transform(
        ...     train_inputs,
        ...     stage='train')
        >>> cdssm_preprocessor.context['input_shapes'][0][1]
        37
        >>> type(rv_train)
        <class 'NoneType'>
        >>> context = cdssm_preprocessor.context
        >>> cdssm_preprocessor_test = CDSSMPreprocessor()
        >>> cdssm_preprocessor_test.context = context

    """

    def __init__(self):
        """Initialization."""
        self._datapack = None
        super().__init__()

    def _prepare_stateless_units(self) -> list:
        """Prepare needed process units."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit(),
            preprocessor.NgramLetterUnit()
        ]

    def fit(self, inputs: typing.List[tuple]):
        """
        Fit pre-processing context for transformation.

        :param inputs: Inputs to be preprocessed.
        :return: class:`DSSMPreprocessor` instance.
        """
        vocab = []
        units = self._prepare_stateless_units()

        logger.info("Start building vocabulary & fitting parameters.")

        # Convert user input into a datapack object.
        self._datapack = self.segmentation(inputs, stage='train')

        # Loop through user input to generate tri-letters.
        # 1. Used for build vocabulary of tri-letters (get dimension).
        # 2. Cached tri-letters can be further used to perform input
        #    transformation.
        content = self._datapack.content

        for key, val in tqdm(content.items()):
            # For each piece of text, apply process unit sequentially.
            text = val['text']
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)

        # Initialize a vocabulary process unit to build tri-letter vocab.
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)

        # Store the fitted parameters in context.
        self.context['term_index'] = vocab_unit.state['term_index']
        dim_triletter = len(vocab_unit.state['term_index']) + 1
        self.context['input_shapes'] = [(None, dim_triletter),
                                        (None, dim_triletter)]
        self._datapack.context = self.context
        return self

    def transform(
        self,
        inputs: typing.List[tuple],
        stage: str
    ) -> datapack.DataPack:
        """
        Apply transformation on data, create `tri-letter` representation.

        :param inputs: Inputs to be preprocessed.
        :param stage: Pre-processing stage, `train` or `test`.

        :return: Transformed data as :class:`DataPack` object.
        """
        pass
