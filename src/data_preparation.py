import os

import numpy as np

from pathlib import Path
from tqdm import tqdm
from typing import List, Union


class DataPreparator:
    def __init__(self, dirs: List, class_numbers: List) -> None:
        self.dirs = dirs
        self.class_numbers = class_numbers

    def _extract_paths_and_labels(self) -> Union[List, List]:
        """Gets paths and corresponding labels.

        Returns:
            Union[List, List]: paths, labels
        """
        paths = list()
        labels = list()
        for dir, class_num in zip(self.dirs, self.class_numbers):
            for path in tqdm(os.listdir(dir)):
                paths.append(str(Path(dir).joinpath(path)))
                labels.append(class_num)
        return paths, labels

    def _shuffle_extracted_data(self) -> List:
        """Shuffles the paths and labels

        Returns:
            List: shuffled lists
        """
        paths, labels = self._extract_paths_and_labels()
        joined_lists = list(zip(paths, labels))
        shuffled_lists = np.random.permutation(joined_lists)
        return shuffled_lists

    def get_training_data(self):
        shuffled_data = self._shuffle_extracted_data()
        val_paths = list()
        val_labels = list()
        train_paths = list()
        train_labels = list()

        for i, (path, label) in enumerate(shuffled_data):
            if i < 240:
                val_paths.append(path)
                val_labels.append(label)
            else:
                train_paths.append(path)
                train_labels.append(label)
        return train_paths, train_labels, val_paths, val_labels
