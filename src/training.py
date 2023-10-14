import tensorflow.keras as tfk

from typing import List

from .data_pipeline import DataGenerator

tfkc = tfk.callbacks


class Trainer:
    def train(self, model: tfk.Model, train_seq: DataGenerator, train_steps: int,
              val_seq: DataGenerator, val_steps: int, epochs: int) -> None:
        """Fits the model

        Args:
            model (tfk.Model): compiled model
            train_seq (DataGenerator): train generator
            train_steps (int): train steps
            val_seq (DataGenerator): val generator
            val_steps (int): val steps
            epochs (int): epochs
        """
        callbacks = self._get_callbacks()
        model.fit(train_seq, steps_per_epoch=train_steps, validation_data=val_seq,
                  validation_steps=val_steps, epochs=epochs, callbacks=callbacks)

    def _get_callbacks(self) -> List:
        """Gets callbacks

        Returns:
            List: Model checkpoints
        """
        callbacks = list()
        mc = tfkc.ModelCheckpoint(filepath='checkpoint_path', save_weights_only=False)
        callbacks.append(mc)
        return callbacks
