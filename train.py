from src.data_pipeline import DataGenerator
from src.data_preparation import DataPreparator
from src.model_building import ModelBuilder
from src.training import Trainer


def main():
    dirs = [
                '/kaggle/input/yawn-eye-dataset-new/dataset_new/train/Closed',
                '/kaggle/input/yawn-eye-dataset-new/dataset_new/train/Open',
                '/kaggle/input/yawn-eye-dataset-new/dataset_new/train/yawn',
                '/kaggle/input/yawn-eye-dataset-new/dataset_new/train/no_yawn'
           ]
    class_numbers = [0, 1, 2, 3]
    preparator = DataPreparator(dirs, class_numbers)
    train_paths, train_labels, val_paths, val_labels = preparator.get_training_data()
    train_steps = int(len(train_paths) // 32)
    val_steps = int(len(val_paths) // 32)
    train_seq = DataGenerator(train_paths, train_labels, train_steps)
    val_seq = DataGenerator(val_paths, val_labels, val_steps)
    model_builder = ModelBuilder()
    model = model_builder.get_compiled_model()
    trainer = Trainer()
    trainer.train(model, train_seq, train_steps, val_seq, val_steps, 20)
