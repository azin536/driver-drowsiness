import tensorflow.keras as tfk

tfkl = tfk.layers
tfkm = tfk.models


class ModelBuilder:
    def _build_model(self) -> tfk.Model:
        """Builds the architecture of the model.

        Returns:
            tfk.Model: built model
        """
        input_layer = tfkl.Input(shape=(24, 24, 1))
        x = tfkl.Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = tfkl.MaxPooling2D(pool_size=(1,1))(x)
        x = tfkl.Conv2D(32, (3, 3), activation='relu')(x)
        x = tfkl.MaxPooling2D(pool_size=(1,1))(x)
        x = tfkl.Conv2D(64, (3, 3), activation='relu')(x)
        x = tfkl.MaxPooling2D(pool_size=(1,1))(x)
        x = tfkl.Dropout(0.25)(x)
        x = tfkl.Flatten()(x)
        x = tfkl.Dense(128, activation='relu')(x)
        x = tfkl.Dropout(0.5)(x)
        output_layer = tfkl.Dense(4, activation='softmax')(x)
        model = tfkm.Model(inputs=input_layer, outputs=output_layer)
        return model

    def get_compiled_model(self):
        model = self._build_model()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])