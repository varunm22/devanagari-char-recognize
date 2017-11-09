import math
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam

def get_test_generator(batch_size):
	test_datagen = ImageDataGenerator(
        rescale=1./255,
    )

	test_generator = test_datagen.flow_from_directory(
    	'Data/Test',
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size
    )

	return test_generator


if __name__ == "__main__":
	batch_size = 32
	model = load_model('models/alexnet_1.h5')

	optimizer = Adam(lr=1e-4)

	model.compile(loss='categorical_crossentropy',
	              optimizer=optimizer,
	              metrics=['accuracy', 'top_k_categorical_accuracy'])

	test_generator = get_test_generator(batch_size)

	scores = model.evaluate_generator(
		test_generator,
		steps=int(math.ceil(float(test_generator.samples) / batch_size)),
	)

	print("Accuracy: ", scores[1])
	print("Top 5 Accuracy: ", scores[2])

