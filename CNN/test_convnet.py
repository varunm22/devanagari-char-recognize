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
        batch_size=batch_size,
        shuffle=False
    )

	return test_generator


if __name__ == "__main__":
	batch_size = 32
	model_1 = load_model('models/resnet_34_1.h5')
	# model_2 = load_model('models/resnet_34_3.h5')
	models = [model_1]

	# scores = model.evaluate_generator(
	# 	test_generator,
	# 	steps=int(math.ceil(float(test_generator.samples) / batch_size)),
	# )
	predictions_list = []
	for model in models:
		test_generator = get_test_generator(batch_size)
		predictions = model.predict_generator(
			test_generator,
			steps=int(math.ceil(float(test_generator.samples) / batch_size)),
		)
		files = test_generator.filenames
		order = sorted(range(len(files)), key=files.__getitem__)
        predictions_list.append(predictions[order, :])

	
	actual_labels = np.array(test_generator.classes)
	predicted_labels = np.argmax(sum(predictions_list), axis=1)
	print(predicted_labels[1000:1100])
	num_correct = np.sum(actual_labels == predicted_labels)

	print("Number correct: ", num_correct)
	print("Accuracy: ", float(num_correct)/actual_labels.shape[0])
	# print("Top 5 Accuracy: ", scores[2])

