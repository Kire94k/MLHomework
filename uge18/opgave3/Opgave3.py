#Det nedenstående virker men accuracy er udregnet forkert da jeg burde gør det på nn modellen og ikke lave en ny
#men ved ikke hvordan man gør:(
#---------------------------------------------------------------------------------
#HUSK ændre stierne til csv data og image data med at bruge copy full path #
#------------------------------------------------------------------------------
import pandas as pd
import tensorflow as tf
from keras import layers, Model
from PIL import Image
import os

# Define relevant variables for the ML task
batch_size = 60
num_classes = 2  # Hotdog and not-hotdog
learning_rate = 0.001
num_epochs = 20


# Custom Dataset class to load images and labels from CSV
class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, csv_file, root_dir, batch_size):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch_data = self.data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        labels = []
        for _, row in batch_data.iterrows():
            img_name = row[0]
            img_path = os.path.join(self.root_dir, img_name)
            image = Image.open(img_path)
            image = image.resize((32, 32))  # Resize images to (32, 32)
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)
            labels.append(row[1])
        return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)

# Load the custom dataset from CSV
train_dataset = CustomDataset(csv_file=r"C:\Users\carin\Desktop\MLhomework\MLHomework\uge18\opgave3\hotdogtraining\CSV\Uge18HotdogData.csv",
                              root_dir=r"C:\Users\carin\Desktop\MLhomework\MLHomework\uge18\opgave3\hotdogtraining\Data",
                              batch_size=batch_size)


# Define the CNN model
class ConvNeuralNet(Model):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        # Define layers...
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# Create model instance
model = ConvNeuralNet(num_classes)

# Define loss function and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for step in range(len(train_dataset)):
        images, labels = train_dataset[step]
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, step+1, len(train_dataset), loss.numpy()))

# Evaluate the trained model
# You can write evaluation code here using a separate dataset.
# Load the train dataset for evaluation
train_dataset_for_evaluation = CustomDataset(csv_file=r"C:\Users\carin\Desktop\MLhomework\MLHomework\uge18\opgave3\hotdogtraining\CSV\Uge18HotdogData.csv",
                              root_dir=r"C:\Users\carin\Desktop\MLhomework\MLHomework\uge18\opgave3\hotdogtraining\Data",
                              batch_size=batch_size)

# Function to calculate accuracy
def calculate_accuracy(dataset):
    total_correct = 0
    total_samples = 0
    for step in range(len(dataset)):
        images, labels = dataset[step]
        predictions = model(images)
        predicted_labels = tf.argmax(predictions, axis=1)
        # Cast labels to int64 for compatibility with tf.equal()
        labels = tf.cast(labels, dtype=tf.int64)
        total_correct += tf.reduce_sum(tf.cast(tf.equal(predicted_labels, labels), dtype=tf.int32))
        total_samples += labels.shape[0]
    accuracy = total_correct / total_samples
    return accuracy.numpy()


# Evaluation loop
train_accuracy = calculate_accuracy(train_dataset_for_evaluation)
print(f"Train Accuracy: {train_accuracy}")
