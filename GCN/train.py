import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.utils import batch_iterator
from dataset_split import *
from graphplt import *

l2_regularization = 4e-4
total_classes = 2#Health and Patient

class GraphNet(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GraphConv(32, activation='elu', kernel_regularizer=l2(l2_regularization))
        self.conv2 = GraphConv(32, activation='elu', kernel_regularizer=l2(l2_regularization))
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(total_classes, activation='softmax')

    def call(self, inputs):
        x, fltr = inputs
        x = self.conv1([x, fltr])
        x = self.conv2([x, fltr])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output


def train_model():
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    test_loss = []
    test_acc = []
    learning_rate = 1e-3  # Learning rate for Adam
    batch_size = 32       # Batch size
    epochs = 1000      # Number of training epochs dsx 1000
    train_data, train_label, val_data, val_label, test_data, test_label, graph_created = load_data()
    train_data, val_data, test_data = train_data[..., None], val_data[..., None], test_data[..., None]
    # Create filter for GCN and convert to sparse tensor
    fltr = sp_matrix_to_sp_tensor(GraphConv.preprocess(graph_created))
    model = GraphNet()
    optimizer = Adam(lr=learning_rate)
    loss_fn = SparseCategoricalCrossentropy()
    accuracy_fn = SparseCategoricalAccuracy()
    # Training step
    @tf.function
    def train(x, y):
        with tf.GradientTape() as tape:
            predictions = model([x, fltr], training=True)
            loss = loss_fn(y, predictions)
            loss += sum(model.losses)
        acc = accuracy_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        model.summary()
        return loss, acc


    # Evaluation step
    @tf.function
    def evaluate(x, y):
        predictions = model([x, fltr], training=False)
        loss = loss_fn(y, predictions)
        loss += sum(model.losses)
        acc = accuracy_fn(y, predictions)
        return loss, acc


    curent_batch = 0
    batches_in_epoch = int(np.ceil(train_data.shape[0] / batch_size))
    batches_tr = batch_iterator([train_data, train_label], batch_size=batch_size, epochs=epochs)
    results_train = []
    for batch in batches_tr:
        curent_batch += 1
        loss_train, acc_train = train(*batch)
        results_train.append((loss_train, acc_train))
        if curent_batch == batches_in_epoch:
            batches_val = batch_iterator([val_data, val_label], batch_size=batch_size)
            results_val = [evaluate(*batch) for batch in batches_val]
            results_val = np.array(results_val)
            loss_val, acc_val = results_val.mean(0)
            batches_te = batch_iterator([test_data, test_label], batch_size=batch_size)
            results_test = [evaluate(*batch) for batch in batches_te]
            results_test = np.array(results_test)
            results_train = np.array(results_train)
            print('Train loss: {:.4f}, acc: {:.4f} | '
                  'Valid loss: {:.4f}, acc: {:.4f} | '
                  'Test loss: {:.4f}, acc: {:.4f}'
                  .format(*results_train.mean(0),
                          *results_val.mean(0),
                          *results_test.mean(0)))
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            val_loss.append(loss_val)
            val_acc.append(acc_val)
            loss_test, acc_test = results_test.mean(0)
            test_loss.append(loss_test)
            test_acc.append(acc_test)
            # Reset epoch
            results_train = []
            curent_batch = 0
    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


if __name__ == "__main__":
    train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = train_model()
    graphplt(train_loss, "Train_loss", "No. of epochs", "Loss During Training")
    graphplt(train_acc, "Train_Accuracy", "No. of epochs", "Accuracy During Training")
    graphplt(val_loss, "Validation_loss", "No. of epochs", "Loss During Validation")
    graphplt(val_acc, "Validation_Accuracy", "No. of epochs", "Accuracy During Validation")
    graphplt(test_loss, "Testing_loss", "No. of epochs", "Loss During Testing")
    graphplt(test_acc, "Testing_Accuracy", "No. of epochs", "Accuracy During Testing")