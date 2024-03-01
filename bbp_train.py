# Import libraries
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import Normalizer
# ignore the warning
import warnings
warnings.filterwarnings("ignore")

# Import Tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# Import self-defined functions
from utils.metrics import AverageMeter
from utils.metrics import AUC
from utils.loss import rank_loss
from model.model_ranking import AutoInt_Vanilla, AutoInt_JRC
from dataloader_ranking import get_bbp_set, get_pointwise_set
# Hyperparameters
batch_size = 256
base_lr = 1e-3 * batch_size / 256
weight_decay = 1e-5
num_epoch = 10
loss_lambda = 0.5 

# dataset and dataloader
# training, validation, testing dataset
file_path = '/data/QiweiAmazon/cloth/' # cloth, music, electro
print('Loading data from {}'.format(file_path))

idlabel_cols = ['userid','itemid','label']
text_cols = ['text_col{}'.format(i) for i in range(128)]
user_cols = ['user_col{}'.format(i) for i in range(128)]
item_cols = ['item_col{}'.format(i) for i in range(128)]
img_cols = ['img_col{}'.format(i) for i in range(128)]

columns = idlabel_cols + text_cols + user_cols + item_cols + img_cols

train_data = pd.read_csv(os.path.join(file_path, 'train_qw.csv'), header=0)
val_data = pd.read_csv(os.path.join(file_path, 'val.csv'), names=columns)
test_data = pd.read_csv(os.path.join(file_path, 'test.csv'), names=columns)

user_id_list, item_id_list, label_list, score_list, feature_matrix = get_bbp_set(train_data)
val_user_id_list, val_item_id_list, val_label_list, val_feature_matrix = get_pointwise_set(val_data)
test_user_id_list, test_item_id_list, test_label_list, test_feature_matrix = get_pointwise_set(test_data)

print("preprocessing data...")
normalizer = Normalizer()
normalizer.fit(feature_matrix)
feature_matrix = normalizer.transform(feature_matrix)
val_feature_matrix = normalizer.transform(val_feature_matrix)
test_feature_matrix = normalizer.transform(test_feature_matrix)
print("done")

trainset = tf.data.Dataset.from_tensor_slices((user_id_list, item_id_list, label_list, score_list, feature_matrix))
valset = tf.data.Dataset.from_tensor_slices((val_user_id_list, val_item_id_list, val_label_list, val_feature_matrix))
testset = tf.data.Dataset.from_tensor_slices((test_user_id_list, test_item_id_list, test_label_list, test_feature_matrix))

trainset = trainset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
valset = valset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
testset = testset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# model and optimizer
model = AutoInt_Vanilla(val_feature_matrix.shape[1], embed_dim=8, n_heads=2, head_dim=4,
                        attention_num=3, hidden_units=[512,256,64])

optimizer = Adam(learning_rate=base_lr)



# training and validation
loss_meter = AverageMeter()
acc_meter = AverageMeter()
auc_meter = AverageMeter()
val_loss_meter = AverageMeter()
val_acc_meter = AverageMeter()
val_auc_meter = AverageMeter()
test_loss_meter = AverageMeter()
test_acc_meter = AverageMeter()
test_auc_meter = AverageMeter()
best_val_auc_meter = 0.0
iter = 0
for epoch in range(num_epoch):


    print("Epoch: {}, lr: {}".format(epoch, optimizer.lr.numpy()))
    for (user_id_list, item_id_list, label_list, score_list, feature_matrix) in tqdm(trainset):
        iter += 1
        with tf.GradientTape() as tape:
            pred = model(feature_matrix)
            pred = tf.nn.sigmoid(pred) + 1e-9
            loss = tf.reduce_mean(rank_loss(pred, score_list)) * (1 - loss_lambda) + tf.keras.losses.BinaryCrossentropy()(label_list, pred) * loss_lambda

        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(grad, 5) for grad in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_meter.update(loss.numpy())
        
        # add a sigmoid to the prediction
        acc_meter.update(accuracy_score(label_list, np.round(pred)))
        auc_meter.update(AUC(label_list, pred))

        if iter % 100 == 0:
            tqdm.write("Epoch: {}, loss: {:.4f}, acc: {:.4f}, auc: {:.4f}".format(epoch, loss_meter.avg, acc_meter.avg, auc_meter.avg))
            # validation
            val_loss_meter.reset()
            val_acc_meter.reset()
            val_auc_meter.reset()            
            for (user_id_list, item_id_list, label_list, feature_matrix) in valset:
                pred = model(feature_matrix)
                pred = tf.nn.sigmoid(pred)
                loss = tf.keras.losses.BinaryCrossentropy()(label_list, pred)
                val_loss_meter.update(loss.numpy())
                val_acc_meter.update(accuracy_score(label_list.numpy(), np.round(pred.numpy())))
                val_auc_meter.update(AUC(label_list.numpy(), pred.numpy()))
            print("Epoch: {}, val_loss: {:.4f}, val_acc: {:.4f}, val_auc: {:.4f}".format(epoch, val_loss_meter.avg, val_acc_meter.avg, val_auc_meter.avg))
            if val_auc_meter.avg > best_val_auc_meter:
                best_epoch = epoch
                # test
                test_loss_meter.reset()
                test_auc_meter.reset()
                for (user_id_list, item_id_list, label_list, feature_matrix) in testset:
                    pred = model(feature_matrix)
                    pred = tf.nn.sigmoid(pred)
                    logloss = log_loss(label_list.numpy(), pred.numpy())                
                    test_loss_meter.update(logloss)
                    test_auc_meter.update(AUC(label_list, pred))
                best_test_auc_meter = test_auc_meter.avg
                best_test_loss_meter = test_loss_meter.avg
                print("Best at: {}, LogLoss: {:.4f}, AUC: {:.4f}".format(best_epoch, best_test_loss_meter, best_test_auc_meter))
            tqdm.write("Val Epoch: {}, LogLoss: {:.4f}, AUC: {:.4f}".format(epoch, val_loss_meter.avg, val_auc_meter.avg))