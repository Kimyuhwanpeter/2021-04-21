# -*- coding:utf-8 -*-
from model_V14 import *
from random import shuffle
from collections import Counter


import numpy as np
import os
import datetime
import easydict

FLAGS = easydict.EasyDict({"img_height": 128,
                           
                           "img_width": 88,
                           
                           "tr_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/train.txt",
                           
                           "tr_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_train.txt",
                           
                           "tr_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "te_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/test.txt",
                           
                           "te_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_test_fix.txt",
                           
                           "te_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "batch_size": 137,
                           
                           "epochs": 300,
                           
                           "num_classes": 86,
                           
                           "lr": 0.001,
                           
                           "save_checkpoint": "",
                           
                           "graphs": "C:/Users/Yuhwan/Downloads/", 
                           
                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": ""})

optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

lab_list = np.arange(2, 88, dtype=np.int32)
lab_list_buf = np.zeros_like(lab_list, dtype=object)

def tr_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_png(img, 1)
    img = tf.image.resize(img, [FLAGS.img_height, FLAGS.img_width])
    #img = tf.image.convert_image_dtype(img, dtype=tf.float32) / 127.5 - 1.
    img = tf.image.per_image_standardization(img)

    #lab = lab_list - 2
    n_age = 10
    generation = 0.
    lab = 0
    if lab_list >= 2 and lab_list < 10:
        generation = 0
        generation = tf.one_hot(generation, 9)
        lab = n_age - (10 - lab_list)
    if lab_list >= 10 and lab_list < 20:
        generation = 1
        generation = tf.one_hot(generation, 9)
        lab = n_age - (20 - lab_list)
    if lab_list >= 20 and lab_list < 30:
        generation = 2
        generation = tf.one_hot(generation, 9)
        lab = n_age - (30 - lab_list)
    if lab_list >= 30 and lab_list < 40:
        generation = 3
        generation = tf.one_hot(generation, 9)
        lab = n_age - (40 - lab_list)
    if lab_list >= 40 and lab_list < 50:
        generation = 4
        generation = tf.one_hot(generation, 9)
        lab = n_age - (50 - lab_list)
    if lab_list >= 50 and lab_list < 60:
        generation = 5
        generation = tf.one_hot(generation, 9)
        lab = n_age - (60 - lab_list)
    if lab_list >= 60 and lab_list < 70:
        generation = 6
        generation = tf.one_hot(generation, 9)
        lab = n_age - (70 - lab_list)
    if lab_list >= 70 and lab_list < 80:
        generation = 7
        generation = tf.one_hot(generation, 9)
        lab = n_age - (80 - lab_list)
    if lab_list >= 80:
        generation = 8
        generation = tf.one_hot(generation, 9)
        lab = n_age - (90 - lab_list)



    return img, lab, generation, lab_list - 2

def te_func(img, lab):

    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, 1)
    img = tf.image.resize(img, [FLAGS.img_height, FLAGS.img_width])
    #img = tf.image.convert_image_dtype(img, dtype=tf.float32) / 127.5 - 1.
    img = tf.image.per_image_standardization(img)

    #lab = lab - 2

    return img, lab

def make_label_V2(ori_label):
    l = []
    for i in range(FLAGS.batch_size):
        label = [1] * (ori_label[i].numpy() + 1) + [0] * (10 - (ori_label[i].numpy() + 1))
        label = tf.cast(label, tf.float32)
        l.append(label)
    return tf.convert_to_tensor(l, tf.float32)

def make_label(ori_label):
    l = []
    for i in range(FLAGS.batch_size):
        label = [1] * (ori_label[i].numpy() + 1) + [0] * (FLAGS.num_classes - (ori_label[i].numpy() + 1))
        label = tf.cast(label, tf.float32)
        l.append(label)
    return tf.convert_to_tensor(l, tf.float32)

def run_model(model, images, training=True):
    return model(images, training=training)

def cal_CDF(logits, labels, num_feature):

    logit_CDF = []
    label_CDF = []
    for i in range(FLAGS.batch_size):
        log = 0.
        lab = 0.
        logit = logits[i]
        logit = logit.numpy()
        label = labels[i]
        label = label.numpy()
        log_buf = []
        lab_buf = []
        for j in range(num_feature):
            log += logit[j]
            lab += label[j]
            log_buf.append(log)
            lab_buf.append(lab)

        logit_CDF.append(log_buf)
        label_CDF.append(lab_buf)

    return logit_CDF, label_CDF

real_age = tf.range(2, 88, dtype=tf.float32)

def cal_loss(model, images, image2, age_labels, gener_labels):
    total_loss = 0.
    with tf.GradientTape() as tape:

        logits = run_model(model, [images, image2], True)

        max_feature = tf.reduce_max(logits, 2)  # [None, 9]

        age_generation_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(gener_labels, max_feature)

        age_feature = tf.reduce_mean(logits, 1) # [None, 10]

        logit_distribution = tf.nn.softmax(max_feature, 1)
        label_distribution = tf.nn.softmax(gener_labels, 1)

        logit_CDF, label_CDF = cal_CDF(logit_distribution, label_distribution, 9)
        logit_CDF, label_CDF = tf.convert_to_tensor(logit_CDF, dtype=tf.float32), tf.convert_to_tensor(label_CDF, dtype=tf.float32)

        CDF_loss = tf.keras.losses.MeanSquaredError()(label_CDF, logit_CDF)

        age_distribution = tf.nn.softmax(age_feature, 1)
        age_label_distribution = tf.nn.softmax(age_labels, 1)

        age_CDF, label_age_CDF = cal_CDF(age_distribution, age_label_distribution, 10)

        age_CDF_loss = tf.keras.losses.MeanSquaredError()(label_age_CDF, age_CDF)
        age_CDF_loss = tf.cast(age_CDF_loss, tf.float32)

        #age_feature = tf.nn.sigmoid(2*age_feature)
        
        #loss = -tf.reduce_sum(age_labels * tf.math.log(age_feature + 0.000001) + (1 - age_labels) * tf.math.log(1 - age_feature + 0.000001), 1)
        loss = (-tf.reduce_sum( (tf.math.log_sigmoid(age_feature)*age_labels + (tf.math.log_sigmoid(age_feature) - age_feature)*(1-age_labels)), 1))
        loss = tf.reduce_mean(loss)

        total_loss = age_generation_loss + loss + (CDF_loss + age_CDF_loss)*10 

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss

def cal_mae(model, images, images2, labels):

    logits = run_model(model, [images, images2], False)

    max_feature = tf.nn.softmax(tf.reduce_max(logits, 2), 1)    # [None, 9]

    predict_generation = tf.argmax(max_feature, 1, tf.int32)

    ae = 0
    predict_generation = predict_generation.numpy()

    age_feature = tf.reduce_mean(logits, 1) # [None, 10]

    for i in range(137):
        predict_generation_ = predict_generation[i]

        age_predict = tf.nn.sigmoid(age_feature[i])
        age_predict = tf.cast(tf.less(0.5, age_predict), tf.int32)
        age_predict = tf.reduce_sum(age_predict) - 1

        final_age = (predict_generation_ * 10) + age_predict

        ae += tf.abs(final_age - labels[i])

    return ae

def main():
    model = modifed_GEI_network()
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!!")

    if FLAGS.train:
        count = 0;

        tr_img = np.loadtxt(FLAGS.tr_txt_name, dtype="<U100", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + img + ".png"for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        C = Counter(tr_lab - 2)

        te_img = np.loadtxt(FLAGS.te_txt_name, dtype="<U100", skiprows=0, usecols=0)
        te_img = [FLAGS.te_img_path + img + ".png" for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.map(te_func)
        te_gener = te_gener.batch(137)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        #############################
        # Define the graphs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + "_Adam_V13" + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        val_log_dir = FLAGS.graphs + current_time + "_Adam_V13" + '/val'
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        #############################

        for epoch in range(FLAGS.epochs):
            TR = list(zip(tr_img, tr_lab))
            shuffle(TR)
            tr_img, tr_lab = zip(*TR)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab, dtype=np.int32)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(tr_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(tr_img) // FLAGS.batch_size

            for step in range(tr_idx):
                batch_images, batch_labels, batch_age_gener, real_age_label = next(tr_iter)
                batch_labels = make_label_V2(batch_labels)
                select_label = np.arange(0, FLAGS.batch_size, dtype=np.float32)
                select_label = np.zeros_like(select_label, dtype=object)

                if count == 0:
                    temp = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
                    temp = temp.shuffle(len(tr_img))
                    temp = temp.map(tr_func)
                    temp = temp.batch(137)
                    temp = temp.prefetch(tf.data.experimental.AUTOTUNE)

                    temp_iter = iter(temp)
                    temp_idx = len(tr_img) // 137

                    for k in range(temp_idx):
                        batch_images_, _, _, real_age_label_ = next(temp_iter)
                        for i in range(137):

                            img = batch_images_[i].numpy()
                            lab = real_age_label_[i].numpy()

                            for j in lab_list:
                                if j == (lab + 2):
                                    lab_list_buf[lab] += img / C[lab]

                        print("Saving image per age...{}".format(k))

                select_buf = []
                for i in range(FLAGS.batch_size):
                    lab = real_age_label[i].numpy()
                    #select_label[i] = lab_list_buf[lab]
                    select_buf.append(lab_list_buf[lab])
                select_buf = tf.convert_to_tensor(select_buf, dtype=tf.float32)

                loss = cal_loss(model, batch_images, select_buf, batch_labels, batch_age_gener)

                with train_summary_writer.as_default():
                    tf.summary.scalar('Loss', loss, step=count)

                if count % 10 == 0:
                    print("Epochs = {} [{}/{}] Loss = {}".format(epoch, step + 1, tr_idx, loss))

                if count % 100 == 0:
                    te_iter = iter(te_gener)
                    te_idx = len(te_img) // 137
                    ae = 0
                    for i in range(te_idx):
                        imgs, labs = next(te_iter)

                        ae += cal_mae(model, imgs, imgs, labs)

                    MAE = ae / len(te_img)
                    print("================================")
                    print("step = {}, MAE = {}".format(count, MAE))
                    print("================================")
                    with val_summary_writer.as_default():
                        tf.summary.scalar('MAE', MAE, step=count)

                    #num_ = int(count // 100)
                    #model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                    #if not os.path.isdir(model_dir):
                    #   os.makedirs(model_dir)
                    #   print("Make {} files to save checkpoint".format(num_))

                    #ckpt = tf.train.Checkpoint(model=model, optim=optim)
                    #ckpt_dir = model_dir + "/" + "New_age_estimation_{}.ckpt".format(count)
                    #ckpt.save(ckpt_dir)


                count += 1

if __name__ == "__main__":
    main()