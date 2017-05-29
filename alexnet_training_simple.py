import tensorflow as tf
import numpy as np
import sys
import os
import pickle
import time
import getopt
import cv2
from datetime import datetime

import alexnet_simple
from caffe_classes import class_names
from datagenerator import ImageDataGenerator



class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

def prune_weights(cRates, weights, weights_mask, biases, biases_mask, mask_dir, f_name):
    keys = ['cov1','cov2','fc1','fc2','fc3']
    new_mask = {}
    for key in keys:
        w_eval = weights[key].eval()
        threshold_off = 0.9*(np.mean(w_eval) + cRates[key] * np.std(w_eval))
        threshold_on = 1.1*(np.mean(w_eval) + cRates[key] * np.std(w_eval))
        mask_off = np.abs(w_eval) < threshold_off
        mask_on = np.abs(w_eval) > threshold_on
        new_mask[key] = np.logical_or(((1 - mask_off) * weights_mask[key]),mask_on).astype(int)
    with open(mask_dir + f_name, 'wb') as f:
        pickle.dump((new_mask,biases_mask), f)



def prune_info(weights, counting):
    if (counting == 0):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'].eval())
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
        # print(weights['cov1'].eval())
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'].eval())
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'].eval())
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc3'].eval())
        print('fc3 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
    if (counting == 1):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'])
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
        # print(weights['cov1'].eval())
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'])
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'])
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'])
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc3'])
        print('fc3 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
def plot_weights(weights,pruning_info):
        keys = ['cov1','cov2','fc1', 'fc2','fc2']
        fig, axrr = plt.subplots( 2, 2)  # create figure &  axis
        fig_pos = [(0,0), (0,1), (1,0), (1,1)]
        index = 0
        for key in keys:
            weight = weights[key].eval().flatten()
            # print (weight)
            size_weight = len(weight)
            weight = weight.reshape(-1,size_weight)[:,0:size_weight]
            x_pos, y_pos = fig_pos[index]
            #take out zeros
            weight = weight[weight != 0]
            # print (weight)
            hist,bins = np.histogram(weight, bins=100)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            axrr[x_pos, y_pos].bar(center, hist, align = 'center', width = width)
            axrr[x_pos, y_pos].set_title(key)
            index = index + 1
        fig.savefig('fig_v3/weights'+pruning_info)
        plt.close(fig)

def save_pkl_model(weights, biases, save_dir ,f_name):
    keys = ['cov1','cov2','fc1','fc2','fc3']
    weights_val = {}
    biases_val = {}
    for key in keys:
        weights_val[key] = weights[key].eval()
        biases_val[key] = biases[key].eval()
    with open(save_dir + f_name, 'wb') as f:
        print('Created a pickle file')
        pickle.dump((weights_val, biases_val), f)

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def compute_file_name(thresholds):
    keys_cov = ['cov1', 'cov2']
    keys_fc = ['fc1', 'fc2', 'fc3']
    name = ''
    for key in keys_cov:
        name += 'cov'+ str(int(thresholds[key]*10))
    for key in keys_fc:
        if (key == 'fc1'):
            name += 'fc'+ str(int(thresholds[key]*100))
        else:
            name += 'fc'+ str(int(thresholds[key]*10))
    return name


def main(argv = None):
    if (argv is None):
        argv = sys.argv
    try:
        try:
            opts = argv
            first_time_load = True
            parent_dir = './'
            WITH_BIASES = False
            save_for_next_iter = False
            TEST = False

            for item in opts:
                print (item)
                opt = item[0]
                val = item[1]
                if (opt == '-cRates'):
                    cRates = val
                if (opt == '-first_time'):
                    first_time_load = val
                if (opt == '-file_name'):
                    file_name = val
                if (opt == '-train'):
                    TRAIN = val
                if (opt == '-prune'):
                    PRUNE = val
                if (opt == '-test'):
                    TEST = val
                if (opt == '-parent_dir'):
                    parent_dir = val
                if (opt == '-lr'):
                    lr = val
                if (opt == '-with_biases'):
                    WITH_BIASES = val
                if (opt == '-lambda1'):
                    lambda_1 = val
                if (opt == '-lambda2'):
                    lambda_2 = val
                if (opt == '-save'):
                    SAVE = val
                if (opt == '-org_file_name'):
                    org_file_name = val
        except getopt.error, msg:
            raise Usage(msg)

        epochs = 5
        dropout = 0.5
        batch_size = 128
        num_classes = 1000

        mask_dir = parent_dir
        weights_dir = parent_dir
        LOCAL_TEST = 0

        # compute the file name
        file_name_part = compute_file_name(cRates)

        # load masks
        (weights_mask, biases_mask)= alexnet_simple.initialize_weights_mask(first_time_load, mask_dir, 'mask'+org_file_name + '.pkl')


        if (LOCAL_TEST):
            index_file_dir = 'cpu_test_data/'
        else:
            meta_data_dir = '/local/scratch/share/ImageNet/ILSVRC/Data/CLS-LOC/'
            index_file_dir = '/local/scratch/share/ImageNet/ILSVRC/ImageSets/CLS-LOC/'

        if (TRAIN):
            train_file_txt = index_file_dir + 'train.txt'
            val_file_txt = index_file_dir + 'val.txt'
            test_file_txt = index_file_dir + 'test.txt'
        else:
            test_file_txt = index_file_dir + 'test.txt'
            test_file_txt = index_file_dir + 'val.txt'

        # if (first_time_load):
        #     PREV_MODEL_EXIST = 0
        #     weights, biases = initialize_variables(PREV_MODEL_EXIST, '')
        # else:
        #     PREV_MODEL_EXIST = 1
        #     if (save_for_next_iter):
        #         file_name_part = org_file_name
        #     else:
        #         file_name_part = compute_file_name(cRates)
        #     weights, biases = initialize_variables( PREV_MODEL_EXIST,
        #                                             weights_dir + 'weights' + file_name_part + '.pkl')
        #
        batch_size = 128
        if (TRAIN):
            weights, biases = alexnet_simple.initialize_variables(new_model = TRAIN)
        elif(TEST):
            weights, biases = alexnet_simple.initialize_variables(new_model = TRAIN, weights_path='base.npy')
        else:
            weights, biases = alexnet_simple.initialize_variables(new_model = TRAIN)


        x = tf.placeholder(tf.float32, [None, 227, 227, 3])
        y = tf.placeholder(tf.float32, [None, num_classes])
        keep_prob = tf.placeholder(tf.float32)

        # initilize the model from the class constructer
        # model = AlexNet(x, keep_prob, num_classes, weights_mask, new_model = first_time_load)


        score, test_conv = alexnet_simple.conv_network(x, weights, biases, keep_prob, batch_size = batch_size)
        print('score shape is {}'.format(score.get_shape()))
        softmax = tf.nn.softmax(score)

        var_list = tf.trainable_variables()
        var_name_list = [v.name for v in var_list]

        with tf.name_scope("cross_ent"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))

        with tf.name_scope("train"):
            # l1_norm = lambda_1 * l1
            # l2_norm = lambda_2 * l2
            # regulization_loss = l1_norm + l2_norm

            # opt = tf.train.AdamOptimizer(lr)
            # grads = opt.compute_gradients(loss)
            # org_grads = [(ClipIfNotNone(grad), var) for grad, var in grads]
            # train_step = opt.apply_gradients(org_grads)
            print('check var list :{}'.format(var_name_list))
            opt = tf.train.AdamOptimizer(learning_rate=lr)
            gradients = opt.compute_gradients(loss)
            clipped_grads = [(ClipIfNotNone(grad), var) for grad, var in gradients]
            train_step = opt.apply_gradients(clipped_grads)


        with tf.name_scope("accuracy"):
            check = tf.argmax(score,1)
            correct_prediction = tf.equal(tf.argmax(score,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        init = tf.global_variables_initializer()
        accuracy_list = np.zeros(20)
        train_acc_list = []

        # Launch the graph
        print('Graph launching ..')

        if (TRAIN):
            train_generator = ImageDataGenerator(train_file_txt,
                                                 horizontal_flip = False, shuffle = True)
            val_generator = ImageDataGenerator(val_file_txt)

            # Get the number of training/validation steps per epoch
            train_batches_per_epoch = train_generator.data_size / batch_size
            val_batches_per_epoch = val_generator.data_size / batch_size

        if (TEST):
            test_generator = ImageDataGenerator(test_file_txt)
            test_batches_per_epoch = test_generator.data_size / batch_size
            print('data size is {}'.format(test_generator.data_size))
            print('Number of test batches per epoch is {}'.format(test_batches_per_epoch))


        with tf.Session() as sess:
            sess.run(init)
            epoch_acc = []
            epoch_entropy = []

            keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
            weights_init= {}
            for key in keys:
                weights_init[key] = weights[key].eval()

            if (TRAIN):
                print("{} Start training...".format(datetime.now()))
                for i in range(0,epochs):
                    print("{} Epoch number: {}".format(datetime.now(), i+1))
                    for step in range(train_batches_per_epoch):
                        # print("batch size {}".format(batch_size))
                        (batch_x, batch_y) = train_generator.next_batch(batch_size, meta_data_dir+'train/')
                        # print("size of x {}".format(np.shape(batch_x)))
                        # print("size of y {}".format(np.shape(batch_y)))
                        test_val = sess.run([check], feed_dict = {
                                        x: batch_x,
                                        y: batch_y,
                                        keep_prob: 1.0})
                        # print("size of score {}".format(np.shape(score_val)))
                        # print("pred {}".format(test_val))
                        # print("actual {}".format(np.argmax(batch_y,1)))

                        train_acc, cross_en = sess.run([accuracy, loss], feed_dict = {
                                        x: batch_x,
                                        y: batch_y,
                                        keep_prob: 1.0})

                        if (step % DISPLAY_FREQ == 0):
                            # print('check weights update')
                            # for key in keys:
                            #     w_val = weights[key].eval()
                            #     print("{}: are they equal {}".format(key, weights_init[key] == w_val))
                            #     print("check some info")
                            #     print("mean value is {}, max value is {}, min value is {}".format(np.mean(w_val),np.max(w_val), np.min(w_val)))
                            # if (PRUNE):
                            #     print('This is the {}th of {}pruning, time is {}'.format(
                            #         i,
                            #         cRates,
                            #         datetime.now()
                            #     ))
                            # print("accuracy is {} and cross entropy is {}".format(
                            #     train_acc,
                            #     cross_en
                            # ))
                            accuracy_list = np.concatenate((np.array([train_acc]),accuracy_list[0:19]))
                            epoch_acc.append(train_acc)
                            epoch_entropy.append(cross_en)
                            if (step%(DISPLAY_FREQ*50) == 0 and step != 0):
                                train_acc_list.append(train_acc)
                                alexnet_simple.save_weights(weights, biases)
                                # file_name_part = compute_file_name(cRates)
                                # save_pkl_model(weights, biases, weights_dir, 'weights' + file_name_part + '.pkl')
                                # print("saved the network")
                                with open ('acc_hist.txt', 'wb') as f:
                                    for item in epoch_acc:
                                        f.write("{}\n".format(item))
                                with open ('entropy_hist.txt', 'wb') as f:
                                    for item in epoch_entropy:
                                        f.write("{}\n".format(item))
                            # if (np.mean(accuracy_list) > 0.8):
                            #     accuracy_list = np.zeros(20)
                            #     test_acc = sess.run(accuracy, feed_dict = {
                            #                             x: images_test,
                            #                             y: labels_test,
                            #                             keep_prob: 1.0})
                            #     print('test accuracy is {}'.format(test_acc))
                            #     if (test_acc > 0.823):
                            #         print("training accuracy is large, show the list: {}".format(accuracy_list))
                        _ = sess.run(train_step, feed_dict = {
                                        x: batch_x,
                                        y: batch_y,
                                        keep_prob: dropout})
                    test_acc_list = []
                    for _ in range(val_batches_per_epoch):
                        # Taverse one epoch
                        (batch_tx, batch_ty) = val_generator.next_batch(batch_size, meta_data_dir + 'val/')
                        tmp_acc, c_pred, c_softmax = sess.run([accuracy, score, softmax], feed_dict = {
                            x: batch_tx,
                            y: batch_ty,
                            keep_prob: 1.0})
                        test_acc_list.append(tmp_acc)
                    test_acc_list = np.array(test_acc_list)
                    test_acc = np.mean(test_acc_list)
                    print("Time {}, Validation Accuracy = {}".format(datetime.now(), test_acc))
                    # reset both pointers
                    val_generator.reset_pointer()
                    train_generator.reset_pointer()

            if (TEST):
                test_acc_list = []
                if (LOCAL_TEST):
                    image_dir = "cpu_test_data/tmp_images/"
                    imagenet_mean = np.array([104., 117., 124.], dtype = np.float32)
                    test_batches_per_epoch = 3
                    img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]
                    imgs_test = []
                    for i,f in enumerate(img_files):
                        tmp = cv2.imread(f)
                        tmp = cv2.resize(tmp.astype(np.float32), (227,227))
                        tmp -= imagenet_mean[i]
                        tmp = tmp.reshape((1,227,227,3))
                        imgs_test.append(tmp)
                    names = []
                    probs = []
                    for step in range(test_batches_per_epoch):
                        prob = sess.run(softmax, feed_dict = {
                                                x: imgs_test[step],
                                                # y: labels_test,
                                                keep_prob: 1.0})
                        name = class_names[np.argmax(prob)]
                        probs.append(np.max(prob))
                        names.append(name)
                    print("names are {}".format(names))
                    print("probs are {}".format(probs))
                    sys.exit()
                else:
                    test_acc_list = []
                    # Taverse one epoch
                    for step in range(test_batches_per_epoch):
                        (batch_x, batch_y) = test_generator.next_batch(batch_size, meta_data_dir + 'val/')
                        tmp_acc, c_pred, c_softmax = sess.run([accuracy, score, softmax], feed_dict = {
                            x: batch_x,
                            y: batch_y,
                            keep_prob: 1.0})
                        test_acc_list.append(tmp_acc)
                    test_acc_list = np.array(test_acc_list)
                    test_acc = np.mean(test_acc_list)
                    print("test accuracy of AlexNet is {}".format(test_acc))
                    sys.exit()


            # if (save_for_next_iter):
            #     print('saving for the next iteration of dynamic surgery')
            #     file_name_part = compute_file_name(cRates)
            #     file_name = 'weights'+ file_name_part+'.pkl'
            #     save_pkl_model(weights, biases, parent_dir, file_name)
            #
            #     file_name_part = compute_file_name(cRates)
            #     with open(parent_dir + 'mask' + file_name_part + '.pkl','wb') as f:
            #         pickle.dump((weights_mask, biases_mask),f)
            # if (TRAIN):
            #     file_name_part = compute_file_name(cRates)
            #     save_pkl_model(weights, biases, weights_dir, 'weights' + file_name_part + '.pkl')
            #     with open(parent_dir + 'training_data'+file_name_part+'.pkl', 'wb') as f:
            #         pickle.dump(train_acc_list, f)
            #
            # if (PRUNE):
            #     print('saving pruned model ...')
            #     f_name = compute_file_name(cRates)
            #     prune_weights(  cRates,
            #                     weights,
            #                     weights_mask,
            #                     biases,
            #                     biases_mask,
            #                     mask_dir,
            #                     'mask' + f_name + '.pkl')
            #     file_name_part = compute_file_name(cRates)
            #     save_pkl_model(weights, biases, weights_dir, 'weights' + file_name_part + '.pkl')
            # return test_acc
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return
if __name__ == '__main__':
    sys.exit(main())
