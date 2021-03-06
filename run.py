import os
import alexnet_training_simple
import alexnet_training
import sys
import download

def compute_file_name(p):
    name = ''
    name += 'cov' + str(int(p['cov1'] * 10))
    name += 'cov' + str(int(p['cov2'] * 10))
    name += 'fc' + str(int(p['fc1'] * 100))
    name += 'fc' + str(int(p['fc2'] * 10))
    name += 'fc' + str(int(p['fc3'] * 10))
    return name

acc_list = []
count = 0
retrain = 0
parent_dir = 'assets/'
# lr = 1e-5
lr = 1e-4
crates = {
    'cov1': 0.,
    'cov2': 0.,
    'fc1': 0.,
    'fc2': 0.,
    'fc3': 0.
}

retrain_cnt = 0
roundrobin = 0
with_biases = False

# check npy files
url = 'http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy'
download_dir = '.'
download.maybe_download_and_extract(url, download_dir)
INITIAL_TRAIN = True
if (INITIAL_TRAIN):
    # TRAIN
    param = [
        ('-cRates', crates),
        ('-first_time', True),
        # ('-first_time', False),
        ('-train', False),
        ('-test', True),
        ('-prune', False),
        # ('-test', True),
        ('-lr', lr),
        ('-with_biases', with_biases),
        ('-parent_dir', parent_dir),
        ('-lambda1', 1e-5),
        ('-lambda2', 1e-5)
        ]
    _ = alexnet_training_simple.main(param)
    # _ = alexnet_training.main(param)
    sys.exit()

#
# # Prune
# while (crates['fc1'] < 2):
#     count = 0
#     iter_cnt = 0
#     while (iter_cnt < 7):
#         if (iter_cnt > 3 and iter_cnt < 5):
#             lr = 5e-5
#         elif (iter_cnt >= 5):
#             lr = 1e-5
#         else:
#             lr = 1e-4
#         param = [
#             ('-cRates', crates),
#             ('-first_time', False),
#             ('-train', False),
#             ('-prune', True),
#             ('-lr', lr),
#             ('-with_biases', with_biases),
#             ('-parent_dir', parent_dir),
#             ('-lambda1', 1e-5),
#             ('-lambda2', 1e-5)
#             ]
#         # _ = train.main(param)
#
#         # TRAIN
#         param = [
#             ('-cRates', crates),
#             ('-first_time', False),
#             ('-train', True),
#             ('-prune', False),
#             ('-lr', lr),
#             ('-with_biases', with_biases),
#             ('-parent_dir', parent_dir),
#             ('-lambda1', 1e-5),
#             ('-lambda2', 1e-5)
#             ]
#         _ = alexnet_training.main(param)
#
#         # TEST
#         param = [
#             ('-cRates', crates),
#             ('-first_time', False),
#             ('-train', False),
#             ('-prune', False),
#             ('-test', True),
#             ('-lr', lr),
#             ('-with_biases', with_biases),
#             ('-parent_dir', parent_dir),
#             ('-lambda1', 1e-5),
#             ('-lambda2', 1e-5)
#             ]
#         acc = alexnet_training.main(param)
#
#
#         if (acc > 0.823 or iter_cnt == 7):
#             file_name = compute_file_name(crates)
#             crates['fc1'] = crates['fc1'] + 0.05
#             # crates['cov2'] = crates['cov2'] + 0.2
#             # crates['fc2'] = crates['fc2'] + 0.5
#             # crates['fc3'] = crates['fc3'] + 0.1
#             # crates['cov2'] = crates['cov2'] + 0.5
#             # crates['cov1'] = crates['cov1'] + 0.2
#             acc_list.append((crates,acc))
#             param = [
#                 ('-first_time', False),
#                 ('-train', False),
#                 ('-prune', False),
#                 ('-lr', lr),
#                 ('-with_biases', with_biases),
#                 ('-parent_dir', parent_dir),
#                 ('-iter_cnt',iter_cnt),
#                 ('-cRates',crates),
#                 ('-save', True),
#                 ('-lambda1', 1e-5),
#                 ('-lambda2', 1e-5),
#                 ('-org_file_name', file_name)
#                 ]
#             _ = train.main(param)
#             break
#         else:
#             iter_cnt = iter_cnt + 1
#     print('accuracy summary: {}'.format(acc_list))
#
#
# print('accuracy summary: {}'.format(acc_list))
