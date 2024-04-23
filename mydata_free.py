# detecting malicious samples that triggers backdoor via:
# optimize on the inner embedding (between Conv and FCs) & observe behaviors of the middle-layer neurons

from __future__ import print_function
import os
import argparse
import ast
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from vgg_face import VGG_16
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from networks import partial_models_adaptive
from networks.resnet import ResNet
from networks.vgg import VGG16
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

from networks.BadEncoderOriginalModels.simclr_model import SimCLR, SimCLRBase
from networks.BadEncoderOriginalModels.nn_classifier import NeuralNet

from networks.networks_partial_models import ResNet18LaterPart, \
    VGG16LaterPart, VGG16SingleFCLaterPart, VGG16DropoutLaterPart, VGGNetBinaryLaterPart
from networks.BadEncoderOriginalModels import bad_encoder_full_model_partial


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--gpu_index', type=str, default='4')

    # model dataset
    parser.add_argument('--model', type=str, default='simple_cnn',
                        choices=['resnet50', 'resnet18',
                                 'vgg16',
                                 'google_net',
                                 'simple_cnn',
                                 'bad_encoder_full_model'
                                 ])
    parser.add_argument('--n_cls', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--size', type=int, default=28,
                        help='size of the input image')
    parser.add_argument('--inspect_layer_position', type=int, default=None,  # default=2
                        help='which part as the partial model')

    # model to be detected
    parser.add_argument('--ckpt', type=str,
                        default=
                        f'/home/zq/projects/FreeEagle/backdoor_attack_simulation/saved_models/poisoned_mnist_models/'
                        f'poisoned_mnist_simple_cnn_class-agnostic_targeted=9_patched_img-trigger/last.pth',
                        help='path to pre-trained model')

    parser.add_argument('--num_important_neurons', type=int, default=5)
    parser.add_argument('--num_dummy', type=int, default=1)
    parser.add_argument('--metric', type=str, default='softmax_score', choices=['logit', 'softmax_score'])

    parser.add_argument('--use_transpose_correction', type=ast.literal_eval, default=False,
                        help='mul the correction factor -- (a,b)/(b,a) if (a,b) is larger')

    opt = parser.parse_args()
    set_default_settings(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.gpu_index}'
    print(f'Running on GPU:{opt.gpu_index}')

    return opt


def set_default_settings(opt):
    opt.num_dummy = 1
    # set opt.in_dims according to the size of the input image
    if opt.size == 32:
        opt.in_dims = 512
    elif opt.size == 64:
        opt.in_dims = 2048
    elif opt.size == 224:
        opt.in_dims = 2048
    elif opt.size == 28:
        pass
    else:
        raise ValueError

    # set default inspected layer position
    if opt.inspect_layer_position is None:
        if 'resnet' in opt.model:
            opt.inspect_layer_position = 2
        elif 'face' in opt.model:
            opt.inspect_layer_position = 5
        elif 'vgg' in opt.model:
            opt.inspect_layer_position = 5
        elif 'google' in opt.model:
            opt.inspect_layer_position = 4
        elif 'simple_cnn' in opt.model:
            opt.inspect_layer_position = 1
        else:
            raise ValueError('Unexpected model arch.')

    # set opt.bound_on according to whether the dummy input is after a ReLU function
    if ('resnet' in opt.model and opt.inspect_layer_position >= 1) \
            or ('vgg16' in opt.model and opt.inspect_layer_position >= 2) \
            or ('google' in opt.model and opt.inspect_layer_position >= 1)\
            or ('cnn' in opt.model and opt.inspect_layer_position >= 1)\
            or ('face' in opt.model and opt.inspect_layer_position >= 1):
        opt.bound_on = True
    else:
        opt.bound_on = False
    print(f'opt.bound_on:{opt.bound_on}')


def load_model(opt):
    print(f'opt.inspect_layer_position:{opt.inspect_layer_position}')
    if 'face' in opt.model:
        net = partial_models_adaptive.FaceAdaptivePartialModel(
            num_classes=opt.n_cls,  # in_dims=opt.in_dims,
            inspect_layer_position=opt.inspect_layer_position,
            original_input_img_shape=(1, 3, opt.size, opt.size)
        )
        num_ftrs = net.fc8.in_features
        net.fc8 = nn.Linear(num_ftrs, 526)
        net.eval()
        net.cuda()
        state_dict = torch.load(opt.ckpt)
        net.load_state_dict(state_dict, strict=False)
        return net
    if 'resnet' in opt.model:
        if '50' in opt.model:
            layer_setting = [3, 4, 6, 3]
            block_setting = Bottleneck
        elif '18' in opt.model:
            layer_setting = [2, 2, 2, 2]
            block_setting = BasicBlock
        else:
            raise NotImplementedError("Not implemented ResNet Setting!")
        model_classifier = partial_models_adaptive.ResNetAdaptivePartialModel(
            num_classes=opt.n_cls,
            inspect_layer_position=opt.inspect_layer_position,
            original_input_img_shape=(1, 3, opt.size, opt.size),
            layer_setting=layer_setting,
            block_setting=block_setting
        )
    elif 'vgg16' in opt.model:
        model_classifier = partial_models_adaptive.VGGAdaptivePartialModel(
            num_classes=opt.n_cls,  # in_dims=opt.in_dims,
            inspect_layer_position=opt.inspect_layer_position,
            original_input_img_shape=(1, 3, opt.size, opt.size)
        )
    elif 'google' in opt.model:
        model_classifier = partial_models_adaptive.GoogLeNetAdaptivePartialModel(
            num_classes=opt.n_cls,
            inspect_layer_position=opt.inspect_layer_position,
            original_input_img_shape=(1, 3, opt.size, opt.size)
        )
    elif 'simple_cnn' in opt.model:
        if 'mnist' in opt.ckpt:
            model_classifier = partial_models_adaptive.SimpleCNNAdaptivePartialModel(
                original_input_img_shape=(1, 1, 28, 28),
                in_channels=1
            )
        else:
            model_classifier = partial_models_adaptive.SimpleCNNAdaptivePartialModel()
    elif 'bad_encoder_full_model' in opt.model:
        # load bad encoder
        bad_encoder_model = SimCLR()
        bad_encoder_ckpt = torch.load('./BadEncoderSavedModels/good/bad_encoder_gtsrb.pth')
        bad_encoder_model.load_state_dict(bad_encoder_ckpt['state_dict'])
        # load cls
        classifier_in_bad_encoder = NeuralNet(512, [512, 256], 43)
        cls_ckpt = torch.load('./BadEncoderSavedModels/good/cls_gtsrb.pth')
        classifier_in_bad_encoder.load_state_dict(cls_ckpt['model'])
        model_classifier = bad_encoder_full_model_partial.BadEncoderFullModelAdaptivePartialModel(
            encoder=bad_encoder_model,
            classifier=classifier_in_bad_encoder,
            inspect_layer_position=opt.inspect_layer_position,
            original_input_img_shape=(1, 3, opt.size, opt.size)
        )
    else:
        raise NotImplementedError('Model not supported!')

    if 'bad_encoder_full_model' not in opt.model:
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        if 'Troj' not in opt.ckpt:
            try:
                state_dict = ckpt['net_state_dict']
            except KeyError:
                try:
                    print("wow! it is dict model")
                    state_dict = ckpt['model']
                except KeyError:
                    state_dict = ckpt['state_dict']
        else:
            model_classifier = ckpt

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     model_classifier = torch.nn.DataParallel(model_classifier)
        model_classifier = model_classifier.cuda()
        cudnn.benchmark = True
        if 'Troj' not in opt.ckpt and 'bad_encoder_full_model' not in opt.model:
            model_classifier.load_state_dict(state_dict)

    return model_classifier


def calculate_top2_predicted_class(image_tensor, model, purify_mal_channels_id_list=None, p=0.):
    output = model(x=image_tensor, pass_channel_id=-1,
                   purify_mal_channels_id_list=purify_mal_channels_id_list, dropout_p=p)
    output = torch.softmax(output, dim=1)
    # print("output:", output)
    _, pred = output.topk(2)
    pred = pred.t()
    pred = pred.cpu().numpy()
    return pred[0][0], pred[1][0]

def mysoftmax(x):
    a = torch.exp(x)
    tmp = torch.sum(a)
    a /= tmp
    return a
def calculate_predicted_scores(image_tensor, model, purify_mal_channels_id_list=None, p=0.):
    output = model(x=image_tensor, pass_channel_id=-1,
                   purify_mal_channels_id_list=purify_mal_channels_id_list, dropout_p=p)
    output = torch.softmax(output, dim=1)
    # print("output:", output)
    return output.detach().cpu().numpy()


def bound_dummy_input(dummy_input, lower_bound_template_tensor, upper_bound_template_tensor):
    # dummy_input should be restricted within the valid interval of an input image
    dummy_input = torch.where(dummy_input > upper_bound_template_tensor, upper_bound_template_tensor, dummy_input)
    dummy_input = torch.where(dummy_input < lower_bound_template_tensor, lower_bound_template_tensor, dummy_input)
    return dummy_input

#对于特定类，生成该类对应的repre层的表示
def optimize_inner_embedding(opt, model_classifier_part, inner_embedding_tensor_template,num_activation,desired_class):
    layer_name = 'classifier.4.weight'
    if opt.model == 'vgg16':
        layer_name = 'classifier.4.weight'
        bias_name = 'classifier.4.bias'
    elif opt.model == 'simple_cnn':
        layer_name = 'm2.1.weight'
        bias_name = 'm2.1.bias'
    elif opt.model == 'google_net':
        layer_name = 'fc.weight'
        bias_name = 'fc.bias'
    model_classifier_part.eval()
    weight = model_classifier_part.state_dict()
    dummy_inner_embedding_tensor = torch.rand_like(inner_embedding_tensor_template)
    dummy_inner_embedding_tensor.requires_grad = True
    criterion_adversarial = torch.nn.CrossEntropyLoss()
    label = torch.tensor([desired_class])
    if torch.cuda.is_available():
        model_classifier_part = model_classifier_part.cuda()
        label = label.cuda()
        dummy_inner_embedding_tensor = dummy_inner_embedding_tensor.cuda()
        cudnn.benchmark = True

    optimizer_adversarial_perturb = torch.optim.Adam([dummy_inner_embedding_tensor], lr=0.1,
                                                     weight_decay=0.005)  # scale of L2 norm

    for iters in range(1000):
        optimizer_adversarial_perturb.zero_grad()
        _pred = model_classifier_part(dummy_inner_embedding_tensor)
        loss_adversarial_perturb = criterion_adversarial(_pred, label)
        loss_adversarial_perturb.backward()
        optimizer_adversarial_perturb.step()

        if opt.bound_on:
            with torch.no_grad():
                dummy_inner_embedding_tensor.clamp_(0., 999.)
    sort_obj = torch.sort(dummy_inner_embedding_tensor.reshape(-1), descending=True)
    max_indices = sort_obj.indices.cpu().numpy()
    collected_max_indices = max_indices[:50]


    # #以下是新增的处理
    _dummy_inner_embedding = torch.rand_like(inner_embedding_tensor_template)
    ori_size = dummy_inner_embedding_tensor.shape
    sort_obj = torch.sort(dummy_inner_embedding_tensor.reshape(-1) * weight[layer_name][desired_class], descending=True)
    max_indices = sort_obj.indices.cpu().numpy()
    collected_max_indices = max_indices[:num_activation]

    # filename = str(desired_class)
    # file = open('%s.txt'%filename,'w')
    # array_str = '\n'.join(str(x) for x in collected_max_indices)
    # file.write(array_str)
    # file.close()

    _dummy_inner_embedding.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss()
    cudnn.benchmark = True
    optimizer_adversarial = torch.optim.Adam([_dummy_inner_embedding], lr=1e-2,
                                                     weight_decay=0.005)  # scale of L2 norm
    mask_factor=torch.ones_like(dummy_inner_embedding_tensor.reshape(-1))
    for ei in collected_max_indices:
        mask_factor[ei] = 0
    mask_factor=mask_factor.reshape(ori_size)

    for iters in range(1000):
        # optimization 1: adversarial perturb
        optimizer_adversarial.zero_grad()
        _dummy = torch.mul(_dummy_inner_embedding, mask_factor)
        _pred = model_classifier_part(_dummy)
        loss_adversarial = criterion(_pred, label)
        loss_adversarial.backward()

        optimizer_adversarial.step()

        if opt.bound_on:
            with torch.no_grad():
                _dummy_inner_embedding.clamp_(0., 999.)
    _dummy_inner_embedding = torch.mul(_dummy_inner_embedding.data, mask_factor)
    return _dummy_inner_embedding.detach()

# metrics for one targeted class: class-num (-1) class pairs
def compute_metrics_one_source(opt, model_cls, source, dummy_inner_embeddings_all):
    # compute average dummy of the targeted class
    dummies_target = dummy_inner_embeddings_all[source]
    dummy_sum_target = torch.zeros_like(dummies_target)
    for dummy in dummies_target:
        dummy_sum_target += dummy
    dummy_avg_target = dummy_sum_target / opt.num_dummy
    # feed dummy_avg_target to the model_cls, obtain the logits
    _logits = model_cls(dummy_avg_target)
    _scores = F.softmax(_logits, dim=1)
    _logits = _logits.detach().cpu().numpy()[0]
    _scores = _scores.detach().cpu().numpy()[0]
    return _scores[source]



def observe_important_neurons_for_one_class(opt, model_classifier_part,num_ac,source_class):
    """for the desired class, compute the important neurons by optimization on the inner embedding
    """
    model_classifier_part.eval()
    try:
        input_shape = model_classifier_part.input_shapes[opt.inspect_layer_position]
    except IndexError:
        input_shape = model_classifier_part.input_shapes[1]
    inner_embedding_template_tensor = torch.ones(size=input_shape)

    if torch.cuda.is_available():
        inner_embedding_template_tensor = inner_embedding_template_tensor.cuda()
    model_classifier_part = model_classifier_part.eval()

    # observe the active neurons of the optimized dummy input
    _dummy_inner_embedding = optimize_inner_embedding(opt, model_classifier_part, inner_embedding_template_tensor,
                                                      num_activation=num_ac,desired_class= source_class)

    # collect important neuron ids
    sort_obj = torch.sort(_dummy_inner_embedding.reshape(-1), descending=True)
    max_values = sort_obj.values.cpu().numpy()
    max_indices = sort_obj.indices.cpu().numpy()
    non_minor_id = opt.num_important_neurons
    collected_max_indices = max_indices[:non_minor_id]
    #排序一下，选择了前几个重要的激活对应的Indices
    #print(desired_class)
    # print(collected_max_indices)
    return _dummy_inner_embedding, collected_max_indices


def normalization_min_max(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def compute_metrics_for_array(anomaly_metric_value_array):
    _a_flat = anomaly_metric_value_array.flatten()
    _a_flat = _a_flat[_a_flat != 0.]

    _a_flat = np.sort(_a_flat)
    _length = len(_a_flat)
    q1_pos = int(0.25 * _length)
    q3_pos = int(0.75 * _length)
    _q1 = _a_flat[q1_pos]
    _q3 = _a_flat[q3_pos]
    _iqr = _q3 - _q1
    _anomaly_metric = (np.max(_a_flat) - _q3) / _iqr

    return _anomaly_metric


def compute_dummy_inner_embeddings(model_classifier, opt,num_ac):
    np.set_printoptions(precision=2, suppress=True)
   # dummy_inner_embeddings_all = [[] for i in range(opt.n_cls)]
    dummy_inner_embeddings_all = []
    max_ids_all = [set({}) for i in range(opt.n_cls)]
    print("\nStart generating and recording dummy inner embeddings for each class......")
    for i in range(opt.n_cls):
        _dummy_inner_embedding, max_ids = observe_important_neurons_for_one_class(opt, model_classifier, num_ac,i)
        dummy_inner_embeddings_all.append(_dummy_inner_embedding)
    # 对于每个类，对生成num_dummy=1次，每次都是repre
    return dummy_inner_embeddings_all


def inspect_saved_model(opt):
    # build partial model
    model_classifier = load_model(opt)
    model_classifier = model_classifier.eval()

    # #以下是特征融合
    # dummy_inner_embeddings_all=compute_dummy_inner_embeddings(model_classifier, opt, num_ac=0)
    # t = []
    # for i in range(opt.n_cls):
    #     dummies_target = dummy_inner_embeddings_all[i]
    #     dummy_sum_target = torch.zeros_like(dummies_target[0])
    #     for dummy in dummies_target:
    #         dummy_sum_target += dummy
    #     dummy_avg_target = dummy_sum_target / opt.num_dummy
    #     t.append(dummy_avg_target)
    # # feed dummy_avg_target to the model_cls, obtain the logits
    # _logits = model_classifier((sum(t)/opt.n_cls))
    # _scores = F.softmax(_logits, dim=1)
    # _logits = _logits.detach().cpu().numpy()[0]
    # _scores = _scores.detach().cpu().numpy()[0]
    # print(_scores)

    logits = []
    num_acs=[]
    for k in range(opt.n_cls):
        logits.append([])

    num_acs = range(0,3200,50)
    for i in num_acs:
        dummy_inner_embeddings_all = compute_dummy_inner_embeddings(model_classifier, opt, num_ac= i)

        for source_class in range(opt.n_cls):
            logits[source_class].append(compute_metrics_one_source(opt, model_classifier, source_class,
                                                           dummy_inner_embeddings_all))
    newfolder = 'mnist_patched_9'
    os.mkdir(newfolder)
    new_folder_path = os.path.abspath(newfolder)
    for cl in range(opt.n_cls):
        # 在新文件夹中创建一个文件
        new_file_path = os.path.join(new_folder_path, 'poi_confi%s.csv' % cl)
        data2 = pd.DataFrame(data=logits[cl], columns=['confidence'])
        data2.to_csv(new_file_path)
        # data1 = pd.DataFrame(data=simi1, columns=['similarity for each num_acs'])
        # data1.to_csv('similarity_discrease_clean.csv')
        # column_data1 = data1['similarity for each num_acs'].values.tolist()
        # column1 = []
        # for j in column_data1:
        #     column1.append(float(j))
        # #num_acs横坐标，logits[cl]纵坐标
        # plt.scatter(num_acs, column1)
        # # 添加坐标轴标签
        # plt.xlabel('num of 0 mask activations')
        # plt.ylabel('similarity')
        # # 添加标题
        # plt.title('similarity')
        # # 显示图形
        # plt.savefig("fea_normal.jpg")
        # plt.show()
        # plt.close()
        # data2 = pd.DataFrame(data=simi2, columns=['similarity for each num_acs'])
        # data2.to_csv('similarity_discrease_poi.csv')
        # column_data2 = data2['similarity for each num_acs'].values.tolist()
        # column2 = []
        # for j in column_data2:
        #     column2.append(float(j))
        # # num_acs横坐标，logits[cl]纵坐标
        # plt.scatter(num_acs, column2)
        # # 添加坐标轴标签
        # plt.xlabel('num of 0 mask activations')
        # plt.ylabel('similarity')
        # # 添加标题
        # plt.title('similarity')
        # # 显示图形
        # plt.savefig("fea_poil.jpg")
        # plt.show()
        # plt.close()



if __name__ == '__main__':
    opt = parse_option()
    inspect_saved_model(opt)
