# !/usr/bin/python3
# -*- codeing = utf-8 -*-
# @Time : 7/28/2021 7:06 PM
# @Author : Liu
# @File : stacking.py
# @Software : PyCharm

'''
版本号：v.1.20
版本功能：
    1、可以进行stacking的融合
    2、可以预测分类模型
    3、可以使用其它评价指标计算得分
    4、可以预测正负的概率
    5、可以选择是使用分类还是得分进行预测
'''

from sklearn import svm as SVM
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as XGB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
import numpy as np
from sklearn.model_selection import ParameterGrid

__all__ = ['Stacking',]

def my_score(y_true, y_pred, y_proba=0):
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()
    acc = metrics.accuracy_score(y_true, y_pred)
    return_data = {'TN':TN, 'FP':FP, 'FN':FN, 'TP':TP, 'ACC':acc}
    return return_data


def my_score_proba(y_true, y_pred, y_proba):
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()
    acc = metrics.accuracy_score(y_true, y_pred)
    MCC = metrics.matthews_corrcoef(y_true, y_pred)
    auroc = metrics.roc_auc_score(y_true, y_proba[:, 1])
    precision, recall, _thresholds = metrics.precision_recall_curve(y_true, y_proba[:, 1])
    auprc = metrics.auc(recall, precision)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    # Precision = TP / (TP + FP)
    F1Score = 2 * TP / (2 * TP + FP + FN)
    recall = TP / (TP + FN)
    return_data = {'TN':TN, 'FP':FP, 'FN':FN, 'TP':TP, 'F1':F1Score, 'RECALL':recall, 'SPE':Specificity, 'SEN':Sensitivity, 'ACC':acc, 'MCC':MCC,'AUPRC':auprc, 'AUROC':auroc }
    return return_data


def get_kfold_grid(xlf_cls, X_train, y_train, param_keys, param_grid_list, cv_number, predict_proba_ctrl, evaluation, prediction_index='class'):
    '''
    :param xlf_cls:
    :param X_train:
    :param y_train:
    :param param_keys:
    :param param_grid_list:
    :param cv_number:
    :return:
    '''

    import numpy as np

    best_param = {}
    for keys in param_keys:
        best_param[keys] = None
    import random
    kfold = StratifiedKFold(n_splits=cv_number, shuffle=True, random_state=int(random.random()*100))
    max_auroc = -1.0
    y_true = []
    y_predict = []
    max_evaluation = {}
    y_predict_proba = []
    for param in param_grid_list:
        # xlf_temp = clone(xlf)
        xlf = xlf_cls()
        for keys in param_keys:
            setattr(xlf, keys, param[keys])
        cvscores = []
        all_val_y = []
        all_val_score1 = []
        all_val_score2 = []
        all_val_idx = []
        scores_list = []
        i = 0
        for tr_idx, val_idx in kfold.split(X_train, y_train):
            all_val_idx.append(val_idx)
            tr_X_tmp, val_X_tmp, tr_y, val_y = X_train[tr_idx], X_train[val_idx], y_train[tr_idx], y_train[val_idx]
            xlf.fit(tr_X_tmp, tr_y)
            val_score1 = xlf.predict(val_X_tmp)
            if predict_proba_ctrl == 0:
                val_score2 = xlf.predict_proba(val_X_tmp)
                scores = my_score_proba(val_y, val_score1, val_score2)
                all_val_score2.append(val_score2)
            else:
                scores = my_score(val_y, val_score1)
            all_val_y.append(val_y)
            all_val_score1.append(val_score1)

            cvscores.append(list(scores.values()))
            scores_list.append(scores)
            i = i + 1
            # all_val_y[0] = list(all_val_y[0]).extend(list(all_val_y[j]))
            # all_val_score1[0] = list(all_val_score1[0]).extend(list(all_val_score1[j]))
            # all_val_score2[0] = list(all_val_score2[0]).extend(list(all_val_score2[j]))
        mcvscore = np.mean(cvscores, axis=0)
        if_value = list(scores_list[0].keys()).index(evaluation)
        if mcvscore[if_value] > max_auroc:
            if predict_proba_ctrl == 1:
                for j in range(1, cv_number):
                    all_val_idx[0] = np.concatenate((all_val_idx[0], all_val_idx[j]), axis=0)
                    all_val_y[0] = np.concatenate((all_val_y[0], all_val_y[j]), axis=0)
                    all_val_score1[0] = np.concatenate((all_val_score1[0], all_val_score1[j]), axis=0)
                indx = np.argsort(all_val_idx[0])
                y_true =  all_val_y[0][indx]
                y_predict = all_val_score1[0][indx]

                best_param = param
                max_auroc = mcvscore[if_value]
            else:
                for j in range(1, cv_number):
                    all_val_idx[0] = np.concatenate((all_val_idx[0], all_val_idx[j]), axis=0)
                    all_val_y[0] = np.concatenate((all_val_y[0], all_val_y[j]), axis=0)
                    all_val_score1[0] = np.concatenate((all_val_score1[0], all_val_score1[j]), axis=0)
                    all_val_score2[0] = np.concatenate((all_val_score2[0], all_val_score2[j]), axis=0)
                indx = np.argsort(all_val_idx[0])
                y_true =  all_val_y[0][indx]
                y_predict = all_val_score1[0][indx]
                y_predict_proba = all_val_score2[0][indx]
                best_param = param
                max_auroc = mcvscore[if_value]

            for key in scores_list[0].keys():
                max_evaluation[key] = mcvscore[list(scores_list[0].keys()).index(key)]


    xlf_best = xlf_cls()
    for keys in best_param.keys():
        setattr(xlf_best, keys, best_param[keys])
    xlf_best.fit(X_train, y_train)
    if prediction_index == 'class':
        return xlf_best, best_param, y_true, y_predict, max_evaluation
    else:
        return xlf_best, best_param, y_true, y_predict_proba[:,1], max_evaluation


def get_kfold_(xlf_cls, X_train, y_train, cv_number, predict_proba_ctrl, evaluation, prediction_index='class'):
    '''

    :param xlf_cls:
    :param X_train:
    :param y_train:
    :param cv_number:
    :return:
    '''
    import numpy as np

    best_param = {}

    kfold = StratifiedKFold(n_splits=cv_number, shuffle=True, random_state=None)
    max_auroc = -1.0
    y_true = []
    y_predict = []
    max_evaluation = {}
    xlf = xlf_cls()
    cvscores = []
    all_val_y = []
    all_val_score1 = []
    all_val_score2 = []
    all_val_idx = []
    scores_list = []
    i = 0
    for tr_idx, val_idx in kfold.split(X_train, y_train):
        all_val_idx.append(val_idx)
        tr_X_tmp, val_X_tmp, tr_y, val_y = X_train[tr_idx], X_train[val_idx], y_train[tr_idx], y_train[val_idx]
        xlf.fit(tr_X_tmp, tr_y)
        val_score1 = xlf.predict(val_X_tmp)
        if predict_proba_ctrl == 0:
            val_score2 = xlf.predict_proba(val_X_tmp)
            scores = my_score_proba(val_y, val_score1, val_score2)
            all_val_score2.append(val_score2)
        else:
            scores = my_score(val_y, val_score1)
        all_val_y.append(val_y)
        all_val_score1.append(val_score1)

        cvscores.append(list(scores.values()))
        scores_list.append(scores)
        i = i + 1
        # all_val_y[0] = list(all_val_y[0]).extend(list(all_val_y[j]))
        # all_val_score1[0] = list(all_val_score1[0]).extend(list(all_val_score1[j]))
        # all_val_score2[0] = list(all_val_score2[0]).extend(list(all_val_score2[j]))
    mcvscore = np.mean(cvscores, axis=0)

    if_value = list(scores_list[0].keys()).index(evaluation)
    # if mcvscore[if_value] > max_auroc:
    if predict_proba_ctrl == 1:
        for j in range(1, cv_number):
            all_val_idx[0] = np.concatenate((all_val_idx[0], all_val_idx[j]), axis=0)
            all_val_y[0] = np.concatenate((all_val_y[0], all_val_y[j]), axis=0)
            all_val_score1[0] = np.concatenate((all_val_score1[0], all_val_score1[j]), axis=0)
        indx = np.argsort(all_val_idx[0])
        y_true = all_val_y[0][indx]
        y_predict = all_val_score1[0][indx]
        y_predict_proba = all_val_score1[0][indx]

        max_auroc = mcvscore[if_value]
    else:
        for j in range(1, cv_number):
            all_val_idx[0] = np.concatenate((all_val_idx[0], all_val_idx[j]), axis=0)
            all_val_y[0] = np.concatenate((all_val_y[0], all_val_y[j]), axis=0)
            all_val_score1[0] = np.concatenate((all_val_score1[0], all_val_score1[j]), axis=0)
            all_val_score2[0] = np.concatenate((all_val_score2[0], all_val_score2[j]), axis=0)
        indx = np.argsort(all_val_idx[0])
        y_true = all_val_y[0][indx]
        y_predict = all_val_score1[0][indx]

        max_auroc = mcvscore[if_value]
    I_key = 0
    for key in scores_list[0].keys():
        max_evaluation[key] = mcvscore[I_key]
        I_key += 1

    best_param = None
    xlf_best = xlf_cls()
    for keys in best_param.keys():
        setattr(xlf_best, keys, best_param[keys])
    xlf_best.fit(X_train, y_train)
    if prediction_index == 'class':
        return xlf_best, best_param, y_true, y_predict, max_evaluation
    else:
        return xlf_best, best_param, y_true, y_predict_proba[:,1], max_evaluation


class Stacking(object):
    __methods_list = []
    __param_inlist = {}
    __xlf_dict = {}

    def __init__(self, tatol_layer, evaluate_indicator='ACC', Prediction_index='class'):
        '''
        :param tatol_layer:
        :param evaluate_indicator:
            {'TN':TN, 'FP':FP, 'FN':FN, 'TP':TP, 'F1':F1Score, \
            'RECALL':recall, 'SPE':Specificity, 'SEN':Sensitivity, \
            'ACC':acc, 'MCC':MCC,'AUPRC':prc, 'AUROC':roc }
        :Prediction_index:
            'class','score'
        :param layer: 大于等于2
        :param x_data:  无
        :param y_data:  无
        '''

        if not (Prediction_index in ['class', 'score']):
            raise Exception('Prediction_index错误...')

        self.layer = tatol_layer
        self.evaluate_indicator = evaluate_indicator
        self.prediction_index = Prediction_index # 控制使用得分还是类别当作新的x值
        if tatol_layer <= 1:
            raise Exception('layer应当大于等于2...')

        self.xlf_list = []  # 开始的算法
        self.xlf_over_list = []  # 训练好的模型
        self.xlf_layer_xy = []  # 层的x,y
        self.layer_layer_number = []  # 存储层与层之间的关系
        self.__layer_ctrl = []
        self.__predict_proba_ctrl = 0  # 0代表proba可以使用，1代表不可用

        self.__train_scores = []
        self.__predict_scores = []

        for i in range(0, tatol_layer):
            self.xlf_list.append([])
            self.xlf_over_list.append([])
            self.xlf_layer_xy.append([[],[]])
            # self.layer_layer_number.append([])
            self.__layer_ctrl.append(0)
            self.__train_scores.append([])
            self.__predict_scores.append([])

    def creat_method(self,):
        pass

    def __estimator_type(self, xlf):
        try:
            return xlf()._estimator_type
        except Exception:
            raise Exception('xlf_append: method类型错误, 您输入的可能不是一个分类器“类”...')

    def __param_grid_type(self, xlf, param_grid):
        parm_keys = []
        for i in param_grid.keys():
            if not (hasattr(xlf(), i)):
                raise Exception('xlf_append: {} 属性在{}中不存在..'.format(i, xlf()))
            parm_keys.append(i)
        return parm_keys
    def __predict_proba_type(self, xlf, param):
        try:
            param_grid_dict = list(ParameterGrid(param))[0]
        except TypeError as te:
            raise te('请加[]...')
        xlf_ = xlf()
        for keys in param_grid_dict.keys():
            setattr(xlf_, keys, param_grid_dict[keys])
        if not (hasattr(xlf_, 'predict_proba')):
            print('xlf_append: 警告！！{}模型没有 predict_proba 属性，因此会导致整个stacking无法进行predict_proba...'.format(xlf()))
            print('xlf_append: 警告！！evaluate_indicator 被调整为默认值 ACC')
            print('xlf_append: 警告！！prediction_index被调整为默认值 class')
            self.__predict_proba_ctrl = 1
            self.evaluate_indicator = 'ACC'
            self.prediction_index = 'class'
            return 1
        else:
            return 0

    def xlf_append(self, layer_th, method, param_grid=None, weight=1, split_number=2):
        '''
        :param layer_th: 构建的第几层
        :param method: 方法，包括:
        :param param: 类似于: param_grid = {
                                    'max_depth': [14,16],
                                    'learning_rate': [0.005, 0.01],
                                    'n_estimators': [1800, 1900],
                                    }
        :param weight: 权重
        :return:
        '''
        layer_th = layer_th - 1
        self.__estimator_type(method)
        self.__param_grid_type(method, param_grid)
        self.__predict_proba_type(method, param_grid)
        if split_number <=1:
            print('xlf_append: 警告！！split_number不能小于2，已经自动设置为2...')
            split_number = 2

        if [method, param_grid, weight, split_number] in self.xlf_list[int(layer_th)]:
            print('xlf_append: 警告！！在第{}层中已经有完全相同的算法...'.format(layer_th))
        # if not(len(param) == self.__param_inlist[method]):
        #     raise Exception('xlf_append: {}算法的参数为{}个，您输入错误了...'.format(method, self.__param_inlist[method]))
        if layer_th+1 == self.layer:
            if len(self.xlf_list[int(layer_th)]) != 0:
                print('xlf_append: 警告！！最后一层只能有一个算法,自动使用第一次构建的算法...')
                return
        if self.__layer_ctrl != 0:
            pass

        self.xlf_list[int(layer_th)].append([method, param_grid, weight, split_number])

    def layer_connect_layer(self, layer1, layer2, weight=1):  # 层与层之间的连接关系
        if layer1 >= layer2:
            raise Exception('layer_connect_layer: layer2不能小于或等于layer1...')
        if layer1 >= self.layer or layer2 > self.layer:
            raise Exception('layer_connect_layer: layer2不能大于层数，layer1不能大于等于层数...')
        if layer1 <=0 or layer2 <= 1:
            raise Exception('layer_connect_layer: layer2不能小于2，layer1不能小于1...')

        self.layer_layer_number.append([layer1, layer2, weight])

    def set_data(self,x,y):
        self.__x_data = np.array(x)
        self.__y_data = np.array(y)
        ## 判断x与y的维数是否相等
        if self.__x_data.shape[0] != self.__y_data.shape[0]:
            raise Exception('x,y维度不一致，分别为{}，{}'.format(self.__x_data.shape, self.__y_data.shape))

    def train_stacking(self, cv_number=3):
        # self.cv_number = cv_number ## 后期拓展自动优化权重时使用
        x_data = self.__x_data
        y_data = self.__y_data
        neg_layer, pos_layer = self.__layer_chart()
        for layer_start in range(0,self.layer):
            print('正在计算第{}层，请稍等...'.format(layer_start+1))
            for number_start in range(0, len(self.xlf_list[layer_start])):
                self.__Models_calculation(layer_start, number_start, x_data, y_data)
            # x值(y_pred):
            self.xlf_layer_xy[layer_start][1] = np.array([self.xlf_over_list[layer_start][i][3] for i in range(0, len(self.xlf_list[layer_start]))]).T
            # y_true值:
            self.xlf_layer_xy[layer_start][0] = self.xlf_over_list[layer_start][0][2]
            if layer_start+1 < self.layer:
                need_layer = neg_layer[layer_start+1]

                if len(need_layer)>1:
                    x_data = np.concatenate([self.xlf_layer_xy[need_layer[i][0]-1][1]*need_layer[i][1] for i in range(0,len(need_layer)) ],axis=1)
                    # need_layer[i][0] 输入层
                else:
                    in_layer_number = need_layer[0][0]-1
                    x_data = self.xlf_layer_xy[in_layer_number][1]*need_layer[0][1]

                y_data = self.xlf_layer_xy[layer_start][0]  ## 这里可能有问题，还没想好怎么解决

    def predict(self,X, ydata=None):
        print('-'*40)
        x_data = X
        neg_layer, pos_layer = self.__layer_chart()
        for layer_start in range(0, self.layer):
            print('正在预测第{}层，请稍等...'.format(layer_start + 1))
            x_data_list = []
            for number_start in range(0, len(self.xlf_list[layer_start])):
                x_data_list.append(self.__Models_predict(layer_start, number_start, x_data,ydata))

            self.xlf_layer_xy[layer_start][1] = np.array(
                [x_data_list[i] for i in range(0, len(x_data_list))]).T
            self.xlf_layer_xy[layer_start][0] = 0
            if layer_start + 1 < self.layer:
                need_layer = neg_layer[layer_start + 1]

                if len(need_layer) > 1:
                    x_data = np.concatenate(
                        [self.xlf_layer_xy[need_layer[i][0]-1][1] * need_layer[i][1] for i in range(0, len(need_layer))],
                        axis=1)
                    # need_layer[i][0] 输入层
                else:
                    x_data = self.xlf_layer_xy[need_layer[0][0]-1][1] * need_layer[0][1]
        y_pre_ = (self.xlf_layer_xy[self.layer - 1][1].T)[0]
        y_pre_[y_pre_>0.5] = 1
        y_pre_[y_pre_<=0.5] = 0
        return y_pre_

    def predict_proba(self, X, ydata=None):
        print('-' * 40)
        if self.__predict_proba_ctrl == 1:
            print('predict_proba: 无法计算...')
            return None
        else:
            x_data = X
            neg_layer, pos_layer = self.__layer_chart()
            for layer_start in range(0, self.layer):
                print('正在预测第{}层，请稍等...'.format(layer_start + 1))
                x_data_list = []
                if layer_start + 1 != self.layer:
                    for number_start in range(0, len(self.xlf_list[layer_start])):
                        x_data_list.append(self.__Models_predict(layer_start, number_start, x_data, ydata))
                else:
                    x_data_list.append(self.__Models_predict(layer_start, 0, x_data, ydata, proba_need=0))

                self.xlf_layer_xy[layer_start][1] = np.array(
                    [x_data_list[i] for i in range(0, len(x_data_list))]).T
                self.xlf_layer_xy[layer_start][0] = 0

                if layer_start + 1 < self.layer:
                    need_layer = neg_layer[layer_start + 1]

                    if len(need_layer) > 1:
                        x_data = np.concatenate(
                            [self.xlf_layer_xy[need_layer[i][0] - 1][1] * need_layer[i][1] for i in
                             range(0, len(need_layer))],
                            axis=1)
                        # need_layer[i][0] 输入层
                    else:
                        x_data = self.xlf_layer_xy[need_layer[0][0] - 1][1] * need_layer[0][1]
            return (self.xlf_layer_xy[self.layer - 1][1].T)[0]

    def __Models_calculation(self, layer, number, x_data, y_data):
        xlf_name = self.xlf_list[layer][number][0]
        param_grid = self.xlf_list[layer][number][1]
        weight = self.xlf_list[layer][number][2]
        split_number = self.xlf_list[layer][number][3]

        if isinstance(param_grid,dict):
            param_keys = self.__param_grid_type(xlf_name, param_grid)
            param_grid_list = list(ParameterGrid(param_grid))
            xlf_best, best_param, y_true, y_predict, scores = get_kfold_grid(xlf_name, x_data, y_data, param_keys, param_grid_list=param_grid_list, cv_number=split_number, predict_proba_ctrl=self.__predict_proba_ctrl, evaluation=self.evaluate_indicator)
            print('第{}层第{}个算法超参数最优为: {}'.format(layer + 1, number + 1, best_param))
        else:
            xlf_best, best_param, y_true, y_predict, scores = get_kfold_(xlf_name, x_data, y_data, cv_number=split_number, predict_proba_ctrl=self.__predict_proba_ctrl, evaluation=self.evaluate_indicator)
            print('第{}层第{}个算法超参数为: 默认参数')
        self.__train_scores[layer].append(scores)
        print('{}:{}'.format(self.evaluate_indicator,scores[self.evaluate_indicator]))
        y_predict = y_predict*float(weight)
        self.xlf_over_list[layer].append([xlf_best, best_param, y_true, y_predict])

    def __Models_predict(self, layer, number, x_data, y_data, proba_need=1):
        weight = self.xlf_list[layer][number][2]
        xlf = self.xlf_over_list[layer][number][0]
        y_pre = xlf.predict(x_data)
        if self.__predict_proba_ctrl == 0:
            y_proba = xlf.predict_proba(x_data)
        else:
            y_proba = None
        if not (y_data is None):
            scores = self.__pred_score(y_data, y_pre, y_proba)
            self.__predict_scores[layer].append(scores)
            print('第{}层第{}个算法预测{}为：{}'.format(layer + 1, number + 1,self.evaluate_indicator,scores[self.evaluate_indicator]))
        else:
            print('第{}层第{}个算法完成..'.format(layer + 1, number + 1))
        if self.prediction_index == 'class':
            if proba_need == 0:
                return y_proba
            else:
                return y_pre * float(weight)
        else:
            if proba_need == 0:
                return y_proba
            else:
                return y_proba[:,1] * float(weight)

    def __pred_score(self, y_data, y_pre, y_proba=None):
        if self.__predict_proba_ctrl == 1:
            return my_score(y_data, y_pre)
        else:
            return my_score_proba(y_data, y_pre, y_proba)

    def __layer_chart(self):
        layer = self.layer
        pos_layer = []
        neg_layer = []
        for l in range(0, layer):
            neg_layer.append([])
        neg_layer[0] = 'x'
        for i in range(layer-1, 0, -1):
            for connect in self.layer_layer_number:
                if i+1 == connect[1]:
                    neg_layer[i].append([connect[0],connect[2]])  # 层，权重

        for l in range(0, layer):
            pos_layer.append([])
        pos_layer[layer-1] = 'y'
        for i in range(0, layer-1):
            for connect in self.layer_layer_number:
                if i+1 == connect[0]:
                    pos_layer[i].append([connect[1],connect[2]])
        if [] in neg_layer :
            raise Exception('layer_connect_layer: 第{}层有向无环图构建错误, 缺少输入层...'.format(neg_layer.index([])))
        if [] in pos_layer:
            raise Exception('layer_connect_layer: 第{}层有向无环图构建错误, 缺少输出层...'.format(pos_layer.index([])))
        return neg_layer, pos_layer

    @property
    def get_layer(self):
        return self.layer

    @property
    def get_chart(self):
        neg_layer, pos_layer = self.__layer_chart()
        return neg_layer

    @property
    def get_train_scores(self):
        return self.__train_scores

    @property
    def get_predict_scores(self):
        return self.__predict_scores
