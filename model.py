#-*- coding:utf-8 -*-
import sklearn

class DT(object):
    '''
    决策树，决策树可以处理离散值和连续值
    '''
    def __init__(self,data):
        from sklearn import tree
        X = data.drop('target',axis=1)
        Y = data['target']
        self.col_num = len(X.columns.values)
        print '输入维度',self.col_num
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(X,Y)

    def predict(self,predict):
        #预测结果带标签
        print len(predict.columns.values)
        if len(predict.columns.values) == self.col_num:
            X = predict
        elif len(predict.columns.values)-1 == self.col_num:
            X = predict.drop('target', axis=1)
        else:
            raise ('The input demension does not match')
        out = self.clf.predict(X)
        return out

