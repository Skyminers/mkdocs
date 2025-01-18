
# 实现决策树

首先我们准备了一些数据用于演示：

| age         | income | student | credit_rating | buys_computer |
| ----------- | ------ | ------- | ------------- | ------------- |
| youth       | high   | no      | fair          | no            |
| youth       | high   | no      | excellent     | no            |
| middle_aged | high   | no      | fair          | yes           |
| senior      | medium | no      | fair          | yes           |
| senior      | low    | yes     | fair          | yes           |
| senior      | low    | yes     | excellent     | no            |
| middle_aged | low    | yes     | excellent     | yes           |
| youth       | medium | no      | fair          | no            |
| youth       | low    | yes     | fair          | yes           |
| senior      | medium | yes     | fair          | yes           |
| youth       | medium | yes     | excellent     | yes           |
| middle_aged | medium | no      | excellent     | yes           |
| middle_aged | high   | yes     | fair          | yes           |
| senior      | medium | no      | excellent     | no            |

## 1. 数据处理

给定的数据中包含`age, income, student, credit_rating`四个维度的数据，预测`buys_computer`值，即根据个人信息来预测这个人会不会购买电脑。

为了方便处理，我们将这份表格数据转换为 Python 代码中的 `List` 来进行储存：

```python
from math import log
import operator
import copy

def createDataSet():
    # 数据集
    dataSet = [['youth', 'high', 'no', 'fair', 'no'],
               ['youth', 'high', 'no', 'excellent', 'no'],
               ['middle_aged', 'high', 'no', 'fair', 'yes'],
               ['senior', 'medium', 'no', 'fair', 'yes'],
               ['senior', 'low', 'yes', 'fair', 'yes'],
               ['senior', 'low', 'yes', 'excellent', 'no'],
               ['middle_aged', 'low', 'yes', 'excellent', 'yes'],
               ['youth', 'medium', 'no', 'fair', 'no'],
               ['youth', 'low', 'yes', 'fair', 'yes'],
               ['senior', 'medium', 'yes', 'fair', 'yes'],
               ['youth', 'medium', 'yes', 'excellent', 'yes'],
               ['middle_aged', 'medium', 'no', 'excellent', 'yes'],
               ['middle_aged', 'high', 'yes', 'fair', 'yes'],
               ['senior', 'medium', 'no', 'excellent', 'no']]
    labels = ['age', 'income', 'student', 'credit_rating']
    return dataSet, labels
```

## 2. 构建决策树

构建决策树是一个递归的过程，我们不断从当前需要进行构建的节点中挑选出一个可以“更好地”将训练数据集划分开的特征进行分割。构建决策树的主函数如下：

```python
def createTree(dataSet, labels):
		# 取出每一个数据集的 buys_computer 标签
    classList = [example[-1] for example in dataSet]
    # 如果所有的数据都是同样的 buys_computer 标签则直接终止
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 若已无特征可以用于划分，则找出数量更多的结果作为该节点的结果
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

		# 找出最合适的特征来进行划分，并将使用过的特征从特征列表中删除
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])

    # 构建决策树的转移
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), copy.copy(labels))

    return myTree
```

上面的代码中最需要注意的是函数`chooseBestFeatureToSplit`的实现，在这个函数中，我们需要分别计算每一个特征 $A$ 带来的信息增益。

$$
Gain(A) = Info(D) - Info_A(D), \\ Info(D) = -\sum_{i=1}^mp_i\log_2p_i, \\ Info_A(D) = \sum_{j=1}^v \frac{|D_j|}{|D|}\times Info(D_j)
$$

我们先来实现对 $Info(D)$ 进行计算的函数：

```python
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}

    # 遍历当前数据集中所有的数据
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
		    # 计算样本属于类别 key 的概率
        prob = float(labelCounts[key]) / numEntires

        # 计算香农熵
        shannonEnt -= prob*log(prob, 2)

    return shannonEnt
```

然后完成 $Gain(A)$ 的计算和特征 $A$ 的挑选：

```python
def chooseBestFeatureToSplit(dataSet):

		# 特征数量
    numFeatures = len(dataSet[0]) - 1

    # 计算 Info(D)
    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0
    bestFeature = -1
    # 计算每一个 Feature 的 Info_A(D)
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
		        # splitDataSet 函数的作用是将数据集中第 i 个特征为 value 的样本提取出来
            subDataSet = splitDataSet(dataSet, i, value)

            # 计算 Info_A(D)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        print("Gains on the %d-th feature are %.3f" % (i, infoGain))
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature
```

上面的代码可以完成对决策树的构建的主要过程，其中使用到了两个没有实现的函数 `majorityCnt` 和 `splitDataSet` ，这两个函数的作用分别为：

- `majorityCnt(D)` : 计算数据集 `D` 中哪一个类别的样本出现的最多
- `splitDataSet(D, i, value)` : 将数据集中第 $i$ 维特征值为 `value` 的样本提取出来组合成新的数据集

代码实现如下：

```python
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 通过 sort 函数排序的方式挑选出出现最多的类别
    sortedClassCount = sorted(classCount.items(),
							  key = operator.itemgetter(1),
							  reverse = True)
    return sortedClassCount[0][0]

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
		    # 通过 if 语句筛选符合要求的样本
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet
```

## 3. 使用决策树进行预测

在上一个实验环节构造的决策树保存在名为 `myTree` 的变量中，该变量是一个嵌套的 `dictionariy` ，我们可以通过 `print` 语句输出 `myTree` 来观察一下结构：

```json
{
  "age": {
    "middle_aged": "yes",
    "senior": {
      "credit_rating": {
        "fair": "yes",
        "excellent": "no"
      }
    },
    "youth": {
      "student": {
        "no": "no",
        "yes": "yes"
      }
    }
  }
}
```

举个例子，如果我们需要预测一个特征为：`['youth', 'low', 'yes', 'excellent']`的样本是否会购买电脑，那我们可以通过这个决策树来进行预测：

1. 根据决策树的结构，首先我们判断其中的 `age` 特征的值，由于样本中该特征的值为 `youth` 所以我们进入到 `youth` 分枝。
2. 接下来我们判断 `student` 特征的值，由于样本中该特征的值为 `yes` 所以我们进入到 `yes` 分枝
3. 此时我们抵达了决策树的叶子结点 `yes` ，因此我们得到了对该样本的预测：即该样本会选择购买电脑。

我们将上述的过程用代码来实现就可以自动使用决策树进行预测：

```python
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))

    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = None

    # 判断下一个决策树节点是否为叶子结点
    if type(secondDict[testVec[featIndex]]).__name__ == 'dict':
		    # 若不为叶子结点的话通过递归处理，进行下一次特征判断
        classLabel = classify(secondDict[testVec[featIndex]], featLabels, testVec)
    else:
		    # 若为叶子结点则可以得出预测结果
        classLabel = secondDict[testVec[featIndex]]

    return classLabel


def main():
	# 获取数据集
    dataSet, features = createDataSet()
    # 构建 Decision Tree
    myTree = createTree(dataSet, features)

		# 特征标注
    labels = ['age', 'income', 'student', 'credit_rating']
    # 测试数据
    testVec = ['youth', 'low', 'yes', 'excellent']

    # 样本预测
    result = classify(myTree, labels, testVec)

    print(result) # 输出预测结果
    print(myTree) # 输出 Decision Tree
```

输出结果：`yes`

## 通过 Sklearn 库实现决策树\*

在 Python 中，sklearn 包包含了决策树的实现，所以在实际应用中我们可以直接通过 Sklearn 包进行决策树的训练与预测。

首先需要安装 sklearn 包，在命令行中通过命令 `python3.9 -m pip install scikit-learn` 进行安装（其中的 `python3.9` 需要替换为你的 python 版本）如果电脑中只安装了一个 python 版本也可以使用命令：`pip install scikit-learn` 来进行安装。

在使用 sklearn 包内的决策树时，我们需要将数据集从 `string` 类型转换为 `int` 类型以方便处理，所以我们添加一段代码到数据集生成部分进行处理，然后使用 Sklearn 包进行决策树训练和预测即可。全部的代码如下：

```python
from sklearn import tree

age_map = {'youth': 0, 'middle_aged': 1, 'senior': 2}
income_map = {'low': 0, 'medium': 1, 'high': 2}
student_map = {'no': 0, 'yes': 1}
credit_rating_map = {'fair': 0, 'excellent': 1}
buys_computer_map = {'no': 0, 'yes': 1}

def createDataSet():
    # 数据集
    dataSet = [['youth', 'high', 'no', 'fair', 'no'],
               ['youth', 'high', 'no', 'excellent', 'no'],
               ['middle_aged', 'high', 'no', 'fair', 'yes'],
               ['senior', 'medium', 'no', 'fair', 'yes'],
               ['senior', 'low', 'yes', 'fair', 'yes'],
               ['senior', 'low', 'yes', 'excellent', 'no'],
               ['middle_aged', 'low', 'yes', 'excellent', 'yes'],
               ['youth', 'medium', 'no', 'fair', 'no'],
               ['youth', 'low', 'yes', 'fair', 'yes'],
               ['senior', 'medium', 'yes', 'fair', 'yes'],
               ['youth', 'medium', 'yes', 'excellent', 'yes'],
               ['middle_aged', 'medium', 'no', 'excellent', 'yes'],
               ['middle_aged', 'high', 'yes', 'fair', 'yes'],
               ['senior', 'medium', 'no', 'excellent', 'no']]
    # 循环枚举每个样本，将样本中的 string 数据转换为 int 数据
    for data in dataSet:
        data[0] = age_map[data[0]]
        data[1] = income_map[data[1]]
        data[2] = student_map[data[2]]
        data[3] = credit_rating_map[data[3]]
        data[4] = buys_computer_map[data[4]]
    labels = ['age', 'income', 'student', 'credit_rating']
    return dataSet, labels

if __name__ == '__main__':
    dataSet, _ = createDataSet()
    X = []
    Y = []
    for data in dataSet:
        X.append(data[:-1])
        Y.append(data[-1])

		# 定义决策树
    decisionTree = tree.DecisionTreeClassifier()

    # 通过数据集训练决策树
    decisionTree.fit(X, Y)

    # 使用决策树进行预测
    # ['youth', 'low', 'yes', 'excellent']
    print(decisionTree.predict([[0, 0, 1, 1], [1,1,0,0]]))
```
