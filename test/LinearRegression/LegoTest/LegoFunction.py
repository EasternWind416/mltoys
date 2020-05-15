import numpy as np
from bs4 import BeautifulSoup

from Main.Regression.RidgeRegress import RR


def scrapePage(retX, retY, fileName, yr, numPce, origPrc):

    # 打开并读取HTML文件
    with open(fileName, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read())
        i = 1

        # 根据HTML页面结构进行解析
        currentRow = soup.findAll('table', r="%d" % i)
        while(len(currentRow)!=0):
            currentRow = soup.findAll('table', r="%d" % i)
            title = currentRow[0].findAll('a')[1].text
            lwrTitle = title.lower()

            # 查找是否有全新标签
            if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
                newFlag = 1.0
            else:
                newFlag = 0.0

            # 查找是否已经标志出售，我们只收集已出售的数据
            soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
            if len(soldUnicde)==0:
                print("item #%d did not sell" % i)
            else:
                # 解析页面获取当前价格
                soldPrice = currentRow[0].findAll('td')[4]
                priceStr = soldPrice.text
                priceStr = priceStr.replace('$','') #strips out $
                priceStr = priceStr.replace(',','') #strips out ,
                if len(soldPrice)>1:
                    priceStr = priceStr.replace('Free shipping', '')
                sellingPrice = float(priceStr)

                # 去掉不完整的套装价格
                if  sellingPrice > origPrc * 0.5:
                        print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                        retX.append([yr, numPce, newFlag, origPrc])
                        retY.append(sellingPrice)
            i += 1
            currentRow = soup.findAll('table', r="%d" % i)


# 依次读取六种乐高套装的数据，并生成数据矩阵
def setDataCollect(retX, retY):
    scrapePage(retX, retY, './setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, './setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, './setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, './setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, './setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, './setHtml/lego10196.html', 2009, 3263, 249.99)


# 交叉验证
def crossValid(x, y, numVal=10):
    idx = [i for i in range(len(y))]
    numLam = 30
    errMat = np.zeros((numVal, numLam))

    x = np.array(x)
    y = np.array(y)

    w = np.zeros((numVal, numLam, x.shape[1] + 1))

    for curVal in range(numVal):
        """
        随机拆分数据
        90：训练
        10：测试
        """
        trainX = []
        trainY = []
        validX = []
        validY = []

        np.random.shuffle(idx)

        for cur_idx in range(len(y)):
            if cur_idx < .9 * len(y):
                trainX.append(x[idx[cur_idx]])
                trainY.append(y[idx[cur_idx]])
            else:
                validX.append(x[idx[cur_idx]])
                validY.append(y[idx[cur_idx]])

        trainX = np.array(trainX)
        trainY = np.array(trainY)
        validX = np.array(validX)
        validY = np.array(validY)

        trainXmean = trainX.mean(axis=0)
        trainXvar = trainX.var(axis=0)
        trainX = (trainX - trainXmean) / trainXvar
        validX = (validX - trainXmean) / trainXvar

        ones = np.ones(validX.shape[0]).reshape(-1, 1)
        validX = np.concatenate([ones, validX], axis=1)

        trainYmean = trainY.mean()
        trainY = trainY - trainYmean

        for curLam in range(numLam):
            lam = np.exp(curLam - 10)
            theta, *_ = RR().regress(trainX, trainY, lam)

            w[curVal, curLam, :] = theta.T

            preds = validX.dot(theta) + trainYmean

            errMat[curVal, curLam] = ((preds - validY) ** 2).sum()

            print('\nthe {0}th val, {1}th lam regression is ended'.format(curVal, curLam))


    errMean = errMat.mean(axis=0)
    best_lam_idx = np.argmin(errMean)

    errMean_lam = errMat[:, best_lam_idx]
    best_val_idx = np.argmin(errMean_lam)

    best_w = w[best_val_idx, best_lam_idx, :].T
    print(errMean)
    print('the min err is: ', errMat[best_val_idx, best_lam_idx])
    print('the best lam is: ', best_lam_idx)

    # 还原数据
    xMean = x.mean(axis=0)
    xVar = x.var(axis=0)

    yMean = y.mean()

    origin_best_w = best_w.copy()
    origin_best_w[1: ] = best_w[1: ] / xVar
    origin_best_w[: 1] = best_w[: 1]

    # 输出模型
    print('\nthe best model params from ridge regression is:\n', origin_best_w)
    print('\nwith constant term: ', yMean - xMean.dot(origin_best_w[1: ]))
