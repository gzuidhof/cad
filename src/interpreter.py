from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
def load_data(filenames):
    data = []
    for filename in filenames:
        data.append(joblib.load('../data/{0}'.format(filename)))
    return data


def boxplot(data,labels):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('Boxplots different classifiers')
    plt.boxplot(data)
    xtickNames = plt.setp(ax1, xticklabels=labels)
    plt.setp(xtickNames, rotation=25, fontsize=12)
    for file in data:
        print labels,np.mean(file),np.std(file)

if __name__ == "__main__":
    filenames = ['logreg_dice.pkl','rf200_dice.pkl','adaboost200_dice.pkl']
    data = load_data(filenames)
    boxplot(data,['LogReg','rf200','AdaBoost200'])
    plt.show()
