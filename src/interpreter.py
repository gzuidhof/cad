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
    # for file in data:
    #     print np.mean(file),np.std(file)

if __name__ == "__main__":
    filenames = ['logreg_dice.pkl','rf150_dice.pkl','adaboost_dice.pkl']
    data = load_data(filenames)
    boxplot(data,['LogReg','rf150','AdaBoost'])
    plt.show()