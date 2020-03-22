import matplotlib.pyplot as plt

def plot_model_history(model):
    plt.plot(model.history.history["loss"], label="loss")
    plt.plot(model.history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.show()
    plt.close()

    plt.plot(model.history.history["accuracy"], label="accuracy")
    plt.plot(model.history.history["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.show()
    plt.close()