from df1 import df1
import numpy as np

if __name__ == "__main__":
    y_true = np.array([0,1,1,0])
    y_pred = np.array([0,1,0,1])
    x = np.array(["negative text", "positive text"])
    y = np.array(["negative text", "positive text", "this text is positive", "this text is definitely negative"])

    f1 = df1(y_true, y_pred, x, y, average="micro", train_n = 2, model_card = 'all-MiniLM-L6-v2')
    print(f1)
