import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from scipy.ndimage import gaussian_filter

class WTENCV():
    def __init__(
        self,
        wave_numbers,
        wavelet_num=10,
        proportion=1,
        scaling=True,
        cv_fold=5,
        max_iter=10,
        random_state=1,
    ):
        self.GAUSSIAN_RICKER_WIDTH_RATIO = 0.45

        self.wave_numbers = wave_numbers
        self.len_ir_spectrum = len(wave_numbers)
        self.wavelet_num = wavelet_num
        self.proportion = proportion
        self.scaling = scaling
        self.cv_fold = cv_fold
        self.max_iter = max_iter
        self.random_state = random_state   

        self.mms = MinMaxScaler()
        self.model = ElasticNetCV(
            cv=self.cv_fold,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

    def calc_wavelet_positive_results(self, arr):
        arr_width = arr.shape[1]
        wavelet_widths = ((arr_width/self.proportion)**(1/(self.wavelet_num-1))) ** np.arange(self.wavelet_num)
        X = np.array([np.ravel(signal.cwt(x, signal.ricker, wavelet_widths)) for x in arr])
        X[X<0] = 0

        return X

    def preprocessing(self, X):
        if self.wavelet_num != 0:
            X = np.hstack((
                X,
                self.calc_wavelet_positive_results(X)
            ))
        elif self.wavelet_num == 0:
            pass
        else:
            print("Wavelet_num must be a natural number greater than or equal to 0.")

        return X

    def fit(self, X, y):
        X = self.preprocessing(X)
        if self.scaling:
            X = self.mms.fit_transform(X)
        else:
            pass
        
        self.model.fit(X, y)
        self.coef_ = self.model.coef_

    def predict(self, X):
        X = self.preprocessing(X)
        if self.scaling:
            X = self.mms.transform(X)
        else:
            pass

        y_pred = self.model.predict(X)

        return y_pred


    def visualize_coef(self, title=None, color="black", color_intensity_parameter=0.3):
        if self.wavelet_num != 0:
            wavelet_widths = (
                (self.len_ir_spectrum/self.proportion)**(1/(self.wavelet_num-1))
            ) ** (
                np.arange(self.wavelet_num)
            )

        elif self.wavelet_num == 0:
            wavelet_widths = np.ones(1)

        plt.figure(figsize=(15,4))
        if title is not None:
            plt.title(title)

        reshaped_coef_ = self.coef_.reshape(self.wavelet_num+1, self.len_ir_spectrum)
        arr = np.zeros_like(reshaped_coef_)
        for j in range(self.wavelet_num):
             arr[j]= gaussian_filter(reshaped_coef_[j], sigma=wavelet_widths[j] * self.GAUSSIAN_RICKER_WIDTH_RATIO)

        for s,p in zip(arr, (wavelet_widths * self.GAUSSIAN_RICKER_WIDTH_RATIO).T):
            if p==0:
                plt.plot(self.wave_numbers, s, color = color, alpha=1.0)
            else:
                plt.plot(
                    self.wave_numbers,
                    s * np.sqrt(2 * np.pi * p**2),
                    color=color,
                    alpha=min(1, color_intensity_parameter + 1/(p + 1e-13))
                )
        plt.gca().invert_xaxis()
        plt.yticks(ticks=[])

        plt.show()
