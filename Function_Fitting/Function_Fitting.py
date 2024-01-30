import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import Tk, Label, Button, Entry, filedialog, StringVar
from scipy.optimize import curve_fit
from scipy.optimize import minimize

# フィッティング関数の定義
def sine_function(x, A, B, C, D):
    return A * np.sin(2*np.pi / B * x - C) + D

def sinc_function(x, A, B, C, D):
    return A * np.sin(2*np.pi / B * (x + C)) / (x + C) + D

def sech_function(x, A, B, C, D):
    return A * (2 / np.cosh((x - C)/B)) + D
def sinc_fwhm(x, args):
    return (args[0] * np.sin(2*np.pi / args[1] * (x - args[2])) / (x - args[2]) + args[3]) - ((args[0] + args[3]) / 2)

class FittingApp:
    def __init__(self, root):
        self.root = root
        root.title("関数フィッティングアプリ")
        root.geometry('600x550')

        self.label_file = Label(root, text="ファイルのパス:")
        self.label_file.pack()
        
        self.file_path = StringVar()
        self.entry_file = Entry(root, textvariable=self.file_path)
        self.entry_file.pack()
        
        self.browse_button = Button(root, text="参照", command=self.load_file)
        self.browse_button.pack()

        self.label_x = Label(root, text="x軸の列番号:")
        self.label_x.pack()
        self.entry_x = Entry(root)
        self.entry_x.pack()

        self.label_y = Label(root, text="y軸の列番号:")
        self.label_y.pack()
        self.entry_y = Entry(root)
        self.entry_y.pack()
        
        self.label_xlabel = Label(root, text="x軸のラベル:")
        self.label_xlabel.pack()
        self.xlabel = StringVar()
        self.entry_xlabel = Entry(root, textvariable=self.xlabel)
        self.entry_xlabel.pack()

        self.label_ylabel = Label(root, text="y軸のラベル:")
        self.label_ylabel.pack()
        self.ylabel = StringVar()
        self.entry_ylabel = Entry(root, textvariable=self.ylabel)
        self.entry_ylabel.pack()

        self.label_func = Label(root, text="関数の選択 (sine or sech):")
        self.label_func.pack()
        self.function_choice = StringVar()
        self.entry_func = Entry(root, textvariable=self.function_choice)
        self.entry_func.pack()
        
        self.label_initial_A = Label(root, text="振幅Aの初期値:")
        self.label_initial_A.pack()
        self.initial_A = StringVar(value="1")  # デフォルト値として1を設定
        self.entry_initial_A = Entry(root, textvariable=self.initial_A)
        self.entry_initial_A.pack()

        self.label_initial_B = Label(root, text="周期Bの初期値:")
        self.label_initial_B.pack()
        self.initial_B = StringVar(value="1")
        self.entry_initial_B = Entry(root, textvariable=self.initial_B)
        self.entry_initial_B.pack()

        self.label_initial_C = Label(root, text="位相Cの初期値:")
        self.label_initial_C.pack()
        self.initial_C = StringVar(value="1")
        self.entry_initial_C = Entry(root, textvariable=self.initial_C)
        self.entry_initial_C.pack()

        self.label_initial_D = Label(root, text="バイアス値Dの初期値:")
        self.label_initial_D.pack()
        self.initial_D = StringVar(value="1")
        self.entry_initial_D = Entry(root, textvariable=self.initial_D)
        self.entry_initial_D.pack()

        self.fit_button = Button(root, text="フィット", command=self.fit_and_plot)
        self.fit_button.pack()
        
        self.params_label = Label(root, text="フィットのパラメータ: 未計算")
        self.params_label.pack()
        self.params_r2 = Label(root, text="決定係数: 未計算")
        self.params_r2.pack()
        self.fwhm_label = Label(root,text="FWHM :未計算")
        self.fwhm_label.pack()
        

    def load_file(self):
        file_path = filedialog.askopenfilename()
        self.file_path.set(file_path)

    def fit_and_plot(self):
        # ファイルからデータを読み込み
        data = pd.read_csv(self.file_path.get(),delimiter=",")
        x_data = data.iloc[:, int(self.entry_x.get())]
        y_data = data.iloc[:, int(self.entry_y.get())]
        
        # 初期値を取得
        initial_values = (
            float(self.initial_A.get()),
            float(self.initial_B.get()),
            float(self.initial_C.get()),
            float(self.initial_D.get())
        )

        # フィッティング関数の選択
        if self.function_choice.get() == "sine":
            popt, _ = curve_fit(sine_function, x_data, y_data, p0=initial_values)
            y_fit = sine_function(x_data, *popt)
            self.function_form = "y = "+ str(round(popt[0],2)) +  "* sin((2*pi /" + str(round(popt[1],2)) + ") x + " \
                + str(round(popt[2],2)) + ") + " + str(round(popt[3],2))
                
            #決定係数を計算
            residuals =  y_data - sine_function(x_data, popt[0], popt[1], popt[2], popt[3])
            rss = np.sum(residuals**2)#residual sum of squares = rss
            tss = np.sum((y_data-np.mean(y_data))**2)#total sum of squares = tss
            r_squared = 1 - (rss / tss)
            
        elif self.function_choice.get() == "sech":
            popt, _ = curve_fit(sech_function, x_data, y_data, p0=initial_values)
            self.function_form = "y = "+ str(round(popt[0],4)) +  "* sech((x - " \
                + str(round(popt[2],2)) + ") / " + str(round(popt[1],2)) + ") + " + str(round(popt[3],4))
                
            #決定係数を計算    
            residuals =  y_data - sinc_function(x_data, popt[0], popt[1], popt[2], popt[3])
            rss = np.sum(residuals**2)#residual sum of squares = rss
            tss = np.sum((y_data-np.mean(y_data))**2)#total sum of squares = tss
            r_squared = 1 - (rss / tss)
            
            #半値全幅測定用
            y_fit = sech_function(x_data, *popt)
            fwhm = (2*popt[1]) * np.abs(np.log(2+np.sqrt(3)) - np.log(2-np.sqrt(3)))
            print(fwhm)
            self.fwhm_label.config(text="FWHM :" + str(fwhm))
        else:
            return
        
        # フィッティング結果のパラメータを表示
        params_text = f"A: {popt[0]:.5f}, B: {popt[1]:.5f}, C: {popt[2]:.5f}, D: {popt[3]:.5f}"
        self.params_label.config(text="フィットのパラメータ: " + params_text)
        self.params_r2.config(text="決定係数: " + str(r_squared))

        # 結果の描画
        plt.scatter(x_data, y_data, label="Data")
        plt.plot(x_data, y_fit, label="Fit : " + self.function_form, color="red")
        plt.xlabel(self.xlabel.get())
        plt.ylabel(self.ylabel.get())
        plt.legend()
        plt.show()
        

if __name__ == "__main__":
    root = Tk()
    app = FittingApp(root)
    root.mainloop()
