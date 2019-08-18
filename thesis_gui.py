import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import time

import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

class SampleApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.title("Sınıflandırma için Veri Setlerinin Farklı Makine Öğrenmesi Platformlarında Sınanması")
        self.geometry("850x450+350+150")
        self.title_font = tkfont.Font(family="Cambria", size=13, weight="bold", underline=1)
        self.button_font = tkfont.Font(family="Cambria", size=11)
        self.text_font = tkfont.Font(family="Consolas", size=10, weight="bold")
        #window.resizable(FALSE, FALSE)
        #window.state("zoomed") //full_screen
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label1 = tk.Label(self, font=controller.title_font, text="SINIFLANDIRMA İÇİN VERİ SETLERİNİN FARKLI MAKİNE ÖĞRENMESİ PLATFORMLARINDA SINANMASI")
        label1.place(x=25, y=60)

        button1 = tk.Button(self, text="Sınıflandırma İşlemleri", width=25,
                            relief="groove", font=controller.button_font, 
                            command=lambda: controller.show_frame("PageOne"))
        button2 = tk.Button(self, text="Veri Seti Bilgileri", width=25,
                            relief="groove", font=controller.button_font, 
                            command=lambda: controller.show_frame("PageTwo"))
        button1.place(x=25, y=140)
        button2.place(x=25, y=200)
        
        label2 = tk.Label(self, text="Başlangıç Sayfası")
        label2.place(x=390, y=420)


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label1 = tk.Label(self, font=controller.button_font,
                          text="Sınıflandırma işleminin yapılacağı veri setini seçiniz.")
        label1.place(x=25, y=60)
        dataset_list = ["Red Wine Quality", "Adult Census Income", 
                        "Indian Liver Patient Records", "Mammographic Mass", 
                        "Dresses Attribute Sales"]
        dataset_cbox = ttk.Combobox(self, values=dataset_list, width=30, 
                                    font=controller.button_font)
        dataset_cbox.place(x=360, y=60)
        dataset_cbox.set("...")
        models_list = [
            "Logistic Regression", "Linear Discriminant Analysis", "Support Vector Machine", 
            "Naive-Bayes Classifier", "K-Nearest Neighborhood", "Basic Decision Tree",
            "Bagged Tree", "Boosted Tree", "Random Forest"]

        models = [
            (LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)),
            (LinearDiscriminantAnalysis()),
            (SVC(random_state=0, gamma='scale', probability=True)),
            (GaussianNB()),
            (KNeighborsClassifier(n_neighbors=11, weights='distance')),
            (DecisionTreeClassifier(random_state=0)),
            (BaggingClassifier(random_state=0)),
            (GradientBoostingClassifier(random_state=0)),
            (RandomForestClassifier(random_state=0, n_estimators=100))
        ] #available for binary classification
        
        models2 = [
            (LogisticRegression(random_state=0, multi_class='multinomial', 
                                solver='lbfgs', max_iter=7000)),
            (LinearDiscriminantAnalysis(n_components=8)),
            (SVC(random_state=0, gamma='scale')),
            (GaussianNB()),
            (KNeighborsClassifier(n_neighbors=19, weights='distance')),
            (DecisionTreeClassifier(random_state=0)),
            (BaggingClassifier(random_state=0)),
            (GradientBoostingClassifier(random_state=0)),
            (RandomForestClassifier(random_state=0, n_estimators=100))
        ] #available for multiclass classification
        
        string_var2 = tk.StringVar()
        string_var3 = tk.StringVar()
        string_var4 = tk.StringVar()
        string_var5 = tk.StringVar()
        string_var6 = tk.StringVar()
        string_var7 = tk.StringVar()
        string_var8 = tk.StringVar()
        string_var9 = tk.StringVar()
        
        label2 = tk.Label(self, font=controller.button_font, textvariable=string_var2)
        label3 = tk.Label(self, font=controller.text_font, textvariable=string_var3)
        label4 = tk.Label(self, font=controller.text_font, textvariable=string_var4)
        label5 = tk.Label(self, font=controller.text_font, textvariable=string_var5)
        label6 = tk.Label(self, font=controller.text_font, textvariable=string_var6)
        label7 = tk.Label(self, font=controller.text_font, textvariable=string_var7)
        label8 = tk.Label(self, font=controller.text_font, textvariable=string_var8)
        label9 = tk.Label(self, font=controller.text_font, textvariable=string_var9)

        label2.place(x=25, y=90)
        label3.place(x=25, y=120)
        label4.place(x=25, y=140)
        label5.place(x=25, y=160)
        label6.place(x=25, y=200)
        label7.place(x=25, y=220)
        label8.place(x=25, y=260)
        label9.place(x=25, y=280)
        
        def callback1(eventObject):
            if(dataset_cbox.current()==0):
                dataset = pd.read_csv('winequality-red.csv')
                x = dataset.values[:, 0:11]
                y = dataset.values[:, 11]
                x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                train_size=0.7, test_size=0.3, random_state=0, stratify=y)
                string_var2.set("Sınıflandırma modelini seçiniz.")
                models_cbox = ttk.Combobox(self, values=models_list, width=30,
                                           font=controller.button_font)
                models_cbox.place(x=230, y=90)
                models_cbox.set("...")
                def run_callback(): #run button
                    for i in range (0,9):
                        if(models_cbox.current()==i):
                            start_time = time.time()
                            string_var3.set("Model Name                     Accuracy       CV Score")
                            classifier = models2[i]
                            classifier.fit(x_train, y_train)
                            test_result = classifier.predict(x_test)
                            accuracy = accuracy_score(test_result, y_test)
                            kfold = model_selection.KFold(n_splits=5)
                            cv_score = model_selection.cross_val_score(classifier, x, y, cv=kfold)
                            cm = (confusion_matrix(y_test, test_result))
                            string_var4.set("%-30s %-14.4f %-14.4f" % (models_list[i], accuracy, cv_score.mean()))
                            string_var5.set("runtime: %.4s sn" % (time.time() - start_time))
                            string_var6.set("Confusion Matrix ")
                            string_var7.set(str(cm))
                    i = i + 1
                    
                button = tk.Button(self, text="Çalıştır", relief="groove", 
                width=8, font=controller.button_font, command=run_callback)
                button.place(x=520, y=90)
            
            elif(dataset_cbox.current()==1):
                dataset = pd.read_csv("adult_census_income.csv")
                label_encoder = preprocessing.LabelEncoder()
                categorical = list(dataset.select_dtypes(include=['object']).columns.values)
                for cat in categorical:
                    dataset[cat] = label_encoder.fit_transform(dataset[cat].astype('category'))

                x = dataset.values[:, 0:14]
                y = dataset.values[:, 14]
                x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                train_size=0.7, test_size=0.3, random_state=0, stratify=y)
                string_var2.set("Sınıflandırma modelini seçiniz.")
                models_cbox = ttk.Combobox(self, values=models_list, width=30,
                                           font=controller.button_font)
                models_cbox.place(x=230, y=90)
                models_cbox.set("...")
                def run_callback(): #run button
                    for i in range (0,9):
                        if(models_cbox.current()==i):
                            start_time = time.time()
                            string_var3.set("Model Name                     Accuracy       CV Score       ROC-AUC Score")
                            classifier = models[i]
                            classifier.fit(x_train, y_train)
                            test_result = classifier.predict(x_test)
                            accuracy = accuracy_score(test_result, y_test)
                            kfold = model_selection.KFold(n_splits=5)
                            cv_score = model_selection.cross_val_score(classifier, x, y, cv=kfold)
                            roc_result = classifier.predict_proba(x_test)
                            roc_auc = roc_auc_score(y_test, roc_result[:, 1])
                            cm = (confusion_matrix(y_test, test_result))
                            string_var4.set("%-30s %-14.4f %-14.4f %.4f" % (models_list[i], accuracy, cv_score.mean(), roc_auc))
                            string_var5.set("runtime: %.4s sn" % (time.time() - start_time))
                            string_var6.set("Confusion Matrix ")
                            string_var7.set(str(cm))
                            string_var8.set("Sensitivity = %.2f" % (cm[1][1] / (cm[1][0] + cm[1][1])))
                            string_var9.set("Specifity = %.2f" % (cm[0][0] / (cm[0][0] + cm[0][1])))
                    i = i + 1
                    
                button = tk.Button(self, text="Çalıştır", relief="groove", 
                width=8, font=controller.button_font, command=run_callback)
                button.place(x=520, y=90)
                            
            elif(dataset_cbox.current()==2):
                dataset = pd.read_csv("indian_liver_patient.csv")
                label_encoder = preprocessing.LabelEncoder()
                dataset["Gender"] = label_encoder.fit_transform(dataset["Gender"].astype('category'))
                dataset["Dataset"] = label_encoder.fit_transform(dataset["Dataset"].astype('category'))
                dataset["Albumin_and_Globulin_Ratio"].fillna(0, inplace=True)
                x = dataset.values[:, 0:10]
                y = dataset.values[:, 10]
                x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                train_size=0.7, test_size=0.3, random_state=0, stratify=y)
                string_var2.set("Sınıflandırma modelini seçiniz.")
                models_cbox = ttk.Combobox(self, values=models_list, width=30,
                                           font=controller.button_font)
                models_cbox.place(x=230, y=90)
                models_cbox.set("...")
                def run_callback(): #run button
                    for i in range (0,9):
                        if(models_cbox.current()==i):
                            start_time = time.time()
                            string_var3.set("Model Name                     Accuracy       CV Score       ROC-AUC Score")
                            classifier = models[i]
                            classifier.fit(x_train, y_train)
                            test_result = classifier.predict(x_test)
                            accuracy = accuracy_score(test_result, y_test)
                            kfold = model_selection.KFold(n_splits=5)
                            cv_score = model_selection.cross_val_score(classifier, x, y, cv=kfold)
                            roc_result = classifier.predict_proba(x_test)
                            roc_auc = roc_auc_score(y_test, roc_result[:, 1])
                            cm = (confusion_matrix(y_test, test_result))
                            string_var4.set("%-30s %-14.4f %-14.4f %.4f" % (models_list[i], accuracy, cv_score.mean(), roc_auc))
                            string_var5.set("runtime: %.4s sn" % (time.time() - start_time))
                            string_var6.set("Confusion Matrix ")
                            string_var7.set(str(cm))
                            string_var8.set("Sensitivity = %.2f" % (cm[1][1] / (cm[1][0] + cm[1][1])))
                            string_var9.set("Specifity = %.2f" % (cm[0][0] / (cm[0][0] + cm[0][1])))
                    i = i + 1
                    
                button = tk.Button(self, text="Çalıştır", relief="groove", 
                width=8, font=controller.button_font, command=run_callback)
                button.place(x=520, y=90)
    
            elif(dataset_cbox.current()==3):
                dataset = pd.read_csv("mammographic_mass.csv")
                x = dataset.values[:, 0:5]
                y = dataset.values[:, 5]
                x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                train_size=0.7, test_size=0.3, random_state=0, stratify=y)
                string_var2.set("Sınıflandırma modelini seçiniz.")
                models_cbox = ttk.Combobox(self, values=models_list, width=30,
                                           font=controller.button_font)
                models_cbox.place(x=230, y=90)
                models_cbox.set("...")
                def run_callback(): #run button
                    for i in range (0,9):
                        if(models_cbox.current()==i):
                            start_time = time.time()
                            string_var3.set("Model Name                     Accuracy       CV Score       ROC-AUC Score")
                            classifier = models[i]
                            classifier.fit(x_train, y_train)
                            test_result = classifier.predict(x_test)
                            accuracy = accuracy_score(test_result, y_test)
                            kfold = model_selection.KFold(n_splits=5)
                            cv_score = model_selection.cross_val_score(classifier, x, y, cv=kfold)
                            roc_result = classifier.predict_proba(x_test)
                            roc_auc = roc_auc_score(y_test, roc_result[:, 1])
                            cm = (confusion_matrix(y_test, test_result))
                            string_var4.set("%-30s %-14.4f %-14.4f %.4f" % (models_list[i], accuracy, cv_score.mean(), roc_auc))
                            string_var5.set("runtime: %.4s sn" % (time.time() - start_time))
                            string_var6.set("Confusion Matrix ")
                            string_var7.set(str(cm))
                            string_var8.set("Sensitivity = %.2f" % (cm[1][1] / (cm[1][0] + cm[1][1])))
                            string_var9.set("Specifity = %.2f" % (cm[0][0] / (cm[0][0] + cm[0][1])))
                    i = i + 1
                    
                button = tk.Button(self, text="Çalıştır", relief="groove", 
                width=8, font=controller.button_font, command=run_callback)
                button.place(x=520, y=90)
    
            elif(dataset_cbox.current()==4):
                dataset = pd.read_csv("dresses_attribute_sales.csv")
                label_encoder = preprocessing.LabelEncoder()
                categorical = list(dataset.select_dtypes(include=['object']).columns.values)
                for cat in categorical:
                    dataset[cat].fillna('UNK', inplace=True)
                    dataset[cat] = label_encoder.fit_transform(dataset[cat].astype('category'))
            
                dataset["rating"] = label_encoder.fit_transform(dataset["rating"].astype('category'))
                x = dataset.values[:, 0:11]
                y = dataset.values[:, 11]
                x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                train_size=0.7, test_size=0.3, random_state=0, stratify=y)
                string_var2.set("Sınıflandırma modelini seçiniz.")
                models_cbox = ttk.Combobox(self, values=models_list, width=30,
                                           font=controller.button_font)
                models_cbox.place(x=230, y=90)
                models_cbox.set("...")
                def run_callback(): #run button
                    for i in range (0,9):
                        if(models_cbox.current()==i):
                            start_time = time.time()
                            string_var3.set("Model Name                     Accuracy       CV Score       ROC-AUC Score")
                            classifier = models[i]
                            classifier.fit(x_train, y_train)
                            test_result = classifier.predict(x_test)
                            accuracy = accuracy_score(test_result, y_test)
                            kfold = model_selection.KFold(n_splits=5)
                            cv_score = model_selection.cross_val_score(classifier, x, y, cv=kfold)
                            roc_result = classifier.predict_proba(x_test)
                            roc_auc = roc_auc_score(y_test, roc_result[:, 1])
                            cm = (confusion_matrix(y_test, test_result))
                            string_var4.set("%-30s %-14.4f %-14.4f %.4f" % (models_list[i], accuracy, cv_score.mean(), roc_auc))
                            string_var5.set("runtime: %.4s sn" % (time.time() - start_time))
                            string_var6.set("Confusion Matrix ")
                            string_var7.set(str(cm))
                            string_var8.set("Sensitivity = %.2f" % (cm[1][1] / (cm[1][0] + cm[1][1])))
                            string_var9.set("Specifity = %.2f" % (cm[0][0] / (cm[0][0] + cm[0][1])))
                    i = i + 1
                    
                button = tk.Button(self, text="Çalıştır", relief="groove", 
                width=8, font=controller.button_font, command=run_callback)
                button.place(x=520, y=90)
            
        dataset_cbox.bind("<<ComboboxSelected>>", callback1)

        button = tk.Button(self, text="Başlangıç sayfasına dönüş", width=25, 
                           font=controller.button_font, relief="groove",
                           command=lambda: controller.show_frame("StartPage"))
        button.place(x=25, y=350)
        
        label5 = tk.Label(self, text="Sınıflandırma İşlemleri Sayfası")
        label5.place(x=350, y=420)
        

class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label1 = tk.Label(self, font=controller.button_font, 
                          text="Bilgisine ulaşmak istediğiniz veri setini seçiniz.")
        label1.place(x=25, y=60)
        dataset_list = ["Red Wine Quality", "Adult Census Income", 
                        "Indian Liver Patient Records", "Mammographic Mass", 
                        "Dresses Attribute Sales"]
        dataset_cbox = ttk.Combobox(self, values=dataset_list, width=30, 
                                    font=controller.button_font)
        dataset_cbox.place(x=350, y=60)
        dataset_cbox.set("...")

        string_var2 = tk.StringVar()
        string_var3 = tk.StringVar()
        string_var4 = tk.StringVar()
        string_var5 = tk.StringVar()
        string_var6 = tk.StringVar()
        string_var7 = tk.StringVar()
        string_var8 = tk.StringVar()
        string_var9 = tk.StringVar()

        label2 = tk.Label(self, font=controller.text_font, textvariable=string_var2)
        label3 = tk.Label(self, font=controller.text_font, textvariable=string_var3)
        label4 = tk.Label(self, font=controller.text_font, textvariable=string_var4)
        label5 = tk.Label(self, font=controller.text_font, textvariable=string_var5)
        label6 = tk.Label(self, font=controller.text_font, textvariable=string_var6)
        label7 = tk.Label(self, font=controller.text_font, textvariable=string_var7)
        label8 = tk.Label(self, font=controller.text_font, textvariable=string_var8)
        label9 = tk.Label(self, font=controller.text_font, textvariable=string_var9)
        
        label2.place(x=25, y=90)
        label3.place(x=25, y=110)
        label4.place(x=25, y=130)
        label5.place(x=25, y=150)
        label6.place(x=25, y=170)
        label7.place(x=25, y=190)
        label8.place(x=25, y=210)
        label9.place(x=25, y=230)
        
        def callback2(eventObject):
            if (dataset_cbox.current()==0):
                string_var2.set("Başlık (Title): Red Wine Quality")
                string_var3.set("Kaynak (Source): Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV)")
                string_var4.set("Örnek Sayısı (Number of Instances): 1599")
                string_var5.set("Öznitelik Sayısı (Number of Features): 11")
                string_var6.set("Sınıf Sayısı (Number of Classes): 10")
                string_var7.set("Kayıp Veri (Missing Value): False")
                string_var8.set("Output: quality, [1,10]")
                string_var9.set("Sınıflandırma Hedefi (Classification Goal): Çeşitli kırmızı şarapların, içerdiği bazı özniteliklerin oranlarına\nbakarak şarabın kalitesini tahmin etmektir.")
     
            elif (dataset_cbox.current()==1):
                string_var2.set("Başlık (Title): Adult Census Income")
                string_var3.set("Kaynak (Source): Ronny Kohavi and Barry Becker, Data Mining and Visualization, Silicon Graphics. ")
                string_var4.set("Örnek Sayısı (Number of Instances): 32561")
                string_var5.set("Öznitelik Sayısı (Number of Features): 14")
                string_var6.set("Sınıf Sayısı (Number of Classes): 2")
                string_var7.set("Kayıp Veri (Missing Value): False")
                string_var8.set("Output: income, (<=50K || >50K)")
                string_var9.set("Sınıflandırma Hedefi (Classification Goal): Nüfus sayımında kullanılan özniteliklerin değerlerine göre kişinin\nyıllık gelirinin 50.000$’ı geçip geçmediğini tahmin etmektir.")
           
            elif (dataset_cbox.current()==2):
                string_var2.set("Başlık (Title): Indian Liver Patient Records")
                string_var3.set("Kaynak (Source): Lichman, M. (2013). Irvine, CA: University of California, School of Information and Computer Science")
                string_var4.set("Örnek Sayısı (Number of Instances): 583")
                string_var5.set("Öznitelik Sayısı (Number of Features): 10")
                string_var6.set("Sınıf Sayısı (Number of Classes): 2")
                string_var7.set("Kayıp Veri (Missing Value): True")
                string_var8.set("Output: Dataset, (1:Liver patient || 2:Not patient)")
                string_var9.set("Sınıflandırma Hedefi (Classification Goal): Hindistan’ın Andra Pradesh eyaletinden toplanan hasta kayıt\nverilerinden yola çıkarak ilgili kişilerin karaciğer hastası olup olmadığını tahmin etmektir.")

            elif (dataset_cbox.current()==3):
                string_var2.set("Başlık (Title): Mammographic Mass")
                string_var3.set("Kaynak (Source): Prof. Dr. Rüdiger Schulz-Wendtland, Gynaecological Radiology, University Erlangen-Nuremberg, Germany")
                string_var4.set("Örnek Sayısı (Number of Instances): 830")
                string_var5.set("Öznitelik Sayısı (Number of Features): 5")
                string_var6.set("Sınıf Sayısı (Number of Classes): 2")
                string_var7.set("Kayıp Veri (Missing Value): False")
                string_var8.set("Output: Severity, (0:benign || 1:malignant)")
                string_var9.set("Sınıflandırma Hedefi (Classification Goal): İlgili veri setindeki özniteliklerin değerlerine göre hastanın kitle\nlezyonunun ciddiyeti tahmin edilmeye çalışılmaktadır.")
                
            elif (dataset_cbox.current()==4):
                string_var2.set("Başlık (Title): Dresses Attribute Sales")
                string_var3.set("Kaynak (Source): Muhammad Usman & Adeel Ahmed, Air University, Students at Air University.")
                string_var4.set("Örnek Sayısı (Number of Instances): 475")
                string_var5.set("Öznitelik Sayısı (Number of Features): 11")
                string_var6.set("Sınıf Sayısı (Number of Classes): 2")
                string_var7.set("Kayıp Veri (Missing Value): True")
                string_var8.set("Output: recommendation, (0:not recommended || 1:recommended)")
                string_var9.set("Sınıflandırma Hedefi (Classification Goal): Elbise özelliklerine ve satışlarına göre elbisenin tavsiye edilip\nedilmediğini tahmin etmektir.")
          
        dataset_cbox.bind("<<ComboboxSelected>>", callback2)
   
        button = tk.Button(self, text="Başlangıç sayfasına dönüş", width=25, 
                           font=controller.button_font, relief="groove",
                           command=lambda: controller.show_frame("StartPage"))
        button.place(x=25, y=350)
        
        label10 = tk.Label(self, text="Veri Seti Bilgi Sayfası")
        label10.place(x=380, y=420)
        

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()