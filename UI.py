from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
global choosen_btn_style
global normal_btn_style
global file2


class Ui_MainWindow(object):
	choosen_btn_style ='''    
							font: 75 13pt "Orbitron";
							background-color: rgb(255, 97, 100) ; 
							color: white; 
							margin: 1px 0px 1px 0px;
							border: 1px transparent #2A2929; 
							border-radius: 30px;   
						 '''
	normal_btn_style = '''    
							font: 75 13pt "Orbitron";
							background-color: rgb(172, 172, 172) ; 
							color: white; 
							margin: 1px 0px 1px 0px;
							border: 1px transparent #2A2929; 
							border-radius: 30px;   		
						 '''

#-------------------------------------------------------
#---------------------- METHODS ------------------------
#-------------------------------------------------------
	def enable_btn(self,b1,b2,b3,b4,b5,b6,b7):
		self.upload_file_btn.setEnabled(b1)
		self.extract_features_btn.setEnabled(b2)
		self.training_btn.setEnabled(b3)
		self.test_btn.setEnabled(b4)
		self.report_btn.setEnabled(b5)
		self.confusion_matrix_btn.setEnabled(b6)
		self.wordcloud_btn.setEnabled(b7)


	def enable_btn2(self,b1,b2,b3,b4,b5,b6,b7):
		self.upload_file_btn2.setEnabled(b1)
		self.extract_features_btn2.setEnabled(b2)
		self.training_btn2.setEnabled(b3)
		self.test_btn2.setEnabled(b4)
		self.confusion_matrix_btn2.setEnabled(b5)
		self.report_btn2.setEnabled(b6)
		self.wordcloud_btn2.setEnabled(b7)

	def enable_btn3(self,b1,b2,b3,b4,b5,b6,b7):
		self.upload_file_btn3.setEnabled(b1)
		self.extract_features_btn3.setEnabled(b2)
		self.training_btn3.setEnabled(b3)
		self.test_btn3.setEnabled(b4)
		self.confusion_matrix_btn3.setEnabled(b5)
		self.report_btn3.setEnabled(b6)
		self.wordcloud_btn3.setEnabled(b7)


	def browse_file(self):
		self.enable_btn(True,True,False,False,False,False,False)
		file_name = QFileDialog.getOpenFileName()
		if file_name is not None:
			path = file_name[0]
			print("your path is : ",path)
		else:
			msg = QMessageBox()
			msg.setWindowTitle("Read ERROR")
			msg.setIcon(QMessageBox.Critical)
			msg.setText("Invalid File!")
			x = msg.exec_()
		self.enable_btn(True,True,False,False,False,False,False)
		print ("here")
		# read origin file
		file = pd.read_csv(path, usecols=['Tweet', 'followers', 'following', 'is_retweet', 'actions', 'Type'])
		fileData = file.dropna()  # remove rows that have infill value
		print(file.isna().sum())
		newfile = file.copy()  # Create duplicate of data
		newfile.dropna(inplace=True)  # Remove rows with NaN
		newfile["Ratio"] = newfile["followers"] / (newfile["following"] + newfile["followers"]);
		newfile.fillna(0, inplace=True)
		file2 = newfile.to_csv("NewVersion.csv")




	def browse_file2(self):
		file_name = QFileDialog.getOpenFileName()
		if file_name is not None:
			path = file_name[0]
			print("your path is : ",path)
		else:
			msg = QMessageBox()
			msg.setWindowTitle("Read ERROR")
			msg.setIcon(QMessageBox.Critical)
			msg.setText("Invalid File!")
			x = msg.exec_()
		self.enable_btn2(True,True,False,False,False,False,False)
		print ("here")
		# read origin file
		file = pd.read_csv(path, usecols=['Tweet', 'followers', 'following', 'is_retweet', 'actions', 'Type'])
		fileData = file.dropna()  # remove rows that have infill value
		print(file.isna().sum())
		newfile = file.copy()  # Create duplicate of data
		newfile.dropna(inplace=True)  # Remove rows with NaN
		newfile["Ratio"] = newfile["followers"] / (newfile["following"] + newfile["followers"]);
		newfile.fillna(0, inplace=True)
		file2 = newfile.to_csv("NewVersion.csv")





	def browse_file3(self):
		file_name = QFileDialog.getOpenFileName()
		if file_name is not None:
			path = file_name[0]
			print("your path is : ",path)
		else:
			msg = QMessageBox()
			msg.setWindowTitle("Read ERROR")
			msg.setIcon(QMessageBox.Critical)
			msg.setText("Invalid File!")
			x = msg.exec_()
		self.enable_btn3(True,True,False,False,False,False,False)
		print ("here")
		# read origin file
		file = pd.read_csv(path, usecols=['Tweet', 'followers', 'following', 'is_retweet', 'actions', 'Type'])
		fileData = file.dropna()  # remove rows that have infill value
		print(file.isna().sum())
		newfile = file.copy()  # Create duplicate of data
		newfile.dropna(inplace=True)  # Remove rows with NaN
		newfile["Ratio"] = newfile["followers"] / (newfile["following"] + newfile["followers"]);
		newfile.fillna(0, inplace=True)
		file2 = newfile.to_csv("NewVersion.csv")

	def extract_features (self):
		print("work2")
		self.extract_features_btn.setStyleSheet(self.choosen_btn_style)
		self.enable_btn(True,True,False,False,False,False,True)
		self.extract_features_btn.setStyleSheet(self.choosen_btn_style)
		self.training_btn.setStyleSheet(self.choosen_btn_style)
		self.test_btn.setStyleSheet(self.choosen_btn_style)
		self.confusion_matrix_btn.setStyleSheet(self.choosen_btn_style)
		self.report_btn.setStyleSheet(self.choosen_btn_style)

		fileData = pd.read_csv("NewVersion.csv", usecols=['Tweet', 'actions', 'Ratio', 'Type']);
		len = fileData.__len__()
		stem = WordNetLemmatizer()  # take the orgin of the words
		list = []

		for i in range(0, len):
			line = re.sub('[^a-zA-Z]', " ", fileData['Tweet'][i])  # replace not digit vlaues to space
			line = line.lower()  # change it to lower case
			line = line.split()  # similar to token
			line = [stem.lemmatize(word) for word in line if
					not word in set(stopwords.words('english'))]  # remove stop words from the tweet
			line = " ".join(line)
			list.append(line)

		# Sparse Matrix
		vectorizer = CountVectorizer()  # to convert sentence to numbers
		data = vectorizer.fit_transform(list).toarray()
		data = np.column_stack((data, fileData["actions"].values, fileData["Ratio"].values));
		print(data[0])
		print(type(data))
		label = fileData['Type'].values

		# split test and train into 20-80%
		data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2);

		# naive baise classfier
		model = GaussianNB()
		final = model.fit(data_train, label_train)  # concat between vector tweet and labels
		print("Testing score", final.score(data_test, label_test))  # calculate accuracy
		# heat map for naive bais
		svm_predicted = final.predict(data_test)
		svm_confuse = confusion_matrix(label_test, svm_predicted)
		df_cm = pd.DataFrame(svm_confuse)
		plt.figure(figsize=(5, 3.5))
		sb.heatmap(df_cm, annot=True, fmt='g')
		plt.figure(1)
		plt.title("Confusion Matrix Heatmap")
		plt.xlabel("True Label")
		plt.ylabel("Predicted Label")

		# to report other values
		print("Classification Report")
		print(classification_report(label_test, svm_predicted))

		msg = QMessageBox()
		msg.setWindowTitle("report")
		msg.setText(classification_report(label_test, svm_predicted))
		x = msg.exec_()

	def extract_features2 (self):
		print("work2")
		self.extract_features_btn2.setStyleSheet(self.choosen_btn_style)
		self.enable_btn2(True,True,False,False,False,False,True)
		self.extract_features_btn2.setStyleSheet(self.choosen_btn_style)
		self.training_btn2.setStyleSheet(self.choosen_btn_style)
		self.test_btn2.setStyleSheet(self.choosen_btn_style)
		self.confusion_matrix_btn2.setStyleSheet(self.choosen_btn_style)
		self.report_btn2.setStyleSheet(self.choosen_btn_style)

		fileData = pd.read_csv("NewVersion.csv", usecols=['Tweet', 'actions', 'Ratio', 'Type']);
		len = fileData.__len__()
		stem = WordNetLemmatizer()  # take the orgin of the words
		list = []

		for i in range(0, len):
			line = re.sub('[^a-zA-Z]', " ", fileData['Tweet'][i])  # replace not digit vlaues to space
			line = line.lower()  # change it to lower case
			line = line.split()  # similar to token
			line = [stem.lemmatize(word) for word in line if
					not word in set(stopwords.words('english'))]  # remove stop words from the tweet
			line = " ".join(line)
			list.append(line)

		# Sparse Matrix
		vectorizer = CountVectorizer()  # to convert sentence to numbers
		data = vectorizer.fit_transform(list).toarray()
		data = np.column_stack((data, fileData["actions"].values, fileData["Ratio"].values));
		print(data[0])
		print(type(data))
		label = fileData['Type'].values

		# split test and train into 20-80%
		data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2);

		# random forest classifier
		seed = 4353
		rfc = RandomForestClassifier(n_estimators=10, random_state=seed)
		rfc.fit(data_train, label_train)
		predictions = rfc.predict(data_test)
		print(classification_report(label_test, predictions))
		print(confusion_matrix(label_test, predictions))

		# heat map for random forest
		svm_predicted = rfc.predict(data_test)
		svm_confuse = confusion_matrix(label_test, svm_predicted)
		df_cm = pd.DataFrame(svm_confuse)
		plt.figure(figsize=(5, 3.5))
		sb.heatmap(df_cm, annot=True, fmt='g')
		plt.figure(2)
		plt.title("Confusion Matrix Heatmap")
		plt.xlabel("True Label")
		plt.ylabel("Predicted Label")

		msg = QMessageBox()
		msg.setWindowTitle("report")
		msg.setText(classification_report(label_test, predictions))
		x = msg.exec_()


	def extract_features3 (self):
		print("work2")
		self.extract_features_btn3.setStyleSheet(self.choosen_btn_style)
		self.enable_btn3(True,True,False,False,False,False,True)
		self.extract_features_btn3.setStyleSheet(self.choosen_btn_style)
		self.training_btn3.setStyleSheet(self.choosen_btn_style)
		self.test_btn3.setStyleSheet(self.choosen_btn_style)
		self.confusion_matrix_btn3.setStyleSheet(self.choosen_btn_style)
		self.report_btn3.setStyleSheet(self.choosen_btn_style)

		fileData = pd.read_csv("NewVersion.csv", usecols=['Tweet', 'actions', 'Ratio', 'Type']);
		len = fileData.__len__()
		stem = WordNetLemmatizer()  # take the orgin of the words
		list = []

		for i in range(0, len):
			line = re.sub('[^a-zA-Z]', " ", fileData['Tweet'][i])  # replace not digit vlaues to space
			line = line.lower()  # change it to lower case
			line = line.split()  # similar to token
			line = [stem.lemmatize(word) for word in line if
					not word in set(stopwords.words('english'))]  # remove stop words from the tweet
			line = " ".join(line)
			list.append(line)

		# Sparse Matrix
		vectorizer = CountVectorizer()  # to convert sentence to numbers
		data = vectorizer.fit_transform(list).toarray()
		data = np.column_stack((data, fileData["actions"].values, fileData["Ratio"].values));
		print(data[0])
		print(type(data))
		label = fileData['Type'].values

		# split test and train into 20-80%
		data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2);

		# neural network multi layer perceptron
		print("Neural network")
		mlp = MLPClassifier()
		mlp.fit(data_train, label_train)
		print("Activation function=" + mlp.activation)
		expected_y = label_test
		predicted_y = mlp.predict(data_test)
		print(metrics.classification_report(expected_y, predicted_y))
		print(metrics.confusion_matrix(expected_y, predicted_y))

		# heat map of neural network
		svm_predicted = mlp.predict(data_test)
		svm_confuse = confusion_matrix(label_test, svm_predicted)
		df_cm = pd.DataFrame(svm_confuse)
		plt.figure(figsize=(5, 3.5))
		sb.heatmap(df_cm, annot=True, fmt='g')
		plt.figure(3)
		plt.title("Confusion Matrix Heatmap")
		plt.xlabel("True Label")
		plt.ylabel("Predicted Label")
		plt.show()

		msg = QMessageBox()
		msg.setWindowTitle("report")
		msg.setText(metrics.classification_report(expected_y, predicted_y))
		x = msg.exec_()



	def Show_wordclouds(self):
		print("work7")
		self.wordcloud_btn.setStyleSheet(self.choosen_btn_style)
		fileData = pd.read_csv("train.csv", usecols=['Tweet', 'Type'])
		spam = fileData[fileData['Type'] == 'Spam'].to_csv("fake.csv")
		df = pd.read_csv("fake.csv")  # Importing Dataset
		df.dropna(inplace=True)  # Removing NaN Values
		text = " ".join(cat.split()[0] for cat in df.Tweet)  # Creating the text variable
		wordcloud = WordCloud(collocations=False, background_color='white').generate(
			text)  # Creating word_cloud with text as argument in .generate() method
		plt.figure(1)  # Display the generated Word Cloud
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.title("Spam tweets")
		plt.axis("off")

		Quality = fileData[fileData['Type'] == 'Quality'].to_csv("true.csv")
		df1 = pd.read_csv("true.csv")
		df1.dropna(inplace=True)  # Removing NaN Values
		text1 = " ".join(cat.split()[0] for cat in df1.Tweet)  # Creating the text variable
		wordcloud1 = WordCloud(collocations=False, background_color='black').generate(
			text1)  # Creating word_cloud with text as argument in .generate() method
		plt.figure(2)  # Display the generated Word Cloud
		plt.imshow(wordcloud1, interpolation='bilinear')
		plt.axis("off")
		plt.title("Quality tweets")
		plt.show()




	def Show_wordclouds2(self):
		print("work7")
		self.wordcloud_btn2.setStyleSheet(self.choosen_btn_style)

		fileData = pd.read_csv("train.csv", usecols=['Tweet', 'Type'])
		spam = fileData[fileData['Type'] == 'Spam'].to_csv("fake.csv")
		df = pd.read_csv("fake.csv")  # Importing Dataset
		df.dropna(inplace=True)  # Removing NaN Values
		text = " ".join(cat.split()[0] for cat in df.Tweet)  # Creating the text variable
		wordcloud = WordCloud(collocations=False, background_color='white').generate(
			text)  # Creating word_cloud with text as argument in .generate() method
		plt.figure(1)  # Display the generated Word Cloud
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.title("Spam tweets")
		plt.axis("off")

		Quality = fileData[fileData['Type'] == 'Quality'].to_csv("true.csv")
		df1 = pd.read_csv("true.csv")
		df1.dropna(inplace=True)  # Removing NaN Values
		text1 = " ".join(cat.split()[0] for cat in df1.Tweet)  # Creating the text variable
		wordcloud1 = WordCloud(collocations=False, background_color='black').generate(
			text1)  # Creating word_cloud with text as argument in .generate() method
		plt.figure(2)  # Display the generated Word Cloud
		plt.imshow(wordcloud1, interpolation='bilinear')
		plt.axis("off")
		plt.title("Quality tweets")
		plt.show()


	def Show_wordclouds3(self):
		print("work7")
		self.wordcloud_btn3.setStyleSheet(self.choosen_btn_style)

		fileData = pd.read_csv("train.csv", usecols=['Tweet', 'Type'])
		spam = fileData[fileData['Type'] == 'Spam'].to_csv("fake.csv")
		df = pd.read_csv("fake.csv")  # Importing Dataset
		df.dropna(inplace=True)  # Removing NaN Values
		text = " ".join(cat.split()[0] for cat in df.Tweet)  # Creating the text variable
		wordcloud = WordCloud(collocations=False, background_color='white').generate(
			text)  # Creating word_cloud with text as argument in .generate() method
		plt.figure(1)  # Display the generated Word Cloud
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.title("Spam tweets")
		plt.axis("off")

		Quality = fileData[fileData['Type'] == 'Quality'].to_csv("true.csv")
		df1 = pd.read_csv("true.csv")
		df1.dropna(inplace=True)  # Removing NaN Values
		text1 = " ".join(cat.split()[0] for cat in df1.Tweet)  # Creating the text variable
		wordcloud1 = WordCloud(collocations=False, background_color='black').generate(
			text1)  # Creating word_cloud with text as argument in .generate() method
		plt.figure(2)  # Display the generated Word Cloud
		plt.imshow(wordcloud1, interpolation='bilinear')
		plt.axis("off")
		plt.title("Quality tweets")
		plt.show()



	def reset_btns(self):
		self.enable_btn(True,False,False,False,False,False,False)
		self.enable_btn2(True,False,False,False,False,False,False)
		self.enable_btn3(True,False,False,False,False,False,False)
		self.extract_features_btn.setStyleSheet(self.normal_btn_style)
		self.extract_features_btn2.setStyleSheet(self.normal_btn_style)
		self.extract_features_btn3.setStyleSheet(self.normal_btn_style)
		self.training_btn.setStyleSheet(self.normal_btn_style)
		self.training_btn2.setStyleSheet(self.normal_btn_style)
		self.training_btn3.setStyleSheet(self.normal_btn_style)
		self.test_btn.setStyleSheet(self.normal_btn_style)
		self.test_btn2.setStyleSheet(self.normal_btn_style)
		self.test_btn3.setStyleSheet(self.normal_btn_style)
		self.confusion_matrix_btn.setStyleSheet(self.normal_btn_style)
		self.confusion_matrix_btn2.setStyleSheet(self.normal_btn_style)
		self.confusion_matrix_btn3.setStyleSheet(self.normal_btn_style)
		self.wordcloud_btn.setStyleSheet(self.normal_btn_style)
		self.wordcloud_btn2.setStyleSheet(self.normal_btn_style)
		self.wordcloud_btn3.setStyleSheet(self.normal_btn_style)
		self.report_btn.setStyleSheet(self.normal_btn_style)
		self.report_btn2.setStyleSheet(self.normal_btn_style)
		self.report_btn3.setStyleSheet(self.normal_btn_style)



#-------------------------------------------------------- START
# -------------------------------------------------------- START

	def setupUi(self, MainWindow):

		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1000, 680)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.tabs = QtWidgets.QTabWidget(self.centralwidget)
		self.tabs.setGeometry(QtCore.QRect(0, 0, 1000, 680))
		StyleSheet = '''
		QTabWidget {
		    background-color: #00b3b3;
		    height: 30px; 
		    width: 500px;
		    font-size:20px;
		}
		QTabWidget::pane {
   	 		border: 1px solid #31363B;
    		padding: 2px;
    		margin:  0px;
    		background: url(background.png)
		}
		QTabBar {
		    background-color: #004d66;		
		    border: 4px solid   #004d66;
		    color: #ffffff;
		}
		QTabBar::tab{
			height: 30px; 
		    width: 333px;
		}
		QTabBar::tab:top:selected {
		    color: #ff6164;
		    background-color: #ffffff;	
		}
		'''
		self.tabs.setStyleSheet(StyleSheet)

		self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
		self.tabs.setTabShape(QtWidgets.QTabWidget.Triangular)
		self.tabs.setIconSize(QtCore.QSize(20, 20))
		self.tabs.setElideMode(QtCore.Qt.ElideMiddle)
		self.tabs.setDocumentMode(False)
		self.tabs.setTabsClosable(False)
		self.tabs.setMovable(True)
		self.tabs.setTabBarAutoHide(False)
		self.tabs.setObjectName("tabs")
		self.tab_1 = QtWidgets.QWidget()
		self.tab_1.setEnabled(True)
		self.tab_1.setAutoFillBackground(False)
		self.tab_1.setObjectName("tab_1")



# --------------------------------------------------------------------------------------------------------
		self.upload_file_btn = QtWidgets.QPushButton(self.tab_1)                    ##1
		self.upload_file_btn.setGeometry(QtCore.QRect(200, 70, 150, 91))
		self.upload_file_btn.setAutoFillBackground(False)
		self.upload_file_btn.setStyleSheet('''    
								font: 75 15pt "Orbitron";
							    background-color: rgb(255, 97, 100) ; 
							    color: white; 
							    height: 60px;
							    width: 50px;
							    margin: 1px 0px 1px 0px;
							    border: 1px transparent #2A2929; 
							    border-radius: 40px;   
							     ''' )
		self.upload_file_btn.setObjectName("upload_file_btn")
#--------------------------------------------------------------------------------------------------------
		self.extract_features_btn = QtWidgets.QPushButton(self.tab_1)                      ##2
		self.extract_features_btn.setGeometry(QtCore.QRect(40, 260, 220, 91))
		self.extract_features_btn.setAutoFillBackground(False)
		self.extract_features_btn.setStyleSheet(self.normal_btn_style)

		self.extract_features_btn.setObjectName("extract_features_btn")
# --------------------------------------------------------------------------------------------------------
		self.training_btn = QtWidgets.QPushButton(self.tab_1)                             ##3
		self.training_btn.setGeometry(QtCore.QRect(160, 470, 150, 91))
		self.training_btn.setAutoFillBackground(False)
		self.training_btn.setStyleSheet(self.normal_btn_style)
		self.training_btn.setObjectName("training_btn")
#---------------------------------------------------------------------------------------------

		self.test_btn = QtWidgets.QPushButton(self.tab_1)                        ##4
		self.test_btn.setGeometry(QtCore.QRect(390, 535, 171, 81))
		self.test_btn.setAutoFillBackground(False)
		self.test_btn.setStyleSheet(self.normal_btn_style)
		self.test_btn.setObjectName("test_btn")
# --------------------------------------------------------------------------------------------------------

		self.confusion_matrix_btn = QtWidgets.QPushButton(self.tab_1)               #5
		self.confusion_matrix_btn.setGeometry(QtCore.QRect(655, 460, 171, 91))
		self.confusion_matrix_btn.setAutoFillBackground(False)
		self.confusion_matrix_btn.setStyleSheet(self.normal_btn_style)
		self.confusion_matrix_btn.setObjectName("confusion_matrix_btn")
# --------------------------------------------------------------------------------------------------------
		self.report_btn = QtWidgets.QPushButton(self.tab_1)                          ##6
		self.report_btn.setGeometry(QtCore.QRect(750, 260, 171, 91))
		self.report_btn.setAutoFillBackground(False)
		self.report_btn.setStyleSheet(self.normal_btn_style)
		self.report_btn.setObjectName("report_btn")

		# --------------------------------------------------------------------------------------------------------

		self.wordcloud_btn = QtWidgets.QPushButton(self.tab_1)                            ##7
		self.wordcloud_btn.setGeometry(QtCore.QRect(690, 75, 161, 91))
		self.wordcloud_btn.setAutoFillBackground(False)
		self.wordcloud_btn.setStyleSheet(self.normal_btn_style)
		self.wordcloud_btn.setObjectName("wordcloud_btn")
        # --------------------------------------------------------------------------------------------------------
		self.reset = QtWidgets.QPushButton(self.tab_1)
		self.reset.setGeometry(QtCore.QRect(910, 565, 80, 60))
		self.reset.setAutoFillBackground(False)
		self.reset.setStyleSheet(self.normal_btn_style)
		self.reset.setObjectName("reset")
		# --------------------------------------------------------------------------------------------------------

		self.extract_features_btn.raise_()
		self.training_btn.raise_()
		self.wordcloud_btn.raise_()
		self.upload_file_btn.raise_()
		self.test_btn.raise_()
		self.confusion_matrix_btn.raise_()
		self.report_btn.raise_()
# --------------------------------------------------------------------------------------------------------

		self.tabs.addTab(self.tab_1, "")
		self.tab_2 = QtWidgets.QWidget()
		self.tab_2.setObjectName("tab_2")

		self.background_label = QtWidgets.QLabel(self.tab_2)
		self.background_label.setGeometry(QtCore.QRect(0, 0, 995, 632))
		self.background_label.setText("")
		self.background_label.setPixmap(QtGui.QPixmap("background2.png"))
		self.background_label.setScaledContents(True)
		self.background_label.setObjectName("background_label")
		# --------------------------------------------------------------------------------------------------------
		self.upload_file_btn2 = QtWidgets.QPushButton(self.tab_2)  ##1
		self.upload_file_btn2.setGeometry(QtCore.QRect(200, 70, 150, 91))
		self.upload_file_btn2.setAutoFillBackground(False)
		self.upload_file_btn2.setStyleSheet('''    
										font: 75 15pt "Orbitron";
									    background-color: rgb(255, 97, 100) ; 
									    color: white; 
									    height: 60px;
									    width: 50px;
									    margin: 1px 0px 1px 0px;
									    border: 1px transparent #2A2929; 
									    border-radius: 40px;   
									     ''')
		self.upload_file_btn2.setObjectName("upload_file_btn2")
		# --------------------------------------------------------------------------------------------------------
		self.extract_features_btn2 = QtWidgets.QPushButton(self.tab_2)  ##2
		self.extract_features_btn2.setGeometry(QtCore.QRect(40, 260, 220, 91))
		self.extract_features_btn2.setAutoFillBackground(False)
		self.extract_features_btn2.setStyleSheet(self.normal_btn_style)

		self.extract_features_btn2.setObjectName("extract_features_btn2")
		# --------------------------------------------------------------------------------------------------------
		self.training_btn2 = QtWidgets.QPushButton(self.tab_2)  ##3
		self.training_btn2.setGeometry(QtCore.QRect(160, 470, 150, 91))
		self.training_btn2.setAutoFillBackground(False)
		self.training_btn2.setStyleSheet(self.normal_btn_style)
		self.training_btn2.setObjectName("training_btn2")
		# ---------------------------------------------------------------------------------------------

		self.test_btn2 = QtWidgets.QPushButton(self.tab_2)  ##4
		self.test_btn2.setGeometry(QtCore.QRect(390, 535, 171, 81))
		self.test_btn2.setAutoFillBackground(False)
		self.test_btn2.setStyleSheet(self.normal_btn_style)
		self.test_btn2.setObjectName("test_btn2")
		# --------------------------------------------------------------------------------------------------------

		self.confusion_matrix_btn2 = QtWidgets.QPushButton(self.tab_2)  # 5
		self.confusion_matrix_btn2.setGeometry(QtCore.QRect(655, 460, 171, 91))
		self.confusion_matrix_btn2.setAutoFillBackground(False)
		self.confusion_matrix_btn2.setStyleSheet(self.normal_btn_style)
		self.confusion_matrix_btn2.setObjectName("confusion_matrix_btn2")
		# --------------------------------------------------------------------------------------------------------
		self.report_btn2 = QtWidgets.QPushButton(self.tab_2)  ##6
		self.report_btn2.setGeometry(QtCore.QRect(750, 260, 171, 91))
		self.report_btn2.setAutoFillBackground(False)
		self.report_btn2.setStyleSheet(self.normal_btn_style)
		self.report_btn2.setObjectName("report_btn2")

		# --------------------------------------------------------------------------------------------------------

		self.wordcloud_btn2 = QtWidgets.QPushButton(self.tab_2)  ##7
		self.wordcloud_btn2.setGeometry(QtCore.QRect(690, 75, 161, 91))
		self.wordcloud_btn2.setAutoFillBackground(False)
		self.wordcloud_btn2.setStyleSheet(self.normal_btn_style)
		self.wordcloud_btn2.setObjectName("wordcloud_btn2")

		#---------------------------------------------------------------------------------------------------------
		self.reset2 = QtWidgets.QPushButton(self.tab_2)
		self.reset2.setGeometry(QtCore.QRect(910, 565, 80, 60))
		self.reset2.setAutoFillBackground(False)
		self.reset2.setStyleSheet(self.normal_btn_style)
		self.reset2.setObjectName("reset2")
# --------------------------------------------------------------------------------------------------------
		self.background_label.raise_()
		self.extract_features_btn2.raise_()
		self.training_btn2.raise_()
		self.wordcloud_btn2.raise_()
		self.upload_file_btn2.raise_()
		self.test_btn2.raise_()
		self.confusion_matrix_btn2.raise_()
		self.report_btn2.raise_()
		self.reset2.raise_()

		self.tabs.addTab(self.tab_2, "")
		MainWindow.setCentralWidget(self.centralwidget)
		self.menubar = QtWidgets.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 1028, 26))
		self.menubar.setObjectName("menubar")
		MainWindow.setMenuBar(self.menubar)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)
		# --------------------------------------------------------------------------------------------------------

		self.tab_3 = QtWidgets.QWidget()
		self.tab_3.setEnabled(True)
		self.tab_3.setAutoFillBackground(False)
		self.tab_3.setObjectName("tab_3")

		self.background2_label = QtWidgets.QLabel(self.tab_3)
		self.background2_label.setGeometry(QtCore.QRect(0, 0, 995, 632))
		self.background2_label.setText("")
		self.background2_label.setPixmap(QtGui.QPixmap("background4.png"))
		self.background2_label.setScaledContents(True)
		self.background2_label.setObjectName("background_label2")

		# --------------------------------------------------------------------------------------------------------
		self.upload_file_btn3 = QtWidgets.QPushButton(self.tab_3)  ##1
		self.upload_file_btn3.setGeometry(QtCore.QRect(200, 70, 150, 91))
		self.upload_file_btn3.setAutoFillBackground(False)
		self.upload_file_btn3.setStyleSheet('''    
												font: 75 15pt "Orbitron";
											    background-color: rgb(255, 97, 100) ; 
											    color: white; 
											    height: 60px;
											    width: 50px;
											    margin: 1px 0px 1px 0px;
											    border: 1px transparent #2A2929; 
											    border-radius: 40px;   
											     ''')
		self.upload_file_btn3.setObjectName("upload_file_btn3")
		# --------------------------------------------------------------------------------------------------------
		self.extract_features_btn3 = QtWidgets.QPushButton(self.tab_3)  ##2
		self.extract_features_btn3.setGeometry(QtCore.QRect(40, 260, 220, 91))
		self.extract_features_btn3.setAutoFillBackground(False)
		self.extract_features_btn3.setStyleSheet(self.normal_btn_style)

		self.extract_features_btn3.setObjectName("extract_features_btn3")
		# --------------------------------------------------------------------------------------------------------
		self.training_btn3 = QtWidgets.QPushButton(self.tab_3)  ##3
		self.training_btn3.setGeometry(QtCore.QRect(160, 470, 150, 91))
		self.training_btn3.setAutoFillBackground(False)
		self.training_btn3.setStyleSheet(self.normal_btn_style)
		self.training_btn3.setObjectName("training_btn3")
		# ---------------------------------------------------------------------------------------------

		self.test_btn3 = QtWidgets.QPushButton(self.tab_3)  ##4
		self.test_btn3.setGeometry(QtCore.QRect(390, 535, 171, 81))
		self.test_btn3.setAutoFillBackground(False)
		self.test_btn3.setStyleSheet(self.normal_btn_style)
		self.test_btn3.setObjectName("test_btn3")
		# --------------------------------------------------------------------------------------------------------

		self.confusion_matrix_btn3 = QtWidgets.QPushButton(self.tab_3)  # 5
		self.confusion_matrix_btn3.setGeometry(QtCore.QRect(655, 460, 171, 91))
		self.confusion_matrix_btn3.setAutoFillBackground(False)
		self.confusion_matrix_btn3.setStyleSheet(self.normal_btn_style)
		self.confusion_matrix_btn3.setObjectName("confusion_matrix_btn3")
		# --------------------------------------------------------------------------------------------------------

		self.report_btn3 = QtWidgets.QPushButton(self.tab_3)  ##6
		self.report_btn3.setGeometry(QtCore.QRect(750, 260, 171, 91))
		self.report_btn3.setAutoFillBackground(False)
		self.report_btn3.setStyleSheet(self.normal_btn_style)
		self.report_btn3.setObjectName("report_btn3")

		# --------------------------------------------------------------------------------------------------------

		self.wordcloud_btn3 = QtWidgets.QPushButton(self.tab_3)  ##7
		self.wordcloud_btn3.setGeometry(QtCore.QRect(690, 75, 161, 91))
		self.wordcloud_btn3.setAutoFillBackground(False)
		self.wordcloud_btn3.setStyleSheet(self.normal_btn_style)
		self.wordcloud_btn3.setObjectName("wordcloud_btn3")

		# ---------------------------------------------------------------------------------------------------------
		self.reset3 = QtWidgets.QPushButton(self.tab_3)
		self.reset3.setGeometry(QtCore.QRect(910, 565, 80, 60))
		self.reset3.setAutoFillBackground(False)
		self.reset3.setStyleSheet(self.normal_btn_style)
		self.reset3.setObjectName("reset3")
		# --------------------------------------------------------------------------------------------------------
		self.background2_label.raise_()
		self.extract_features_btn3.raise_()
		self.training_btn3.raise_()
		self.wordcloud_btn3.raise_()
		self.upload_file_btn3.raise_()
		self.test_btn3.raise_()
		self.confusion_matrix_btn3.raise_()
		self.report_btn3.raise_()
		self.reset3.raise_()

		self.tabs.addTab(self.tab_3, "")

		# --------------------------------------------------------------------------------------------------------
		self.retranslateUi(MainWindow)
		self.tabs.setCurrentIndex(0)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

	# --------------------------------------------------------------------------------------------------------
	# -------------------------------------- Actions on buttons ----------------------------------------------
	# --------------------------------------------------------------------------------------------------------

		self.upload_file_btn.clicked.connect(self.browse_file)
		self.extract_features_btn.clicked.connect(self.extract_features)
		self.wordcloud_btn.clicked.connect(self.Show_wordclouds)
		self.reset.clicked.connect(self.reset_btns)


		self.upload_file_btn2.clicked.connect(self.browse_file2)
		self.extract_features_btn2.clicked.connect(self.extract_features2)
		self.wordcloud_btn2.clicked.connect(self.Show_wordclouds2)
		self.reset2.clicked.connect(self.reset_btns)
		self.upload_file_btn3.clicked.connect(self.browse_file3)
		self.extract_features_btn3.clicked.connect(self.extract_features3)
		self.wordcloud_btn3.clicked.connect(self.Show_wordclouds3)
		self.reset3.clicked.connect(self.reset_btns)


	# --------------------------------------------------------------------------------------------------------

	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "Tweet Spam Detector"))
		self.extract_features_btn.setText(_translate("MainWindow", "extract features vector"))
		self.training_btn.setText(_translate("MainWindow", "Training"))
		self.wordcloud_btn.setText(_translate("MainWindow", "Show wordcloud"))
		self.upload_file_btn.setText(_translate("MainWindow", "Choose .csv"))
		self.test_btn.setText(_translate("MainWindow", "Accuracy Test"))
		self.confusion_matrix_btn.setText(_translate("MainWindow", "Confusion Matrix"))
		self.tabs.setTabText(self.tabs.indexOf(self.tab_1), _translate("MainWindow", "Naive Bayes"))
		self.report_btn.setText(_translate("MainWindow", "Report"))
		self.reset.setText(_translate("MainWindow", "Reset"))


		self.extract_features_btn2.setText(_translate("MainWindow", "extract features vector"))
		self.training_btn2.setText(_translate("MainWindow", "Training"))
		self.wordcloud_btn2.setText(_translate("MainWindow", "Show wordcloud"))
		self.upload_file_btn2.setText(_translate("MainWindow", "Choose .csv"))
		self.test_btn2.setText(_translate("MainWindow", "Accuracy Test"))
		self.confusion_matrix_btn2.setText(_translate("MainWindow", "Confusion Matrix"))
		self.report_btn2.setText(_translate("MainWindow", "Report"))
		self.reset2.setText(_translate("MainWindow", "Reset"))
		self.tabs.setTabText(self.tabs.indexOf(self.tab_2), _translate("MainWindow", "Random Forest"))

		self.extract_features_btn3.setText(_translate("MainWindow", "extract features vector"))
		self.training_btn3.setText(_translate("MainWindow", "Training"))
		self.wordcloud_btn3.setText(_translate("MainWindow", "Show wordcloud"))
		self.upload_file_btn3.setText(_translate("MainWindow", "Choose .csv"))
		self.test_btn3.setText(_translate("MainWindow", "Accuracy Test"))
		self.confusion_matrix_btn3.setText(_translate("MainWindow", "Confusion Matrix"))
		self.report_btn3.setText(_translate("MainWindow", "Report"))
		self.reset3.setText(_translate("MainWindow", "Reset"))
		self.tabs.setTabText(self.tabs.indexOf(self.tab_3), _translate("MainWindow", "Neural Network"))



if __name__ == "__main__":
	import sys

	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())
