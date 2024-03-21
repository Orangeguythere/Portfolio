import pandas as pd
import numpy as np
import os 

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

#import mne
import matplotlib.pyplot as plt
import seaborn as sns

import tqdm 
import mlflow
import mlflow.sklearn # Wrapper pour scikit-learn
import librosa
from scipy import signal

# suppress scientific notation by setting float_format
#pd.options.display.float_format = '{:.0f}'.format


"""# Identification de l'interface MLflow
server_uri = 'http://127.0.0.1:8080'
mlflow.set_tracking_uri(server_uri)
# Identification du nom du projet MLflow
mlflow.set_experiment("Data_Science_Project_HMS")
"""

#To show the data graphics 

GET_ROW = 10
EEG_PATH = "D:\\DOWNLOADS\\hms-harmful-brain-activity-classification\\train_eegs\\"
SPEC_PATH = "D:\\DOWNLOADS\\hms-harmful-brain-activity-classification\\train_spectrograms\\"

train = pd.read_csv(r"D:\DOWNLOADS\hms-harmful-brain-activity-classification\train.csv")
row = train.iloc[GET_ROW]

eeg = pd.read_parquet(f'{EEG_PATH}{row.eeg_id}.parquet')
eeg_offset = int( row.eeg_label_offset_seconds )
eeg = eeg.iloc[eeg_offset*200:(eeg_offset+50)*200]
  
spectrogram = pd.read_parquet(f'{SPEC_PATH}{row.spectrogram_id}.parquet')
spec_offset = int( row.spectrogram_label_offset_seconds )
spectrogram = spectrogram.loc[(spectrogram.time>=spec_offset)
                     &(spectrogram.time<spec_offset+600)]
print(train)
print(eeg)
print(spectrogram)



#Add target to spec data, with the last column vote (panda)
LL_Spec = ( (eeg["Fp1"] - eeg["F7"]) + (eeg["F7"] - eeg["T3"]) + (eeg["T3"] - eeg["T5"]) + (eeg["T5"] - eeg["O1"]) )/4
LP_Spec = ( (eeg["Fp1"] - eeg["F3"]) + (eeg["F3"] - eeg["C3"]) + (eeg["C3"] - eeg["P3"]) + (eeg["P3"] - eeg["O1"]) )/4
RP_Spec = ( (eeg["Fp2"] - eeg["F4"]) + (eeg["F4"] - eeg["C4"]) + (eeg["C4"] - eeg["P4"]) + (eeg["P4"] - eeg["O2"]) )/4
RL_Spec = ( (eeg["Fp2"] - eeg["F8"]) + (eeg["F8" ]- eeg["T4"]) + (eeg["T4"] - eeg["T6"]) + (eeg["T6"] - eeg["O2"]) )/4

print(LL_Spec)
#LL_Spec=LL_Spec.to_numpy()



g =sns.lineplot(LL_Spec, color='r', label='LL_Spec')
g =sns.lineplot(LP_Spec, color='g', label='LP_Spec')
g =sns.lineplot(RP_Spec, color='b', label='RP_Spec')
g =sns.lineplot(RL_Spec, color='darkorange', label='RL_Spec')
plt.show()   


"""f, t, Sxx = signal.spectrogram(LL_Spec, 200)
plt.figure(figsize=(15,8))
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.xlim([0, 50])
plt.ylim([0, 20])
plt.show()"""

"""fig, ax = plt.subplots()
mel_spec = librosa.feature.melspectrogram(y=LL_Spec, sr=200, hop_length=len(LL_Spec)//256, 
          n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

# LOG TRANSFORM
width = (mel_spec.shape[1]//32)*32
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

# STANDARDIZE TO -1 TO 1
mel_spec_db = (mel_spec_db+40)/40 


plt.imshow(mel_spec_db,aspect='auto',origin='lower')
plt.show()

"""

"""
#print(df['Comedy_Score'].where(df['Rating_Score'] < 50))
for item in train_file["eeg_id"].unique():
    filtered_df = train_file.where(train_file["eeg_id"]==item)
    filtered_df = filtered_df.dropna()
    print(filtered_df)

    code_data_EGG = "\/"+str(item)+".parquet"
    file_EGG=(r"C:\\Users\\Orange\\Downloads\train_eegs"+code_data_EGG)
    #file=(r"D:\DOWNLOADS\hms-harmful-brain-activity-classification\train_eegs")
    #file=(r"D:\DOWNLOADS\hms-harmful-brain-activity-classification\test_spectrograms")

    df_EGG = pd.read_parquet(file_EGG)
    df_EGG = df_EGG.drop('EKG', axis=1)

    #Spectrogram 
    item_SPEC=int(filtered_df["spectrogram_id"].iloc[0])
    code_data_SPEC = "\/"+str(item_SPEC)+".parquet"
    file_SPEC=(r"C:\\Users\\Orange\\Downloads\train_spectrograms"+code_data_SPEC)

    df_SPEC = pd.read_parquet(file_SPEC)
    print(df_SPEC)
    
    #Show plot of EEG 100 values
    g =sns.lineplot(data=df_EGG.iloc[0:50])
    for item in filtered_df["eeg_label_offset_seconds"] : 
        g.axvline(item, color="red", linestyle="--")
    
    plt.show()

    #Show plot of Spec 100 values
    #sns.heatmap(data, cmap='hot', xticklabels=frequency_labels, yticklabels=electrode_labels)
    f =sns.heatmap(data=df_SPEC.iloc[0:1000],xticklabels="time",cmap='viridis')
    
    
    plt.show()
    """


"""
TARGETS = train_file.columns[-6:]

train = train_file.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
train.columns = ['spec_id','min']

tmp = train_file.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_label_offset_seconds':'max'})
train['max'] = tmp

tmp = train_file.groupby('eeg_id')[['patient_id']].agg('first') 
train['patient_id'] = tmp

tmp = train_file.groupby('eeg_id')[TARGETS].agg('sum') # The code sums up the target variable counts (like votes for seizure, LPD, etc.) for each eeg_id.
for t in TARGETS:
    train[t] = tmp[t].values


y_data = train[TARGETS].values # It then normalizes these counts so that they sum up to 1. This step converts the counts into probabilities, which is a common practice in classification tasks.
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

tmp = train_file.groupby('eeg_id')[['expert_consensus']].agg('first') # For each eeg_id, the code includes the expert_consensus on the EEG segment's classification.
train['target'] = tmp

train = train.reset_index() # This makes eeg_id a regular column, making the DataFrame easier to work with.

print(train)
"""


"""def load_eeg_data(eeg_id):
    try:
        filepath = f"C:\\Users\\Orange\\Downloads\\train_eegs\\{eeg_id}.parquet"
        eeg_data = pd.read_parquet(filepath)
        # Preprocess EEG data based on your specific needs (e.g., feature engineering)
        return eeg_data
    except FileNotFoundError:
        print(f"EEG data for ID {eeg_id} not found.")
        return None
    
def load_spectrogram_data(spectrogram_id):
    try:
        filepath = f"C:\\Users\\Orange\\Downloads\\train_spectrograms\\{spectrogram_id}.parquet"
        spectrogram_data = pd.read_parquet(filepath)
        # Preprocess spectrogram data based on your specific needs
        return spectrogram_data
    except FileNotFoundError:
        print(f"Spectrogram data for ID {spectrogram_id} not found.")
        return None

#print(load_eeg_data(387987538))

def prepare_features(main_data_row):
    #eeg_id = main_data_row["eeg_id"]
    spectrogram_id = main_data_row["spectrogram_id"]
    #eeg_data = load_eeg_data(eeg_id)
    spectrogram_data = load_spectrogram_data(spectrogram_id)
    if spectrogram_data is not None:
        # Combine features in a suitable way (e.g., concatenation, stacking)
        features = pd.concat([spectrogram_data], axis=1)
        return features
    else:
        # Handle missing data (e.g., return None, impute values)
        return None

X = []
y = []
for index, row in train_file.iterrows():
    features = prepare_features(row)
    if features is not None:
        X.append(features.to_dict('records'))
        y.append(row["expert_consensus"])  # Adjust based on your actual target column
    if index >= 20:
        print(X)
        print(y)
        break  # Stop after processing 10 items
"""
"""
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train SVM model
model = SVC(kernel='rbf', C=1.0)  # Adjust hyperparameters as needed
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
"""
      
    




