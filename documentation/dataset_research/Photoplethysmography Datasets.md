

# Photoplethysmography Datasets

[https://peterhcharlton.github.io/post/ppg\_datasets/](https://peterhcharlton.github.io/post/ppg_datasets/)

Here's a summary of the photoplethysmography datasets grouped by type:

### **Clinical & Critical Care**

- **MIMIC Critical Care Database** (10,000s subjects): Recordings from critically-ill adults and neonates with ECG, BP, and respiratory signals  
- **MIMIC PERform Training and Testing** (400 subjects): 10-minute recordings from critically-ill patients  
- **MIMIC PERform AF Dataset** (35 subjects): Categorized by atrial fibrillation vs. normal sinus rhythm  
- **BIDMC** (53 subjects): 8-minute recordings from critically-ill adults  
- **CapnoBase** (42 subjects): Recordings during elective surgery and anaesthesia

### **Sleep & Polysomnography**

- **MESA Dataset** (2,056 subjects): Finger PPG during polysomnography  
- **SOMNIA Database** (100s subjects): Wrist PPG from children and adults during sleep studies  
- **Sleep Disordered Breathing Database** (146 subjects): \~3-hour finger recordings from children

### **Healthy Subjects & Daily Activities**

- **UK Biobank** (205,357 subjects): Single finger PPG from middle-aged subjects  
- **PPG-DaLiA Data Set** (15 subjects): \~2.5 hours during daily living activities  
- **WESAD Data Set** (15 subjects): \~2 hour protocol with amusement, stress, and relaxation  
- **Real-World PPG dataset** (35 subjects): Multiple 6-second recordings from ages 10-74

### 

### **Exercise & Physical Activity**

- **VitalDB** (6,153 subjects): Finger PPG during operations  
- **Vortal Dataset** (57 subjects): 10-min recordings before/after exercise  
- **gyro-acc-ppg Dataset** (24 subjects): Wrist recordings during 12-minute exercise protocol  
- **IEEE Signal Processing Cup 2015** (12 subjects): 5-min recordings during intensive exercise  
- **Wrist PPG During Exercise** (8 subjects): Walking, running, and biking

### **Emotion & Stress**

- **DEAP Database** (32 subjects): Thumb recordings while watching videos  
- **ECSMP** (89 subjects): Wrist recordings designed to induce emotions  
- **Eight-Emotion Sentics Dataset** (1 subject): 25-min emotion-evoking protocol over 20 days  
- **AffectiveROAD Dataset** (10 subjects): \~86 minutes of driving recordings

### **Specialized Studies**

- **PPG-BP Database** (219 subjects): Three finger recordings for blood pressure estimation  
- **Pulse transit time during vasoconstriction Dataset** (86 subjects): Cold pressor and active stand tests  
- **MARSH Dataset** (29 subjects): 15-min recordings during spontaneous and metronome-guided breathing  
- **HYPE Dataset** (9 subjects): Wrist recordings from hypertensive subjects

### **Synthetic & Animal Data**

- **Pulse Wave DataBase** (4,374 synthetic): Simulated PPG pulse waves  
- **Photoplethysmography in dogs and cats** (21 animals): Recordings from dogs and cats

## 

## Impact Analysis: 

The **highest impact and feasibility** datasets depend on your goal — here’s a quick guide:

### ✅ **Highest Impact \+ Feasibility (for most researchers & developers):**

#### **1\. UK Biobank (Field 4205\)**

- **Impact**: Massive scale (205k+ subjects) → ideal for population-level studies, AI training, or epidemiological research.  
- **Feasibility**: Publicly available, well-documented, no complex setup needed.  
- **Best for**: Large-scale PPG analysis, cardiovascular risk prediction, or benchmarking algorithms.

#### **2\. MIMIC Critical Care Database**

- **Impact**: Real-world ICU data with multimodal signals (ECG, BP, resp) → gold standard for clinical PPG applications.  
- **Feasibility**: Requires registration (free for researchers), but widely used and well-supported.  
- **Best for**: Critical care monitoring, sepsis detection, or PPG-based vital sign estimation.

#### **3\. PPG-DaLiA / WESAD**

- **Impact**: Daily living \+ emotion/stress context → great for wearable AI, affective computing, or activity recognition.  
- **Feasibility**: Small to medium size, clean annotations, easy to work with.  
- **Best for**: Emotion detection, stress monitoring, or activity classification using wrist PPG.

---

### 🎯 **Quick Recommendation:**

If you want **maximum impact with minimal friction**, start with **UK Biobank**.  
If you’re targeting **clinical applications**, go with **MIMIC**.  
For **wearable/emotion research**, **PPG-DaLiA or WESAD** are ideal.

| Dataset | No. subjs | Other signals | Description |
| ----- | ----- | ----- | ----- |
| [UK Biobank](https://www.ukbiobank.ac.uk/) (Field 4205\) | 205,357 | None simultaneous. BP in same visit | Single finger PPG waves from middle-aged subjects. |
| [MIMIC Critical Care Database](https://mimic.physionet.org/) | 10,000s (growing) | ECG, BP, resp, others | Recordings from critically-ill adults and neonates, lasting from minutes to days. Typically at finger. |
| [VitalDB](https://vitaldb.net/) | 6,153 | ECG, BP, resp, others | Finger PPG recordings from patients during operations. |
| [MESA Dataset](https://sleepdata.org/datasets/mesa) | 2,056 | ECG, resp, others | Finger PPG recordings from adults undergoing polysomnography. |
| [MIMIC PERform Training and Testing](https://doi.org/10.5281/zenodo.6807402) | 400 | ECG, resp | Recordings from critically-ill adults and neonates, lasting 10 minutes. Typically at finger. |
| SOMNIA Database | 100s (growing) | ECG, resp, others | Wrist PPG recordings from children and adults undergoing polysomnography. |
| [PPG-BP Database](https://doi.org/10.6084/m9.figshare.5459299) | 219 | \- | Three finger recordings from adults aged 20-89 with and without CVD,  3 waves per recording. |
| [Sleep Disordered Breathing Database](https://doi.org/10.6084/m9.figshare.1209662.v6) | 146 | \- | Finger recordings lasting  3 hours, acquired from children referred for polysomnography. |
| [ECSMP](https://doi.org/10.17632/vn5nknh3mn.2) | 89 | ECG, accel, others | Wrist recordings acquired from healthy subjects during a protocol designed to induce different emotions. |
| [Pulse transit time during vasoconstriction Dataset](https://doi.org/10.34973/te70-x603) | 86 | ECG, BP |  35-min wrist and finger recordings during cold pressor and active stand tests. |
| [Vortal Dataset](https://doi.org/10.18742/RDM01-194) | 57 | ECG, resp | 10-min finger and ear recordings before and after exercise from healthy adults aged 18-39 and \\textgreater 70\. |
| [BIDMC](https://doi.org/10.13026/C2208R) | 53 | ECG, BP, resp | 8-min recordings from critically-ill adults (a subset of the MIMIC-II dataset). |
| [CapnoBase](http://www.capnobase.org/index.php?id=857) | 42 | ECG, resp | 8-min recordings from paediatrics and adults during elective surgery and anaesthesia. |
| [Bed-based BCG Dataset](https://dx.doi.org/10.21227/77hc-py84) | 40 | ECG, BCG, BP | Recordings from adults whilst at rest. |
| [MIMIC PERform AF Dataset](https://doi.org/10.5281/zenodo.6807402) | 35 | ECG, resp | Recordings from critically-ill adults categorised as either AF (19 subjects) or normal sinus rhythm (16 subjects), lasting 10 minutes. Typically at finger. |
| [Real-World PPG dataset](https://doi.org/10.17632/yynb8t9x3d.2) | 35 | \- | Recordings from healthy subjects aged 10 to 74 years old: several 6-second recordings per subject. |
| [DEAP Database](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html) | 32 | ECG, resp, video, others | Thumb recordings from young, healthy subjects whilst watching one-minute 40 videos. |
| [University of Queensland Vital Signs Dataset](https://doi.org/102.100.100/6914) | 32 | ECG, resp, BP, EEG | Recordings from patients during anaesthesia, ranging from minutes to hours in duration. |
| [ESUM SNF Project Dataset](http://esum.arch.ethz.ch/data) | 31 | accel, EDA | Wrist recordings whilst walking. |
| [MARSH Dataset](https://doi.org/10.5281/zenodo.3673924) | 29 | ECG, resp | 15-min finger recordings during spontaneous and metronome-guided breathing. |
| [Non-invasive BP Estimation](https://www.kaggle.com/mkachuee/noninvasivebp) | 26 | ECG, BP, PCG | Finger recordings from healthy adults. |
| [gyro-acc-ppg Dataset](https://github.com/hooseok/gyro_acc_ppg) | 24 | ECG, accel, gyro | Wrist recordings from healthy subjects during an exercise protocol lasting 12 minutes. |
| [Pulse Transit Time PPG Dataset](https://doi.org/10.13026/g3me-rt62) | 22 | ECG, multiwavelength PPG | Finger recordings from healthy subjects during sitting, walking and running. |
| [Welltory-PPG-dataset](https://github.com/Welltory/welltory-ppg-dataset) | 21 | RR intervals | Smartphone recordings acquired from the index finger in contact with the camera. |
| [Wearable and Clinical Devices Dataset](https://doi.org/10.21979/N9/42BBFA) | 18 | ECG, resp, accel, EDA | Wrist PPG recordings acquired for 5 mins at rest, and 5mins whilst walking on spot. |
| [PPG-DaLiA Data Set](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA) | 15 | ECG, resp, accel, EDA | Recordings acquired for  2.5 hours during a protocol of daily living activities. |
| [WESAD Data Set](https://archive.ics.uci.edu/ml/datasets/WESAD+%5c%28Wearable+Stress+and+Affect+Detection%5c%29) | 15 | ECG, resp, accel, others | Recordings acquired in a  2 hour protocol designed to amuse, stress, and relax. |
| [iAMwell Dataset](https://doi.org/10.5281/zenodo.1012726) | 15 | ECG, resp |  20-min recordings before, during and after running. |
| [Simultaneous Measurements Dataset](https://doi.org/10.13026/chd5-t946) | 13 | ECG, accel, resp | Recordings from adults at rest and during cognitive and physical tasks. |
| [IEEE Signal Processing Cup 2015](https://sites.google.com/site/researchbyzhang/ieeespcup2015) | 12 | ECG, accel, two PPGs |  5-min recordings during intensive physical exercise from males aged 18-35. |
| [AffectiveROAD Dataset](https://affect.media.mit.edu/share-data.php) | 10 | accel, EDA | Recordings acquired whilst driving a car for  86 minutes from mostly young adults. |
| [Labeled raw PPG Signals](https://doi.org/10.4231/1BE9-YY17) | 9 | \- | Finger recordings acquired at rest for  20-40 minutes from healthy adults. |
| [HYPE Dataset](https://github.com/arianesasso/aime-2020) | 9 | intermittent BP | Wrist recordings from hypertensive subjects: 8 subjects during stress tests, and 9 subjects during 24-hour monitoring. |
| [Wrist PPG During Exercise](https://doi.org/10.13026/C2PQ1X) | 8 | BP, accel | Wrist recordings acquired from adults aged 22-32 during walking, running and bike riding. |
| [PPG-ACC Dataset](https://doi.org/10.1016/j.dib.2019.105044) | 7 | accel | Wrist recordings acquired from healthy adults aged 20-52 during rest, squatting and stepping. |
| [Raw PPG Signal in Varying Levels of Activity](https://doi.org/10.4231/8VF2-1729) | 5 | \- | Finger recordings acquired at rest, talking and walking for  10 mins each from healthy adults. |
| [PPG-Diary](https://doi.org/10.5281/zenodo.3268501) | 1 | two PPGs | A 28-day thumb recording from a healthy adult, with annotations of activities of daily living. |
| [Eight-Emotion Sentics Dataset](https://affect.media.mit.edu/share-data.php) | 1 | EMG, GSR, resp | 25-min recordings during a protocol to evoke emotions from 1 subject each day for 20 days. |
| [Pulse Wave DataBase](https://doi.org/10.5281/zenodo.2633175) | 4,374 (synthetic) | BP, blood flow, others | Single simulated PPG pulse waves representative of healthy adults aged 25-75. |
| [Photoplethysmography in dogs and cats](https://doi.org/10.5281/zenodo.1296214) | 21 (animals) | three PPGs |  10-20 sec recordings at several arterial sites from 11 dogs and 10 cats. |

