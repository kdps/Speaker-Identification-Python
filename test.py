import time
import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")

"""
#path to training data
source   = "development_set/"   
modelpath = "speaker_models/"
test_file = "development_set_test.txt"        
file_paths = open(test_file,'r')

"""
# path to training data
source = "SampleData/"

# path where training speakers will be saved
modelpath = "Speakers_models/"

gmm_files = [os.path.join(modelpath, fname) for fname in
             os.listdir(modelpath) if fname.endswith('.gmm')]

# Load the Gaussian gender Models
models = [cPickle.load(open(fname, 'rb')) for fname in gmm_files]
speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
            in gmm_files]

error = 0
total_sample = 0.0


print("단일 오디오를 테스트 하시겠습니까 : '1'또는 전체 테스트 오디오 샘플 : '0'을 누르십시오. ?")
take = int(input().strip())
if take == 1:
    print("테스트 오디오 샘플 컬렉션의 파일 이름 입력 :")
    path = input().strip()
    sr, audio = read(source + path)
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]  # checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
        print("",speakers[i],scores.sum(), i)
        
    try: 
        winner = np.argmax(log_likelihood)
        print("\t감지된 GMM 모델 - ", path, log_likelihood[winner], winner, speakers[winner])
    except ValueError:
        time.sleep(1.0)
        
    time.sleep(1.0)
elif take == 0:
    test_file = "testSamplePath.txt"
    file_paths = open(test_file, 'r')

    # Read the test directory and get the list of test audio files
    for path in file_paths:

        total_sample += 1.0
        path = path.strip()
        print("Testing Audio : ", path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
            print("",speakers[i],scores.sum(), i)
            
        winner = np.argmax(log_likelihood)
        print("\t감지된 GMM 모델 - ", path, log_likelihood[winner], winner, speakers[winner])

        checker_name = path.split("_")[0]
        if speakers[winner] != checker_name:
            error += 1
        time.sleep(1.0)

    print(error, total_sample)
    accuracy = ((total_sample - error) / total_sample) * 100

    print("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", accuracy, "%")


print("화자가 확인되었습니다. 미션이 성공적으로 완료되었습니다.")
