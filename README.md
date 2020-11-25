# Voice-Activity-Detection

- 해당 코드는 https://github.com/nicklashansen/voice-activity-detection 의 Jupyter notebook 코드 기반으로 
한국어 VAD를 만들기 위해 작성된 코드임을 알립니다.


## Dataset
- 한국어 데이터
    - kss Dataset(https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)
    - zeroth-Korean Dataset(http://www.openslr.org/40/)
    - AIHub의 한국어 음성데이터(https://aihub.or.kr/aidata/105)
- Noise Dataset
    - QUT-NOISE Dataset(https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/)
 
## Requirements
- torch==1.7.0
- scikit-learn==0.23.2
- h5py==3.1.0
- h5py-cache==1.0
- numpy==1.19.2
- pydub==0.24.1
- python-speech-features==0.6
- SoundFile==0.10.3.post1
- webrtcvad==2.0.10
- ipython==7.19.0
- matplotlib==3.3.3

## 사용법

