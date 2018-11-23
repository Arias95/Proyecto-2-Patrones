from src.load_audios import normalize
#from src.train import train
#from src.predict import predictNumber
import pathlib
import sys



audio_path = 'audio/data'
normalized_audio_path = 'audio/normalized_data'


if __name__=='__main__':
    pathlib.Path(normalized_audio_path).mkdir(parents=True, exist_ok=True)

    action=sys.argv[1]

    if action=="-n" or action =="--normalize":
        normalize(audio_path,normalized_audio_path)
    # elif action=="-t" or action =="--train":
    #     train(normalized_audio_path)
    # elif action=="-p" or action =="--precit":
    #     audio=sys.argv[2]
    #     number = predictNumber(audio)
    #     print("System Output: " + number)
    # elif action=="-h" or action =="--help":
    #     print("-n or --normalize :   Normalize audio inputs     ")
    #     print("-t or --train     :   Train normalized audio iputs")
    #     print("-p or --predict   :   Predict an audio input")

    # else:
    #     print("Unknown command, please try -h or --help")
