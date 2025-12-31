# -- coding: utf-8 --
# @Time : 2024/1/2
# @LastEdit : 2024/2/8
# @Author : ykk648

r"""
ref https://github.com/coqui-ai/TTS
https://github.com/jianchang512/clone-voice/blob/main/clone/logic.py

pip install TTS

model download to:
linux: ~/.local/share/tts
win: C:\Users\user\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2

"""
from TTS.api import TTS


class TTSFreeVCAPI:
    def __init__(self, device='cuda'):
        self.tts = TTS(model_name='voice_conversion_models/multilingual/vctk/freevc24', progress_bar=False).to(device)

    def forward(self, source_wav, target_wav, output_path):
        self.tts.voice_conversion_to_file(source_wav=source_wav, target_wav=target_wav,
                                          file_path=output_path)


class TTSXTTSAPI:
    def __init__(self, device='cuda'):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def forward(self, source_txt, target_wav, output_path):
        self.tts.tts_to_file(
            text=source_txt,
            speaker_wav=target_wav,
            language="es", file_path=output_path)


class TTSBarkAPI:
    def __init__(self, device='cuda'):
        self.tts = TTS("tts_models/multilingual/multi-dataset/bark").to(device)

    def forward(self, source_txt, target_dir, output_path):
        # Cloning a new speaker
        # This expects to find a mp3 or wav file like `bark_voices/new_speaker/speaker.wav`
        # It computes the cloning values and stores in `bark_voices/new_speaker/speaker.npz`
        self.tts.tts_to_file(
            text=source_txt,
            file_path=output_path,
            voice_dir=target_dir,
            speaker="new_speaker")


if __name__ == '__main__':
    from cv2box.utils import MyTimer

    source_wav = "resources/daihuo.wav"
    target_wav = r"es_10s.wav"

    # ttsfvc = TTSFreeVCAPI()
    # ttsfvc.forward(source_wav, target_wav, 'output.wav')

    with MyTimer() as _:
        ttsxtts = TTSXTTSAPI()

    with MyTimer() as _:
        # source_txt = r'i am in python and try to install fancy impute using pip install but i get the following error (Cannot open include file:). i am try to download windows SDK and add path to environmental variables but also failed '
        source_txt = r'La selección nacional masculina de fútbol de Arabia Saudita es conocida como las “Águilas Verdes de Asia Occidental”. Se unió a la FIFA en 1959. En la década de 1980, el fútbol saudí se desarrolló rápidamente. La selección nacional obtuvo el tercer puesto en los Juegos Asiáticos de 1982 y el subcampeonato en 1986. Ganó la Copa Asiática tres veces en 1984, 1988 y 1996. En 1996, se convirtió en el segundo equipo después de Irán en ganar la Copa Asiática tres veces.'
        # source_txt = r'L’équipe nationale masculine de football d’Arabie saoudite est connue sous le nom de « Aigles verts d’Asie de l’Ouest ». A rejoint la FIFA en 1959. Dans les années 1980, le football saoudien s\'est développé à pas de géant. L\'équipe nationale a remporté la troisième place aux Jeux asiatiques en 1982 et la deuxième place en 1986. Elle a remporté la Coupe d\'Asie à trois reprises en 1984, 1988 et 1996. En 1996, elle est devenue la deuxième équipe après l\'Iran à remporter trois Coupes d\'Asie.'
        # source_txt = r'사우디아라비아 남자 축구 국가대표팀은 \'서아시아 그린 이글스\'로 알려져 있다. 1959년 FIFA에 가입했다. 1980년대 사우디아라비아 축구는 급속도로 발전했다. 국가대표팀은 1982년 아시안게임에서 3위, 1986년 준우승을 차지했다. 1984년, 1988년, 1996년 세 차례 아시안컵 우승을 차지했다. 1996년에는 이란에 이어 두 번째로 아시안컵 우승을 차지했다.'
        # source_txt = r'يُعرف المنتخب السعودي لكرة القدم للرجال باسم "النسور الخضراء لغرب آسيا". انضم إلى الفيفا عام 1959. وفي الثمانينيات، تطورت كرة القدم السعودية بشكل سريع، حيث حصل المنتخب الوطني على المركز الثالث في دورة الألعاب الآسيوية عام 1982، والوصيف عام 1986. فاز بكأس آسيا ثلاث مرات في أعوام 1984 و1988 و1996. وفي عام 1996، أصبح ثاني فريق بعد إيران يفوز بكأس آسيا ثلاث مرات.'
        # source_txt = r'沙特阿拉伯国家男子足球队，号称“西亚绿鹰”。1959年加入国际足联。20世纪80年代沙特阿拉伯足球有突飞猛进的发展，国家队在1982年夺得亚运会足球赛季军、1986年的亚军。1984、1988、1996年三次获亚洲杯赛冠军，1996年成为继伊朗之后，第二支三捧亚洲杯的球队。'
        ttsxtts.forward(source_txt, target_wav, 'es.wav')

    # target_dir = ''
    # ttsbark = TTSBarkAPI()
    # ttsbark.forward(source_txt, target_wav, 'output.wav')
