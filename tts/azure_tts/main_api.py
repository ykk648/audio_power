import os
import azure.cognitiveservices.speech as speechsdk

speech_key, service_region = os.getenv('AZURE_TTS_KEY'), os.getenv('AZURE_TTS_LOCATION')


def get_audio(text=None, role=None, language=None, volume="+0%", pitch="+0Hz", rate=None):
    if language:
        language = language.split("-", maxsplit=1)
    else:
        language = role.split('-', maxsplit=2)
    language = language[0].lower() + ("" if len(language) < 2 else '-' + language[1].upper())

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # audio_config = speechsdk.audio.AudioOutputConfig(filename="./test.wav")
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    # Creates a speech synthesizer using the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # with open("ssml.xml", "rb") as file:
    #     ssml_string = file.read().decode('utf-8')


    ssml_string = """<speak version='1.0' xml:lang='{}' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts'>
    <voice name='{}'>
        <prosody rate="{}" pitch='{}'  volume='{}'>
        {}
        </prosody>
    </voice>
    </speak>""".format(language, role, rate, pitch, volume, text)

    print(ssml_string)
    result = speech_synthesizer.speak_ssml_async(ssml_string).get()

    # Checks result.
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for text [{}]".format('1'))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
        print("Did you update the subscription info?")


if __name__ == '__main__':
    """
    pitch/rate/volume ref https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/speech-synthesis-markup-voice#adjust-prosody
    
    """
    text_in = '太好了,我是云夏,云彩的云,夏天的夏'
    get_audio(text=text_in, role='zh-CN-YunxiaNeural', language='zh-CN', rate="+30.00%", pitch="high", volume='loud')
