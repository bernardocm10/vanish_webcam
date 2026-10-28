"""Interface text in English or Brazilian Portuguese, picked from the system language.

Code, comments and docs are in English; everything the user reads in the app (panel,
terminal messages, errors, --help) goes through t(). The language comes from, in order:
--lang, the VANISHCAM_LANG environment variable, and the system language. Languages
other than Portuguese fall back to English.
"""

from __future__ import annotations

import locale
import os
import platform

LANGUAGES = ("en", "pt")
DEFAULT_LANGUAGE = "en"
ENV_VARIABLE = "VANISHCAM_LANG"


def normalize(name: str | None) -> str | None:
    """'pt_BR.UTF-8', 'Portuguese_Brazil', 'pt-BR' -> 'pt'; unknown or C/POSIX -> None."""
    if not name:
        return None
    name = name.strip().lower().replace("-", "_")
    if name in ("c", "posix") or name.startswith(("c.", "posix.")):
        return None
    if name.startswith(("pt_", "portuguese")) or name == "pt":
        return "pt"
    if name.startswith(("en_", "english")) or name == "en":
        return "en"
    return DEFAULT_LANGUAGE


def windows_language() -> str | None:
    try:
        import ctypes

        lcid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
    except (AttributeError, OSError):
        return None
    return locale.windows_locale.get(lcid)


def detect_language(environ=None, system: str | None = None) -> str:
    environ = os.environ if environ is None else environ
    system = system or platform.system()
    chosen = normalize(environ.get(ENV_VARIABLE))
    if chosen:
        return chosen
    # Same order as gettext: LANGUAGE is a list like "pt_BR:en"; the first entry wins.
    candidates = [(environ.get("LANGUAGE") or "").split(":")[0]]
    candidates += [environ.get(name) for name in ("LC_ALL", "LC_MESSAGES", "LANG")]
    if system == "Windows":
        candidates.insert(0, windows_language())
    try:
        candidates.append(locale.getlocale()[0])
    except ValueError:
        pass
    for candidate in candidates:
        found = normalize(candidate)
        if found:
            return found
    return DEFAULT_LANGUAGE


_language: str | None = None


def language() -> str:
    global _language
    if _language is None:
        _language = detect_language()
    return _language


def set_language(code: str | None) -> None:
    """Use this language from now on; None goes back to the system language."""
    global _language
    _language = normalize(code) if code else None


def t(key: str, **values) -> str:
    text = STRINGS[language()].get(key) or STRINGS[DEFAULT_LANGUAGE][key]
    return text.format(**values) if values else text


STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # Command line.
        "cli.description": "Vanish from your webcam with a hand gesture: Normal, Naruto, Goku, Particles or Ash.",
        "cli.camera": "Webcam index",
        "cli.width": "Video width",
        "cli.height": "Video height",
        "cli.fullscreen": "Start in fullscreen",
        "cli.no_mirror": "Do not mirror the webcam",
        "cli.fps": "Maximum frame rate of the virtual webcam (default: 30)",
        "cli.virtual_camera": "Require the virtual webcam; by default it is used if available. --no-virtual-camera turns it off",
        "cli.virtual_backend": "Virtual webcam driver",
        "cli.virtual_device": "Virtual device name",
        "cli.mode": "Effect to start with (default: the last one used); you can switch it in the panel",
        "cli.call_audio": "Send your voice plus the effect sounds to calls: 'VanishCam Mic' on Linux (PipeWire), VB-CABLE on Windows",
        "cli.static_background": "Do not update the background after capturing it (as in older versions)",
        "cli.background_image": "Background picture to start with (JPG/PNG; default: the last one used); you can switch it in the panel",
        "cli.background_metavar": "FILE",
        "cli.gesture_debug": "Print the gesture recognition state in the terminal (no text on the video)",
        "cli.lang": "Interface language (default: the system language)",
        "cli.fps_range": "The frame rate must be between 1 and 120.",
        # Startup, camera and models.
        "error": "ERROR: {error}",
        "camera.unavailable": "Webcam {index} is unavailable. Close other apps using the camera or try --camera 1.",
        "camera.lost": "The webcam stopped sending images: trying to reconnect...",
        "camera.back": "Webcam reconnected.",
        "reconnect.title": "RECONNECTING THE CAMERA",
        "reconnect.detail": "Webcam {index} · trying for {seconds} s",
        "camera.resolution": "Resolution: {width}x{height}",
        "virtual.unavailable": (
            "Virtual webcam unavailable. On Linux, load v4l2loopback (see the README); "
            "on Windows, install OBS Studio (OBS Virtual Camera driver) or Unity Capture "
            "and stop the OBS virtual camera if it is running. Details: {error}"
        ),
        "virtual.preview_only": "Continuing with the preview window only.",
        "virtual.active": "Virtual output active: {device} ({width}x{height}, up to {fps:g} FPS)",
        "models.downloading": "Downloading {name}...",
        "models.download_failed": "Could not download the MediaPipe models. Check your internet connection and try again.",
        "models.missing": "Model not found: {path}",
        "selftest.missing": "Missing files: {files}",
        "selftest.ok": "Self-test passed: models, images and audio are present.",
        "background.unreadable": "Could not open the background picture: {path}",
        "log.unreadable": "Log picture unavailable: {path}",
        "gesture.failed": "Gesture recognition failed: {error}",
        # Video loop.
        "loop.step_out": "Step out of the frame for 4 seconds: capturing the background.",
        "loop.step_out_again": "Step out of the frame for 4 seconds: capturing the new background.",
        "loop.capturing": "Capturing the background: {seconds} s.",
        "loop.background_ready": "Background ready. You can step back in front of the camera.",
        "loop.background_file": "Background: {path}",
        "loop.panel_hint": "Switch effect and background in the 'VanishCam' panel (or keys 1-{count}). Q quits.",
        "loop.mode_failed": "Could not open the {name} mode: {error}",
        "loop.image_not_added": "Picture not added: {error}",
        # Panel status line.
        "status.calibrating": "Capturing the background: step out ({seconds} s)",
        "status.ready": "Ready: make the gesture",
        "status.vanishing": "Vanishing…",
        "status.vanished": "Invisible",
        "status.appearing": "Reappearing…",
        "status.reconnecting": "Reconnecting the camera…",
        # Panel.
        "panel.effect": "EFFECT",
        "panel.background": "BACKGROUND",
        "panel.controls": "CONTROLS",
        "panel.active": "LIVE",
        "panel.add": "ADD",
        "panel.real": "REAL",
        "panel.image": "IMG {number:02d}",
        "panel.vanish": "VANISH",
        "panel.reappear": "REAPPEAR",
        "panel.recapture": "NEW BACKGROUND",
        "panel.fullscreen": "FULLSCREEN",
        "panel.window": "WINDOW",
        "panel.space": "SPACE",
        "panel.footer": "1–{count} EFFECT   V VANISH   R BACK   Q QUIT",
        # Effects: panel title, panel hint and the instructions printed when selected.
        "mode.normal.title": "Normal",
        "mode.normal.hint": "V sign — dissolve",
        "mode.normal.instructions": "Make a V with your index and middle fingers: you dissolve.",
        "mode.naruto.title": "Naruto",
        "mode.naruto.hint": "palms or clasped hands — log and smoke",
        "mode.naruto.instructions": "Press your palms together, or clasp your hands with the fingers interlaced, and hold for a moment: smoke and a log.",
        "mode.goku.title": "Goku",
        "mode.goku.hint": "two fingers on the forehead, 1 s — teleport",
        "mode.goku.instructions": "Index and middle fingers together on your forehead for 1 second: you vanish. Repeat to come back.",
        "mode.particles.title": "Particles",
        "mode.particles.hint": "salute on the brow — fragments in the wind",
        "mode.particles.instructions": "Salute (flat hand, fingers together on your brow) for a moment: you turn into fragments in the wind.",
        "mode.ash.title": "Ash",
        "mode.ash.hint": "hands up — turn to ash and crumble",
        "mode.ash.instructions": "Raise both open hands beside your head and hold for a moment: you turn to ash and crumble into a pile.",
        "goku.hold": "Hold the pose: {seconds:.2f}s",
        "goku.vanished": "Gone. Lower your hand and repeat the gesture to reappear.",
        "goku.no_sound": "No sound: put {name} in assets/ to enable it.",
        "goku.sound": "Sound: {name}",
        # Sounds and the call microphone.
        "sound.missing": "Audio file not found: {path}",
        "sound.no_player": "To play the effect sounds, install pygame or ffplay.",
        "sound.open_failed": "Could not open the audio: {error}",
        "sound.play_failed": "Could not play the audio: {error}",
        "sound.no_channel": "Could not play the audio: no free channel.",
        "mic.unavailable": "Virtual microphone unavailable: {error}",
        "mic.active": "Virtual microphone active: pick '{name}' as the microphone in your call app.",
        "mic.no_default": "No default microphone in PipeWire.",
        "mic.no_pipewire": "PipeWire unavailable (missing {tools}).",
        "mic.default_is_virtual": "The default microphone is already {name}: pick a real microphone.",
        "mic.already_running": "{name} already exists: close the other VanishCam window.",
        "mic.loopback_exited": "pw-loopback closed right after starting: check PipeWire.",
        "mic.timeout": "PipeWire did not create {name} within {seconds:g} s.",
        "mic.no_sounddevice": "sounddevice unavailable ({error}): reinstall the requirements.",
        "mic.no_default_input": "No default microphone in Windows.",
        "mic.no_cable": "VB-CABLE not found: install it from {url} and start VanishCam again.",
        "mic.cable_is_default_output": "Warning: '{name}' is the Windows default output, so you will not hear your computer. Set your speakers or headset back as the default in the sound settings.",
        "mic.no_mixer": "the audio mixer is not open",
        "mic.sound_unavailable": "The call will not hear {name}: {error}",
        # Preferences and the background gallery.
        "prefs.save_failed": "Could not save the preferences: {error}",
        "gallery.unsupported": "Unsupported format: {name}",
        "picker.title": "Choose a background picture",
        "picker.filter": "Pictures",
        "picker.missing": "No file picker available (install zenity).",
    },
    "pt": {
        "cli.description": "Suma da webcam com um gesto: Normal, Naruto, Goku, Partículas ou Cinzas.",
        "cli.camera": "Índice da webcam",
        "cli.width": "Largura do vídeo",
        "cli.height": "Altura do vídeo",
        "cli.fullscreen": "Abre em tela cheia",
        "cli.no_mirror": "Não espelha a webcam",
        "cli.fps": "FPS máximo da webcam virtual (padrão: 30)",
        "cli.virtual_camera": "Exige a webcam virtual; por padrão ela é usada se existir. --no-virtual-camera desliga",
        "cli.virtual_backend": "Driver da webcam virtual",
        "cli.virtual_device": "Nome do dispositivo virtual",
        "cli.mode": "Efeito inicial (padrão: o último usado); dá para trocar no painel",
        "cli.call_audio": "Manda sua voz e os sons dos efeitos para a chamada: 'VanishCam Mic' no Linux (PipeWire), VB-CABLE no Windows",
        "cli.static_background": "Não atualiza o fundo depois da captura (como nas versões antigas)",
        "cli.background_image": "Imagem de fundo inicial (JPG/PNG; padrão: a última usada); dá para trocar no painel",
        "cli.background_metavar": "ARQUIVO",
        "cli.gesture_debug": "Mostra no terminal o estado do reconhecimento do gesto (sem texto no vídeo)",
        "cli.lang": "Idioma da interface (padrão: o idioma do sistema)",
        "cli.fps_range": "O FPS deve estar entre 1 e 120.",
        "error": "ERRO: {error}",
        "camera.unavailable": "Webcam {index} indisponível. Feche outros apps que usam a câmera ou tente --camera 1.",
        "camera.lost": "A webcam parou de enviar imagens: tentando reconectar...",
        "camera.back": "Webcam reconectada.",
        "reconnect.title": "RECONECTANDO A CÂMERA",
        "reconnect.detail": "Webcam {index} · tentando há {seconds} s",
        "camera.resolution": "Resolução: {width}x{height}",
        "virtual.unavailable": (
            "Webcam virtual indisponível. No Linux, carregue o v4l2loopback (veja o README); "
            "no Windows, instale o OBS Studio (driver OBS Virtual Camera) ou o Unity Capture "
            "e desligue a câmera virtual do OBS se estiver ligada. Detalhe: {error}"
        ),
        "virtual.preview_only": "Continuando só com a janela de preview.",
        "virtual.active": "Saída virtual ativa: {device} ({width}x{height}, até {fps:g} FPS)",
        "models.downloading": "Baixando {name}...",
        "models.download_failed": "Não consegui baixar os modelos do MediaPipe. Verifique a conexão com a internet e tente de novo.",
        "models.missing": "Modelo não encontrado: {path}",
        "selftest.missing": "Arquivos faltando: {files}",
        "selftest.ok": "Autoteste concluído: modelos, imagens e áudio presentes.",
        "background.unreadable": "Não foi possível abrir a imagem de fundo: {path}",
        "log.unreadable": "Imagem do tronco indisponível: {path}",
        "gesture.failed": "O reconhecimento do gesto falhou: {error}",
        "loop.step_out": "Saia da frente da câmera por 4 segundos: capturando o fundo.",
        "loop.step_out_again": "Saia da frente da câmera por 4 segundos: capturando o novo fundo.",
        "loop.capturing": "Capturando o fundo: {seconds} s.",
        "loop.background_ready": "Fundo pronto. Pode voltar para a frente da câmera.",
        "loop.background_file": "Fundo: {path}",
        "loop.panel_hint": "Troque efeito e fundo no painel 'VanishCam' (ou teclas 1-{count}). Q fecha.",
        "loop.mode_failed": "Não consegui abrir o modo {name}: {error}",
        "loop.image_not_added": "Imagem não adicionada: {error}",
        "status.calibrating": "Capturando o fundo: saia da câmera ({seconds} s)",
        "status.ready": "Pronto: faça o gesto",
        "status.vanishing": "Sumindo…",
        "status.vanished": "Invisível",
        "status.appearing": "Reaparecendo…",
        "status.reconnecting": "Reconectando a câmera…",
        "panel.effect": "EFEITO",
        "panel.background": "FUNDO",
        "panel.controls": "CONTROLES",
        "panel.active": "ATIVO",
        "panel.add": "ADICIONAR",
        "panel.real": "REAL",
        "panel.image": "IMG {number:02d}",
        "panel.vanish": "SUMIR",
        "panel.reappear": "REAPARECER",
        "panel.recapture": "NOVO FUNDO",
        "panel.fullscreen": "TELA CHEIA",
        "panel.window": "JANELA",
        "panel.space": "ESPAÇO",
        "panel.footer": "1–{count} EFEITO   V SUMIR   R VOLTAR   Q SAIR",
        "mode.normal.title": "Normal",
        "mode.normal.hint": "sinal de V — dissolução",
        "mode.normal.instructions": "Faça um V com indicador e médio: você se dissolve.",
        "mode.naruto.title": "Naruto",
        "mode.naruto.hint": "palmas ou mãos entrelaçadas — tronco e fumaça",
        "mode.naruto.instructions": "Junte as palmas, ou entrelace os dedos das mãos, e segure um instante: fumaça e tronco.",
        "mode.goku.title": "Goku",
        "mode.goku.hint": "dois dedos na testa, 1 s — teletransporte",
        "mode.goku.instructions": "Indicador e médio juntos na testa por 1 segundo: você some. Repita para reaparecer.",
        "mode.particles.title": "Partículas",
        "mode.particles.hint": "continência na testa — fragmentos no vento",
        "mode.particles.instructions": "Bata continência (mão reta, dedos unidos na testa) por um instante: você vira fragmentos no vento.",
        "mode.ash.title": "Cinzas",
        "mode.ash.hint": "mãos para o alto — vira cinza e desmorona",
        "mode.ash.instructions": "Levante as duas mãos abertas ao lado da cabeça e segure um instante: você vira cinza e desmorona num monte.",
        "goku.hold": "Segure a pose: {seconds:.2f}s",
        "goku.vanished": "Sumiu. Abaixe a mão e repita o gesto para reaparecer.",
        "goku.no_sound": "Sem som: coloque {name} em assets/ para ativar.",
        "goku.sound": "Som: {name}",
        "sound.missing": "Áudio não encontrado: {path}",
        "sound.no_player": "Para tocar os sons do efeito instale o pygame ou o ffplay.",
        "sound.open_failed": "Não foi possível abrir o áudio: {error}",
        "sound.play_failed": "Não foi possível tocar o áudio: {error}",
        "sound.no_channel": "Não foi possível tocar o áudio: nenhum canal livre.",
        "mic.unavailable": "Microfone virtual indisponível: {error}",
        "mic.active": "Microfone virtual ativo: escolha '{name}' como microfone no app da chamada.",
        "mic.no_default": "Nenhum microfone padrão no PipeWire.",
        "mic.no_pipewire": "PipeWire indisponível (falta {tools}).",
        "mic.default_is_virtual": "O microfone padrão já é o {name}: escolha um microfone real.",
        "mic.already_running": "O {name} já existe: feche a outra janela do VanishCam.",
        "mic.loopback_exited": "O pw-loopback fechou logo ao abrir: verifique o PipeWire.",
        "mic.timeout": "O PipeWire não criou {name} em {seconds:g} s.",
        "mic.no_sounddevice": "sounddevice indisponível ({error}): reinstale as dependências.",
        "mic.no_default_input": "Nenhum microfone padrão no Windows.",
        "mic.no_cable": "VB-CABLE não encontrado: instale em {url} e abra o VanishCam de novo.",
        "mic.cable_is_default_output": "Atenção: '{name}' é a saída padrão do Windows, então você não vai ouvir o computador. Volte seus alto-falantes ou fone como padrão nas configurações de som.",
        "mic.no_mixer": "o mixer de áudio não está aberto",
        "mic.sound_unavailable": "A chamada não vai ouvir {name}: {error}",
        "prefs.save_failed": "Não consegui salvar as preferências: {error}",
        "gallery.unsupported": "Formato não suportado: {name}",
        "picker.title": "Escolha uma imagem de fundo",
        "picker.filter": "Imagens",
        "picker.missing": "Nenhuma janela de seleção de arquivo disponível (instale o zenity).",
    },
}
