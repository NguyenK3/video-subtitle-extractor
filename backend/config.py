# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao
@Time    : 2021/3/24 9:36 AM
@FileName: config.py
@desc: Project configuration file. Adjust parameters here to trade time for accuracy or vice versa.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import configparser
import os
import re
import time
from pathlib import Path
from fsplit.filesplit import Filesplit
import paddle
from tools.constant import *

# Project version
VERSION = "2.0.3"

# Project base directory
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)

# ==================== [DO NOT MODIFY] Read configuration files START ====================
# Read settings.ini configuration
settings_config = configparser.ConfigParser()
MODE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini')
if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini')):
    # If no config file exists, default to English interface
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini'), mode='w', encoding='utf-8') as f:
        f.write('[DEFAULT]\n')
        f.write('Interface = English\n')
        f.write('Language = ch\n')
        f.write('Mode = fast')
settings_config.read(MODE_CONFIG_PATH, encoding='utf-8')

# Read interface language configuration, e.g. ch.ini
interface_config = configparser.ConfigParser()
INTERFACE_KEY_NAME_MAP = {
    'English': 'en',
    'Simplified Chinese': 'ch',
    'Traditional Chinese': 'chinese_cht',
    'Korean': 'ko',
    'Japanese': 'japan',
    'Vietnamese': 'vi',
    'Spanish': 'es'
}
interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface',
                              f"{INTERFACE_KEY_NAME_MAP[settings_config['DEFAULT']['Interface']]}.ini")
interface_config.read(interface_file, encoding='utf-8')
# ==================== [DO NOT MODIFY] Read configuration files END ====================


# ==================== [DO NOT MODIFY] Validate program path START ====================
# If the path contains non-ASCII characters or spaces, the program may have bugs
# Default to valid path
IS_LEGAL_PATH = True
# If path contains Chinese characters, mark as invalid
if re.search(r"[\u4e00-\u9fa5]+", BASE_DIR):
    IS_LEGAL_PATH = False
# If path contains spaces, mark as invalid
if re.search(r"\s", BASE_DIR):
    IS_LEGAL_PATH = False
# If path is invalid, keep warning the user
while not IS_LEGAL_PATH:
    print(interface_config['Main']['IllegalPathWarning'])
    time.sleep(3)
# ==================== [DO NOT MODIFY] Validate program path END ====================


# ==================== [DO NOT MODIFY] Detect GPU availability START ====================
# Whether to use GPU (Nvidia)
USE_GPU = False
# If paddlepaddle was compiled with GPU support
if paddle.is_compiled_with_cuda():
    # Check if a GPU is available
    if len(paddle.static.cuda_places()) > 0:
        # Use GPU if available
        USE_GPU = True

# Whether to use ONNX (DirectML/AMD/Intel)
ONNX_PROVIDERS = []
if USE_GPU == False:
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        for provider in available_providers:
            if provider in [
                "CPUExecutionProvider"
            ]:
                continue
            if provider not in [
                "DmlExecutionProvider",         # DirectML, for Windows GPU
                "ROCMExecutionProvider",        # AMD ROCm
                "MIGraphXExecutionProvider",    # AMD MIGraphX
                # "VitisAIExecutionProvider",   # AMD VitisAI, for RyzenAI & Windows
                "OpenVINOExecutionProvider",    # Intel GPU
                "MetalExecutionProvider",       # Apple macOS
                "CoreMLExecutionProvider",      # Apple macOS
                "CUDAExecutionProvider",        # Nvidia GPU
            ]:
                print(interface_config['Main']['OnnxExectionProviderNotSupportedSkipped'].format(provider))
                continue
            print(interface_config['Main']['OnnxExecutionProviderDetected'].format(provider))
            ONNX_PROVIDERS.append(provider)
    except ModuleNotFoundError as e:
        print(interface_config['Main']['OnnxRuntimeNotInstall'])
if len(ONNX_PROVIDERS) > 0:
    USE_GPU = True
# ==================== [DO NOT MODIFY] Detect GPU availability END ====================


# ==================== [DO NOT MODIFY] Read language, model paths, dictionary paths START ====================
# Set recognition language
REC_CHAR_TYPE = settings_config['DEFAULT']['Language']

# Set recognition mode
MODE_TYPE = settings_config['DEFAULT']['Mode']
ACCURATE_MODE_ON = False
if MODE_TYPE == 'accurate':
    ACCURATE_MODE_ON = True
if MODE_TYPE == 'fast':
    ACCURATE_MODE_ON = False
if MODE_TYPE == 'auto':
    if USE_GPU:
        ACCURATE_MODE_ON = True
    else:
        ACCURATE_MODE_ON = False
# Model file directory
# Default model version V4
MODEL_VERSION = 'V4'
# Text detection model
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# Set text recognition model + dictionary
REC_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# V3, V4 models default image recognition shape is 3, 48, 320
REC_IMAGE_SHAPE = '3,48,320'
REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_det')

LATIN_LANG = [
    'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
    'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
    'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
    'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french'
]
ARABIC_LANG = ['ar', 'fa', 'ug', 'ur']
CYRILLIC_LANG = [
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic'
]
DEVANAGARI_LANG = [
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc', 'devanagari'
]
OTHER_LANG = [
    'ch', 'japan', 'korean', 'en', 'ta', 'kn', 'te', 'ka',
    'chinese_cht',
]
MULTI_LANG = LATIN_LANG + ARABIC_LANG + CYRILLIC_LANG + DEVANAGARI_LANG + \
             OTHER_LANG

DET_MODEL_FAST_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')


# If a text recognition language type is set, configure accordingly
if REC_CHAR_TYPE in MULTI_LANG:
    # Define text detection and recognition models
    # In fast mode, use lightweight models
    if MODE_TYPE == 'fast':
        DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    # In auto mode, check GPU availability and select model accordingly
    elif MODE_TYPE == 'auto':
        # If using GPU, use the large model
        if USE_GPU:
            DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
            # For English mode, the ch model performs better than fast
            if REC_CHAR_TYPE == 'en':
                REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'ch_rec')
            else:
                REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
        else:
            DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')
            REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    else:
        DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
    # If default version (V4) has no large model, fall back to V4 fast model
    if not os.path.exists(REC_MODEL_PATH):
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    # If V4 has neither large nor fast model, use V3 large model
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V3'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
    # If V3 has no large model, use V3 fast model
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V3'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')

    if REC_CHAR_TYPE in LATIN_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'latin_rec_fast')
    elif REC_CHAR_TYPE in ARABIC_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'arabic_rec_fast')
    elif REC_CHAR_TYPE in CYRILLIC_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'cyrillic_rec_fast')
    elif REC_CHAR_TYPE in DEVANAGARI_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'devanagari_rec_fast')

    # Define image recognition shape
    if MODEL_VERSION == 'V2':
        REC_IMAGE_SHAPE = '3,32,320'
    else:
        REC_IMAGE_SHAPE = '3,48,320'

    # Check if the full model file exists; if not, merge small files to generate it
    if 'inference.pdiparams' not in (os.listdir(REC_MODEL_PATH)):
        fs = Filesplit()
        fs.merge(input_dir=REC_MODEL_PATH)
    # Check if the full detection model file exists; if not, merge small files
    if 'inference.pdiparams' not in (os.listdir(DET_MODEL_PATH)):
        fs = Filesplit()
        fs.merge(input_dir=DET_MODEL_PATH)
# ==================== [DO NOT MODIFY] Read language, model paths, dictionary paths END ====================


# --------------------- Adjust these settings as needed START -----------------
# Whether to generate TXT text subtitles
GENERATE_TXT = True

# Number of text boxes recognized simultaneously per image; increase with more GPU VRAM
REC_BATCH_NUM = 6
# DB algorithm batch size, default 10
MAX_BATCH_SIZE = 10

# Default subtitle area location
DEFAULT_SUBTITLE_AREA = SubtitleArea.UNKNOWN

# Frames per second to extract for OCR
EXTRACT_FREQUENCY = 3

# Pixel tolerance for detection box deviation
PIXEL_TOLERANCE_Y = 50  # Allow 50px vertical deviation
PIXEL_TOLERANCE_X = 100  # Allow 100px horizontal deviation

# Subtitle area offset in pixels
SUBTITLE_AREA_DEVIATION_PIXEL = 50

# Most likely watermark area count
WATERMARK_AREA_NUM = 5

# Text similarity threshold
# Used for deduplication to determine if two subtitle lines are the same. Higher = stricter.
# Dynamic algorithm: lower threshold for short text, higher for long text
THRESHOLD_TEXT_SIMILARITY = 0.8

# Drop OCR results with confidence below 0.75
DROP_SCORE = 0.75

# Subtitle area deviation tolerance; 0 = no overflow, 0.03 = 3% overflow allowed
SUB_AREA_DEVIATION_RATE = 0

# Output lost subtitle frames (only effective for CJK languages); debug output to: video_path/loss
DEBUG_OCR_LOSS = False

# Whether to keep cache data for debugging
DEBUG_NO_DELETE_CACHE = False

# Whether to delete empty timestamps
DELETE_EMPTY_TIMESTAMP = True

# Whether to re-segment words (for languages without spaces)
WORD_SEGMENTATION = True

# --------------------- Adjust these settings as needed END -----------------------------

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
