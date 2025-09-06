from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import math
import uuid
import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy.fftpack import fft
from scipy.cluster.vq import kmeans2
from scipy.signal import find_peaks
import pretty_midi
from collections import Counter
from werkzeug.utils import secure_filename
import logging
import tempfile
import threading
import random
import sys
import traceback
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')

# 详细的CORS配置
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept"]
    }
})

# 配置
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['AUDIO_OUTPUT_FOLDER'] = os.environ.get('AUDIO_OUTPUT_FOLDER', 'outputs')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024
model_path = os.path.join(os.path.dirname(__file__), "music_generation_model_final.pth")
app.config['MODEL_PATH'] = model_path

app.config['MUSIC_DURATION'] = 10
app.config['SAMPLE_RATE'] = 44100
app.config['BASE_PITCH_MIDI'] = 60  # C4
app.config['PITCH_RANGE_SEMITONES'] = 12  # 一个八度

# 添加和声配置
HARMONY_CONFIG = {
    'harmony_progressions': {
        '欢快': ['I', 'IV', 'V', 'I'],
        '平和': ['I', 'vi', 'IV', 'V'],
        '静谧': ['I', 'iii', 'vi', 'IV'],
        '华丽': ['I', 'V', 'vi', 'III', 'IV', 'I', 'IV', 'V'],
        '舒缓': ['I', 'IV', 'I', 'V']
    },
    'chord_types': {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'seventh': [0, 4, 7, 10],
        'major_seventh': [0, 4, 7, 11],
        'minor_seventh': [0, 3, 7, 10]
    },
    'roman_to_interval': {
        'I': 0,    # 主音
        'ii': 2,   # 上主音
        'iii': 4,  # 中音
        'IV': 5,   # 下属音
        'V': 7,    # 属音
        'vi': 9,   # 下中音
        'vii': 11  # 导音
    }
}

APP_CONFIG = {
    'mode_intervals': {'C_major': [0, 2, 4, 5, 7, 9, 11]},
    'color_style_map': {
        '0-60': {'style': '欢快', 'scale': 'pentatonic', 'tempo': 120, 'style_idx': 0},
        '90-150': {'style': '平和', 'scale': 'jiao_mode', 'tempo': 80, 'style_idx': 1},
        '180-240': {'style': '静谧', 'scale': 'yu_mode', 'tempo': 60, 'style_idx': 2},
        'high_saturation': {'style': '华丽', 'scale': 'mixed_mode', 'style_idx': 3},
        'low_saturation': {'style': '舒缓', 'scale': 'simple_pentatonic', 'style_idx': 4}
    },
    'scale_intervals': {
        'pentatonic': [0, 2, 4, 7, 9],
        'jiao_mode': [0, 3, 5, 7, 10],
        'yu_mode': [0, 2, 5, 7, 10],
        'mixed_mode': [0, 2, 3, 5, 7, 8, 10],
        'simple_pentatonic': [0, 2, 4, 7, 9]
    }
}

# 确保目录存在
for dir_path in [app.config['UPLOAD_FOLDER'], app.config['AUDIO_OUTPUT_FOLDER']]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"创建文件夹：{dir_path}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def cleanup_temp_file(file_path, delay=300):
    def delete_file():
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"清理临时文件：{file_path}")
            except Exception as e:
                logger.warning(f"临时文件清理失败：{str(e)}")
    threading.Timer(delay, delete_file).start()

class EnhancedMusicGenerationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3, style_embedding_size=16):
        super(EnhancedMusicGenerationModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.style_embedding = nn.Embedding(5, style_embedding_size)
        self.lstm = nn.LSTM(
            input_size + style_embedding_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.MultiheadAttention(hidden_size, 4, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, style_indices, hidden=None):
        batch_size, seq_len, _ = x.size()
        style_emb = self.style_embedding(style_indices)
        style_emb = style_emb.unsqueeze(1).repeat(1, seq_len, 1)
        x = torch.cat([x, style_emb], dim=2)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = out.transpose(0, 1)
        attn_out, _ = self.attention(out, out, out)
        out = out + attn_out
        out = self.layer_norm(out)
        out = out.transpose(0, 1)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
            weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        )
        return hidden

ai_model = None
note_to_idx = None
idx_to_note = None
model_loaded = False

def safe_load_ai_model():
    global model_loaded, ai_model, note_to_idx, idx_to_note
    if model_loaded:
        return
    model_path = app.config['MODEL_PATH']
    
    if not os.path.exists(model_path):
        logger.warning(f"未找到AI模型：{model_path}，启用简单模式")
        note_to_idx = {60: 0, 62: 1, 64: 2, 65: 3, 67: 4, 69: 5, 71: 6}
        idx_to_note = {v: k for k, v in note_to_idx.items()}
        model_loaded = True
        return
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        note_to_idx = checkpoint['note_to_idx']
        idx_to_note = checkpoint['idx_to_note']
        vocab_size = len(note_to_idx)
        
        ai_model = EnhancedMusicGenerationModel(
            input_size=vocab_size,
            hidden_size=512,
            output_size=vocab_size,
            num_layers=3
        )
        ai_model.load_state_dict(checkpoint['model_state_dict'])
        ai_model.eval()
        logger.info("AI模型加载成功")
        model_loaded = True
    except Exception as e:
        logger.error(f"AI模型加载失败：{str(e)}，启用简单模式")
        note_to_idx = {60: 0, 62: 1, 64: 2, 65: 3, 67: 4, 69: 5, 71: 6}
        idx_to_note = {v: k for k, v in note_to_idx.items()}
        model_loaded = True

@app.before_request
def load_model_once():
    # 跳过OPTIONS请求的模型加载
    if request.method != 'OPTIONS':
        safe_load_ai_model()

def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

app.after_request(add_cors_headers)

def ai_edge_detection(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"非遗图片读取失败：{image_path}")
        
        max_dim = 800
        height, width = img.shape[:2]
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        line_coords = []
        height, width = edges.shape
        for x in range(width):
            col = edges[:, x]
            y_indices = np.where(col == 255)[0]
            if len(y_indices) > 0:
                y = height - np.min(y_indices)
                line_coords.append((x, y))
        
        line_coords = np.array(line_coords)
        if len(line_coords) < 750:
            x = np.arange(0, 750)
            y = 50 + 20 * np.sin(x / 20 * 2 * np.pi)
            line_coords = np.column_stack((x, y))
        
        logger.info(f"非遗图片描边完成：提取线条坐标{len(line_coords)}个")
        return line_coords, img
    except Exception as e:
        logger.error(f"AI描边出错：{str(e)}（启用默认线条）")
        x = np.arange(0, 750)
        y = 50 + 20 * np.sin(x / 20 * 2 * np.pi)
        line_coords = np.column_stack((x, y))
        default_img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        cv2.putText(default_img, "非遗图片异常", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return line_coords, default_img

def trigonometric_pitch_generation(line_coords):
    # 计算线条长度L和总时间T_total
    L = len(line_coords) if len(line_coords) > 0 else 200
    T_total = app.config['MUSIC_DURATION']
    t_step = T_total / L if L > 0 else 0.025
    time_list = np.arange(0, T_total, t_step)

    # 提取线条的y坐标并归一化到[-1, 1]范围
    if len(line_coords) > 0:
        y_original = line_coords[:, 1]
        y_min, y_max = np.min(y_original), np.max(y_original)
        if y_max - y_min > 0:
            normalized_height = 2 * (y_original - y_min) / (y_max - y_min) - 1
        else:
            normalized_height = np.zeros_like(y_original)
    else:
        # 默认使用正弦波
        normalized_height = np.sin(np.linspace(0, 2 * np.pi, len(time_list)))

    # 使用平滑因子使曲线更加平滑
    smooth_factor = 5
    normalized_height = np.convolve(normalized_height, np.ones(smooth_factor) / smooth_factor, mode='same')

    # 应用论文中的公式：y(t) = C + A_semitones * Normalized_Height(t)
    C = app.config['BASE_PITCH_MIDI']
    A_semitones = app.config['PITCH_RANGE_SEMITONES']
    pitch_midi = C + A_semitones * normalized_height

    # 限制音高在合理范围内(21-108)
    pitch_midi = np.clip(pitch_midi, 21, 108)
    pitch_midi = np.round(pitch_midi).astype(int)

    # 添加节奏变化，每0.1秒一个音符
    rhythm_interval = 0.1
    rhythm_indices = np.arange(0, len(time_list), max(1, int(rhythm_interval / t_step)))
    final_pitch = pitch_midi[rhythm_indices] if len(rhythm_indices) > 0 else [C]
    final_time = time_list[rhythm_indices] if len(rhythm_indices) > 0 else [0]

    logger.info(f"三角函数音高生成完成：{len(final_pitch)}个音符")
    return final_pitch, final_time

def midi_to_freq(midi):
    """将MIDI音符转换为频率"""
    return 440 * (2 ** ((midi - 69) / 12))

def freq_to_midi(freq):
    """将频率转换为MIDI音符（使用论文中的十二平均律公式）"""
    if freq <= 0:
        return app.config['BASE_PITCH_MIDI']
    return round(12 * math.log2(freq / 440) + 69)

def fourier_transform_main_pitch(pitch_midi_list):
    if len(pitch_midi_list) == 0:
        return app.config['BASE_PITCH_MIDI']
    
    # 将MIDI音符转换为频率并创建时域信号
    sample_rate = app.config['SAMPLE_RATE']
    duration_per_note = 0.1
    total_samples = int(sample_rate * duration_per_note * len(pitch_midi_list))
    signal = np.zeros(total_samples)

    # 为每个音符创建正弦波
    for i, midi in enumerate(pitch_midi_list):
        freq = midi_to_freq(midi)
        start = int(i * duration_per_note * sample_rate)
        end = int((i + 1) * duration_per_note * sample_rate)
        t = np.linspace(0, duration_per_note, end - start, endpoint=False)
        signal[start:end] = np.sin(2 * np.pi * freq * t)

    # 执行FFT（使用2的幂次方长度以提高效率）
    n_fft = 2 ** (int(np.log2(len(signal))) + 1)
    fft_result = fft(signal, n_fft)
    magnitude = np.abs(fft_result)[:n_fft // 2]
    freq_axis = np.fft.fftfreq(n_fft, 1 / sample_rate)[:n_fft // 2]

    # 找到峰值频率
    peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * 0.3)
    candidate_freqs = freq_axis[peaks]

    # 将频率转换回MIDI音符
    candidate_midi = []
    for freq in candidate_freqs:
        if freq <= 0:
            continue
        midi_note = freq_to_midi(freq)
        midi_note = np.clip(midi_note, 21, 108)
        candidate_midi.append(midi_note)

    if not candidate_midi:
        return app.config['BASE_PITCH_MIDI']
    
    # 选择出现次数最多的音符作为主音
    midi_counter = Counter(candidate_midi)
    main_pitch = midi_counter.most_common(1)[0][0]

    logger.info(f"傅里叶变换完成：提取主音MIDI={main_pitch}")
    return main_pitch

def generate_harmony_for_ai(main_pitch, style_name='平和'):
    # 选择和声进行
    progression = HARMONY_CONFIG['harmony_progressions'].get(style_name, HARMONY_CONFIG['harmony_progressions']['平和'])
    
    # 选择和弦类型
    chord_type = 'major'
    if style_name in ['静谧', '舒缓']:
        chord_type = 'minor'
    elif style_name == '华丽':
        chord_type = random.choice(['major_seventh', 'minor_seventh'])
    
    # 生成和弦序列
    harmony_sequence = []
    for chord_roman in progression:
        # 获取和弦根音
        root_interval = HARMONY_CONFIG['roman_to_interval'].get(chord_roman, 0)
        root_note = main_pitch + root_interval
        
        # 确保根音在合理范围内
        root_note = max(21, min(108, root_note))
        
        # 获取和弦类型
        chord_intervals = HARMONY_CONFIG['chord_types'][chord_type]
        
        # 生成和弦音符
        chord_notes = [root_note + interval for interval in chord_intervals]
        
        # 确保所有音符在合理范围内
        chord_notes = [max(21, min(108, note)) for note in chord_notes]
        
        harmony_sequence.append(chord_notes)
    
    return harmony_sequence

def cluster_pitch_style(line_coords, main_pitch, image_path):
    # 第一部分：音高聚类筛选
    mode_intervals = APP_CONFIG['mode_intervals']['C_major']
    mode_midis = [main_pitch + interval for interval in mode_intervals]
    mode_midis += [m + 12 for m in mode_midis] + [m - 12 for m in mode_midis]
    mode_midis = list(set(np.clip(mode_midis, 21, 108)))
    mode_midis.sort()

    # 从线条坐标提取候选音高
    if len(line_coords) > 0:
        candidate_pitches = line_coords[:, 1]
        y_min, y_max = np.min(candidate_pitches), np.max(candidate_pitches)
        if y_max - y_min > 0:
            normalized = 2 * (candidate_pitches - y_min) / (y_max - y_min) - 1
        else:
            normalized = np.zeros_like(candidate_pitches)
        candidate_midis = app.config['BASE_PITCH_MIDI'] + app.config['PITCH_RANGE_SEMITONES'] * normalized
        candidate_midis = np.round(np.clip(candidate_midis, 21, 108)).astype(int)
    else:
        candidate_midis = [main_pitch]

    # 筛选符合调式的音高
    valid_pitches = []
    for midi in candidate_midis:
        # 计算与调式内各音的最小半音差
        min_diff = min([abs(midi - mode_midi) for mode_midi in mode_midis])
        if min_diff <= 1:  # 允许1个半音的误差
            valid_pitches.append(midi)
    
    # 如果没有有效音高，使用调式音阶
    if not valid_pitches:
        valid_pitches = mode_midis[:len(candidate_midis)] if len(mode_midis) >= len(candidate_midis) else mode_midis

    # 第二部分：颜色聚类确定风格
    try:
        img = cv2.imread(image_path)
        if img is None:
            return valid_pitches, APP_CONFIG['color_style_map']['90-150']
        
        # 调整图像大小以提高聚类效率
        img_small = cv2.resize(img, (32, 32))
        hsv_img = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        pixels = hsv_img.reshape(-1, 3).astype(np.float32)

        # 使用K-means聚类找到主色调
        k = 2
        centroids, labels = kmeans2(pixels, k, iter=10, minit='points')
        label_count = Counter(labels)
        main_centroid_idx = label_count.most_common(1)[0][0]
        main_h, main_s, main_v = centroids[main_centroid_idx]

        # 根据色相和饱和度确定音乐风格
        main_h = int(main_h)
        main_s = int(main_s)
        style_config = None

        # 根据色相确定基本风格
        if 0 <= main_h <= 60:
            style_config = APP_CONFIG['color_style_map']['0-60']
        elif 90 <= main_h <= 150:
            style_config = APP_CONFIG['color_style_map']['90-150']
        elif 180 <= main_h <= 240:
            style_config = APP_CONFIG['color_style_map']['180-240']
        
        # 根据饱和度调整风格
        if style_config is None:
            if main_s >= 150:  # 高饱和度
                style_config = APP_CONFIG['color_style_map']['high_saturation']
            else:  # 低饱和度
                style_config = APP_CONFIG['color_style_map']['low_saturation']
        elif main_s >= 150:  # 高饱和度增强效果
            if style_config['style'] == '欢快':
                style_config['tempo'] += 20
            elif style_config['style'] == '舒缓':
                style_config['tempo'] -= 10

        logger.info(f"聚类算法完成：有效音高{len(valid_pitches)}个，匹配风格={style_config['style']}")
        return valid_pitches, style_config
    except Exception as e:
        logger.error(f"颜色聚类失败：{str(e)}，使用默认风格")
        return valid_pitches, APP_CONFIG['color_style_map']['90-150']
    
def generate_music(valid_pitches, style_config, main_pitch):
    """
    根据有效音高和风格配置生成音乐，包含和声
    """
    # 创建MIDI对象并设置速度
    midi = pretty_midi.PrettyMIDI(initial_tempo=style_config['tempo'])
    
    # 根据风格选择主旋律乐器
    instrument_program = {
        '欢快': 0,    # 大钢琴
        '平和': 1,    # 亮音大钢琴
        '静谧': 41,   # 小提琴
        '华丽': 49,   # 弦乐合奏
        '舒缓': 80    # 合成主音
    }.get(style_config['style'], 0)
    
    instrument = pretty_midi.Instrument(program=instrument_program)

    # 根据风格选择和声乐器
    harmony_instrument_program = {
        '欢快': 48,   # 弦乐合奏1
        '平和': 52,   # 合唱音效
        '静谧': 46,   # 竖琴
        '华丽': 56,   # 小号
        '舒缓': 91    # 合成背景音
    }.get(style_config['style'], 48)
    
    harmony_instrument = pretty_midi.Instrument(program=harmony_instrument_program)

    # 根据风格选择音阶
    scale_intervals = APP_CONFIG['scale_intervals'][style_config['scale']]
    scale_midis = [main_pitch + interval for interval in scale_intervals]
    scale_midis = list(set(np.clip(scale_midis, 21, 108)))
    scale_midis.sort()

    # 生成主旋律音符序列
    start_time = 0.0
    note_duration = 0.5  # 每个音符持续0.5秒
    
    for i in range(len(valid_pitches)):
        # 确保音符在音阶内
        pitch = valid_pitches[i] if valid_pitches[i] in scale_midis else np.random.choice(scale_midis)
        
        # 创建音符
        note = pretty_midi.Note(
            velocity=np.random.randint(60, 81),  # 随机力度
            pitch=pitch,
            start=start_time,
            end=start_time + note_duration
        )
        instrument.notes.append(note)
        start_time += 0.5  # 每0.5秒一个音符

        # 确保不超过总时长
        if start_time >= app.config['MUSIC_DURATION']:
            break

    # 生成和声
    harmony_sequence = generate_harmony_for_ai(main_pitch, style_config['style'])
    harmony_start_time = 0.0
    harmony_duration = 2.0  # 每个和弦持续2秒
    
    for chord in harmony_sequence:
        # 确保不超过总时长
        if harmony_start_time >= app.config['MUSIC_DURATION']:
            break
            
        for note in chord:
            harmony_note = pretty_midi.Note(
                velocity=70,  # 和声力度稍小
                pitch=note - 12,  # 降低一个八度，避免与主旋律冲突
                start=harmony_start_time,
                end=harmony_start_time + harmony_duration
            )
            harmony_instrument.notes.append(harmony_note)
        
        harmony_start_time += harmony_duration

    # 添加乐器到MIDI对象
    midi.instruments.append(instrument)
    midi.instruments.append(harmony_instrument)
    
    # 保存为临时MIDI文件
    temp_midi_path = os.path.join(tempfile.gettempdir(), f"heritage_music_{uuid.uuid4()}.mid")
    midi.write(temp_midi_path)
    
    # 加载MIDI并渲染成WAV音频数据
    from scipy.io import wavfile
    midi_data = pretty_midi.PrettyMIDI(temp_midi_path)
    # 渲染音频
    audio_data = midi_data.synthesize(fs=app.config['SAMPLE_RATE'])
    # 音频数据归一化
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # 保存为WAV文件（使用配置的输出目录，而非系统临时目录）
    temp_wav_path = os.path.join(app.config['AUDIO_OUTPUT_FOLDER'], f"heritage_music_{uuid.uuid4()}.wav")
    wavfile.write(temp_wav_path, app.config['SAMPLE_RATE'], audio_data)
    # 清理临时MIDI文件
    os.remove(temp_midi_path)
    # 延迟清理WAV文件，给前端足够时间下载
    cleanup_temp_file(temp_wav_path, delay=1800)  # 30分钟后清理

    logger.info(f"WAV音乐生成完成：路径={temp_wav_path}，风格={style_config['style']}")
    return temp_wav_path

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')

# 对静态文件的显式路由
@app.route('/<path:path>', methods=['GET'])
def static_files(path):
    file_path = os.path.join('.', path)
    if os.path.isfile(file_path):
        return send_from_directory('.', path)
    
    # 处理特殊路径
    if path in ['', '/']:
        return send_from_directory('.', 'index.html')
    
    return jsonify({"code": 404, "msg": "资源不存在"}), 404

@app.route('/test', methods=['POST', 'OPTIONS'])
def test_endpoint():
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        return add_cors_headers(response)
    
    if request.method == 'POST':
        return jsonify({"message": "POST请求成功", "data": request.json if request.json else {}})

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        return add_cors_headers(response)
    
    try:
        # 检查是否有文件部分
        if 'file' not in request.files:
            logger.error("请求中未找到 'file' 字段")
            logger.error(f"请求中的文件字段: {list(request.files.keys())}")
            return jsonify({"code": 400, "msg": "未上传非遗图片"}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("文件名为空")
            return jsonify({"code": 400, "msg": "未选择图片"}), 400

        if not allowed_file(file.filename):
            logger.error(f"不支持的文件格式: {file.filename}")
            return jsonify({"code": 400, "msg": "不支持的图片格式（仅支持png/jpg/jpeg/bmp）"}), 400

        filename = secure_filename(f"heritage_{uuid.uuid4()}_{file.filename}")
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"尝试保存文件到: {upload_path}")
        file.save(upload_path)
        logger.info("文件保存成功")
        
        # 延迟清理上传的文件
        cleanup_temp_file(upload_path)

        try:
            logger.info("开始AI边缘检测")
            line_coords, _ = ai_edge_detection(upload_path)
            logger.info(f"边缘检测完成，获取到 {len(line_coords)} 个坐标点")
            
            logger.info("开始三角函数音高生成")
            pitch_midi_list, _ = trigonometric_pitch_generation(line_coords)
            logger.info(f"音高生成完成，生成 {len(pitch_midi_list)} 个音符")
            
            logger.info("开始傅里叶变换提取主音")
            main_pitch = fourier_transform_main_pitch(pitch_midi_list)
            logger.info(f"主音提取完成: {main_pitch}")
            
            logger.info("开始聚类算法筛选音高并确定风格")
            valid_pitches, style_config = cluster_pitch_style(line_coords, main_pitch, upload_path)
            logger.info(f"聚类完成，有效音高: {len(valid_pitches)}，风格: {style_config['style']}")
            
            logger.info("开始生成音乐")
            audio_path = generate_music(valid_pitches, style_config, main_pitch)
            logger.info(f"音乐生成完成: {audio_path}")

            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                logger.error(f"音频文件不存在: {audio_path}")
                return jsonify({"code": 500, "msg": "音频文件生成失败"}), 500

            audio_filename = os.path.basename(audio_path)
            logger.info(f"返回响应，音频文件: {audio_filename}")

            response = jsonify({
                "code": 200,
                "msg": "音乐生成成功",
                "data": {
                    "midiUrl": f"/download/{audio_filename}",
                    "style": style_config['style'],
                    "tempo": style_config['tempo']
                }
            })
            return add_cors_headers(response), 200
            
        except Exception as e:
            logger.error(f"音乐生成过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"code": 500, "msg": f"音乐生成失败：{str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"上传处理过程中出错: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"code": 500, "msg": f"服务器处理失败：{str(e)}"}), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "服务器运行正常"}), 200

@app.route('/routes')
def list_routes():
    """显示所有已注册的路由"""
    routes = []
    for rule in app.url_map.iter_rules():
        methods = ', '.join(sorted(rule.methods - {'OPTIONS', 'HEAD'}))
        routes.append({
            'endpoint': rule.endpoint,
            'methods': methods,
            'path': rule.rule
        })
    
    return jsonify({"routes": routes})
    
@app.route('/debug', methods=['GET'])
def debug_info():
    """提供系统调试信息"""
    info = {
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "upload_folder_exists": os.path.exists(app.config['UPLOAD_FOLDER']),
        "upload_folder": app.config['UPLOAD_FOLDER'],
        "audio_folder_exists": os.path.exists(app.config['AUDIO_OUTPUT_FOLDER']),
        "audio_folder": app.config['AUDIO_OUTPUT_FOLDER'],
        "model_exists": os.path.exists(app.config['MODEL_PATH']),
        "model_path": app.config['MODEL_PATH'],
        "torch_available": torch is not None,
        "cv2_available": cv2 is not None,
        "pretty_midi_available": pretty_midi is not None
    }
    return jsonify(info)
    
@app.route('/download/<filename>', methods=['GET', 'OPTIONS'])
def download_midi(filename):
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        return add_cors_headers(response)
    
    try:
        safe_filename = secure_filename(filename)
        audio_path = os.path.join(app.config['AUDIO_OUTPUT_FOLDER'], safe_filename)
        
        logger.info(f"请求下载文件: {filename}, 安全文件名: {safe_filename}, 路径: {audio_path}")
        
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return jsonify({"code": 404, "msg": "音频文件已过期或不存在"}), 404
        
        logger.info(f"找到音频文件，准备发送: {audio_path}")
        
        # 添加CORS头
        response = send_file(
            audio_path,
            as_attachment=False,
            download_name=f"非遗音乐_{uuid.uuid4()}.wav",
            mimetype="audio/wav"
        )
        return add_cors_headers(response)
    except Exception as e:
        logger.error(f"下载文件出错: {str(e)}")
        return jsonify({"code": 500, "msg": f"文件下载失败：{str(e)}"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"未处理的异常: {str(e)}")
    return jsonify({"code": 500, "msg": "服务器内部错误"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"code": 404, "msg": "资源不存在"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"code": 405, "msg": "方法不允许"}), 405

if __name__ == '__main__':
    logger.info("已注册的路由:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {rule.rule} -> {rule.endpoint} ({', '.join(rule.methods)})")
    
    try:
        port = 5566  # 修改为5566端口以匹配前端期望
        logger.info(f"启动服务器在端口 {port}...")
        logger.info(f"服务器将在 http://0.0.0.0:{port} 上运行")
        logger.info(f"本地访问地址: http://localhost:{port}")
        
        # 使用 Flask 内置服务器
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise