import os   

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
 
import glob
import pretty_midi
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import random
import math
from scipy.fftpack import fft
from scipy.cluster.vq import kmeans2
from scipy.signal import find_peaks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

class EnhancedMusicGenerationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3,
                 dropout=0.3, style_embedding_size=16):
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
        self.attention = nn.MultiheadAttention(hidden_size, 4, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, style_indices, hidden=None):
        try:
            if x.dim() != 3:
                raise ValueError(f"x维度错误: 期望3维，实际{x.dim()}维，形状{x.shape}")
            batch_size, seq_len, input_size = x.size()
        except Exception as e:
            logger.error(f"解析x维度失败: {str(e)}")
            raise
        try:
            if style_indices.dim() != 1 or style_indices.size(0) != batch_size:
                raise ValueError(f"style_indices形状错误: 期望({batch_size},)，实际{style_indices.shape}")
            style_emb = self.style_embedding(style_indices)  
            style_emb = style_emb.unsqueeze(1).repeat(1, seq_len, 1)
        except Exception as e:
            logger.error(f"风格嵌入处理失败: {str(e)}")
            raise
        try:
            x = torch.cat([x, style_emb], dim=2)  
        except Exception as e:
            logger.error(f"拼接x和style_emb失败: x={x.shape}, style_emb={style_emb.shape}")
            raise
        try:
            if hidden is None:
                hidden = self.init_hidden(batch_size)
                hidden = (hidden[0].to(x.device), hidden[1].to(x.device))  
            
            if hidden[0].shape != (self.num_layers, batch_size, self.hidden_size):
                raise ValueError(f"hidden[0]形状错误: 期望({self.num_layers},{batch_size},{self.hidden_size})，实际{hidden[0].shape}")
            out, hidden = self.lstm(x, hidden) 
        except Exception as e:
            logger.error(f"LSTM前向传播失败: {str(e)}")
            raise
        out = self.dropout(out)
        try:
            attn_out, _ = self.attention(out, out, out)
            out = out + attn_out
            out = self.layer_norm(out)
        except Exception as e:
            logger.error(f"注意力机制处理失败: {str(e)}")
            raise
        out = self.fc(out)  
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
            weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        )
        return hidden

class MusicDataset(Dataset):
    def __init__(self, sequences, note_to_idx, style_labels=None):
        self.note_to_idx = note_to_idx
        self.sequences = sequences
        
        if style_labels is None:
            self.style_labels = torch.zeros(len(sequences), dtype=torch.long)
        else:
            if len(style_labels) != len(sequences):
                raise ValueError(f"style_labels与sequences长度不匹配: {len(style_labels)} vs {len(sequences)}")
            self.style_labels = torch.tensor(style_labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引超出范围: {idx} (总长度{len(self)})")
        
        sequence = self.sequences[idx]
        one_hot_seq = []
        for note in sequence:
            if note not in self.note_to_idx:
                note = self._get_fallback_note()
            one_hot = np.zeros(len(self.note_to_idx))
            one_hot[self.note_to_idx[note]] = 1
            one_hot_seq.append(one_hot)
        
        x = np.array(one_hot_seq[:-1])
        y = np.array([self.note_to_idx[note] for note in sequence[1:]])
        
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
                self.style_labels[idx])
    
    def _get_fallback_note(self):
        return max(self.note_to_idx.keys(), key=lambda k: self.note_to_idx[k])

def prepare_music_dataset(data_dir, sequence_length=32):
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")
    
    midi_files = glob.glob(os.path.join(data_dir, "**/*.mid"), recursive=True)
    midi_files += glob.glob(os.path.join(data_dir, "**/*.midi"), recursive=True)
    logger.info(f"找到 {len(midi_files)} 个MIDI文件")
    if not midi_files:
        raise ValueError(f"未找到任何MIDI文件: {data_dir}")
    all_sequences = []
    note_counter = Counter()
    
    for midi_file in midi_files:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        notes.append({
                            'pitch': note.pitch,
                            'start': note.start,
                            'end': note.end,
                            'duration': note.end - note.start,
                            'velocity': note.velocity
                        })
            
            notes.sort(key=lambda x: x['start'])
            note_sequence = [note['pitch'] for note in notes]
            if not note_sequence:
                logger.warning(f"MIDI文件无音符: {midi_file}")
                continue
            
            min_count = 5
            pitch_count = Counter(note_sequence)
            note_sequence = [pitch for pitch in note_sequence if pitch_count[pitch] >= min_count]
            if not note_sequence:
                logger.warning(f"MIDI文件过滤后无有效音符: {midi_file}")
                continue
            
            note_counter.update(note_sequence)
            
            for i in range(0, len(note_sequence) - sequence_length + 1, max(1, sequence_length // 2)):
                sequence = note_sequence[i:i+sequence_length]
                if len(sequence) == sequence_length:
                    all_sequences.append(sequence)
                    
        except Exception as e:
            logger.error(f"处理文件 {midi_file} 时出错: {e}")
            continue
    
    logger.info(f"总共生成了 {len(all_sequences)} 个训练序列")
    if not all_sequences:
        raise ValueError("未生成任何训练序列，请检查MIDI文件内容")
    
    if not note_counter:
        raise ValueError("未统计到任何音符，请检查MIDI文件解析逻辑")
    logger.info(f"音符范围: {min(note_counter.keys())} - {max(note_counter.keys())}")
    logger.info(f"最常见的10个音符: {note_counter.most_common(10)}")
    
    return all_sequences, note_counter

def preprocess_and_augment_sequences(sequences, note_counter, sequence_length=32):
    min_count = 5
    valid_notes = {note for note, count in note_counter.items() if count >= min_count}
    if len(valid_notes) < 30:
        logger.warning(f"有效音符不足30个（当前{len(valid_notes)}个），降低筛选条件")
        new_min_count = min_count - 1
        while len(valid_notes) < 30 and new_min_count >= 0:
            valid_notes = {note for note, count in note_counter.items() if count >= new_min_count}
            new_min_count -= 1
        if len(valid_notes) < 30:
            raise ValueError("有效音符数量过少，需补充数据集")
    note_to_idx = {note: idx for idx, note in enumerate(sorted(valid_notes))}
    idx_to_note = {idx: note for note, idx in note_to_idx.items()}
    logger.info(f"有效音符数量: {len(valid_notes)}（已确保≥30个）")
    filtered_sequences = []
    for seq in sequences:
        filtered_seq = [note for note in seq if note in valid_notes]
        if not filtered_seq:
            continue 
        
        if len(filtered_seq) < sequence_length:
            padding_note = filtered_seq[-1] if filtered_seq else next(iter(valid_notes))
            filtered_seq += [padding_note] * (sequence_length - len(filtered_seq))
        else:
            filtered_seq = filtered_seq[:sequence_length]  
        filtered_sequences.append(filtered_seq)
    if not filtered_sequences:
        raise ValueError("过滤后无有效序列，检查数据集或减小sequence_length")
    logger.info(f"过滤后序列数量: {len(filtered_sequences)}")
    augmented_sequences = []
    for seq in filtered_sequences:
        augmented_sequences.append(seq)
        
        for shift in range(1, 4):
            shifted_up = []
            for note in seq:
                new_note = note + shift
                if new_note > 127 or new_note not in note_to_idx:
                    new_note = note
                shifted_up.append(new_note)
            
            shifted_down = []
            for note in seq:
                new_note = note - shift
                if new_note < 0 or new_note not in note_to_idx:
                    new_note = note
                shifted_down.append(new_note)
                
            augmented_sequences.extend([shifted_up, shifted_down])
    
    logger.info(f"增强后序列数量: {len(augmented_sequences)}")
    return augmented_sequences, note_to_idx, idx_to_note

def train_music_model(data_dir, epochs=100, batch_size=64, learning_rate=0.001, 
                     output_path='music_generation_model_final.pth', sequence_length=32):
    try:
        logger.info("===== 开始准备训练数据 =====")
        sequences, note_counter = prepare_music_dataset(data_dir, sequence_length)
        augmented_sequences, note_to_idx, idx_to_note = preprocess_and_augment_sequences(
            sequences, note_counter, sequence_length)
        if not augmented_sequences:
            raise ValueError("训练数据为空，检查预处理逻辑")
        input_size = len(note_to_idx)
        output_size = len(note_to_idx)
        logger.info(f"模型参数: input_size={input_size}, output_size={output_size}, sequence_length={sequence_length-1}")
        model = EnhancedMusicGenerationModel(
            input_size=input_size,
            hidden_size=512,
            output_size=output_size,
            num_layers=3
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        style_labels = np.random.randint(0, 5, len(augmented_sequences))
        logger.info(f"风格标签形状: {style_labels.shape}")
        try:
            dataset = MusicDataset(augmented_sequences, note_to_idx, style_labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)  
            logger.info(f"数据加载器: 批次大小={batch_size}, 总批次={len(dataloader)}")
            if len(dataloader) == 0:
                raise ValueError(f"批次数量为0，可能batch_size({batch_size})大于样本数({len(dataset)})")
        except Exception as e:
            logger.error(f"构造数据集失败: {e}")
            raise
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"使用设备: {device}")
        logger.info("===== 开始训练模型 =====")
        train_losses = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_idx, (x_batch, y_batch, style_batch) in enumerate(dataloader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                style_batch = style_batch.to(device)
                expected_shape = (batch_size, sequence_length-1, input_size)
                if x_batch.shape != expected_shape:
                    raise ValueError(f"x_batch形状错误: 期望{expected_shape}，实际{x_batch.shape}")
                optimizer.zero_grad()
                try:
                    output, _ = model(x_batch, style_batch)  
                except Exception as e:
                    logger.error(f"批次{batch_idx}前向传播失败: {e}")
                    raise
                loss = criterion(output.transpose(1, 2), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
            avg_loss = total_loss / len(dataloader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)  
            logger.info(f"Epoch {epoch+1}/{epochs} 完成，平均损失: {avg_loss:.4f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'note_to_idx': note_to_idx,
                        'idx_to_note': idx_to_note,
                        'sequence_length': sequence_length
                    }, 'best_music_model.pth')
                    logger.info(f"最佳模型已更新并保存 (损失: {best_loss:.4f})")
                except Exception as e:
                    logger.error(f"保存最佳模型失败: {e}")
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'note_to_idx': note_to_idx,
                'idx_to_note': idx_to_note,
                'sequence_length': sequence_length
            }, output_path)
            logger.info(f"===== 训练完成! 最终模型已保存到: {output_path} =====")
        except Exception as e:
            logger.error(f"保存最终模型失败: {e}")
            raise
        return model, note_to_idx, idx_to_note
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)  
        raise

def generate_music_with_ai(model, seed_sequence, note_to_idx, idx_to_note,
                          style_index=0, length=100, temperature=1.0, 
                          sequence_length=32, device='cpu', style_name='平和'):
    try:
        model.eval()
        model.to(device)
        
        # 确保种子序列有效
        valid_seed = [note for note in seed_sequence if note in note_to_idx]
        if len(valid_seed) < 1:
            logger.warning("种子序列无效，使用随机音符初始化")
            valid_seed = [next(iter(note_to_idx.keys()))]
        
        # 填充或截断种子序列
        if len(valid_seed) < sequence_length - 1:
            last_note = valid_seed[-1]
            valid_seed += [last_note] * ((sequence_length - 1) - len(valid_seed))
        else:
            valid_seed = valid_seed[-(sequence_length - 1):]
        
        current_sequence = valid_seed.copy()
        generated_sequence = []
        
        with torch.no_grad():
            hidden = None
            style_tensor = torch.tensor([style_index], dtype=torch.long).to(device)
            
            for i in range(length):
                input_seq = []
                for note in current_sequence:
                    if note not in note_to_idx:
                        note = max(note_to_idx.keys(), key=lambda k: note_to_idx[k])
                    one_hot = np.zeros(len(note_to_idx))
                    one_hot[note_to_idx[note]] = 1
                    input_seq.append(one_hot)
                
                # 转换为模型输入格式
                input_tensor = torch.tensor([input_seq], dtype=torch.float32).to(device)
                
                # 使用模型生成下一个音符
                output, hidden = model(input_tensor, style_tensor, hidden)
                
                # 应用温度参数
                output = output / temperature
                probs = torch.softmax(output[0, -1, :], dim=-1)
                
                # 从概率分布中采样下一个音符
                next_idx = torch.multinomial(probs, 1).item()
                next_note = idx_to_note.get(next_idx, current_sequence[-1])
                
                # 更新当前序列
                current_sequence.append(next_note)
                current_sequence = current_sequence[1:]
                
                # 添加到生成序列
                generated_sequence.append(next_note)
        
        return generated_sequence
    except Exception as e:
        logger.error(f"AI生成音乐失败: {str(e)}")
        # 返回简单的回退序列
        return [60, 62, 64, 65, 67, 69, 71] * (length // 7 + 1)[:length]

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

def load_trained_model(model_path, device='cpu'):
    """加载训练好的模型"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        note_to_idx = checkpoint['note_to_idx']
        idx_to_note = checkpoint['idx_to_note']
        sequence_length = checkpoint.get('sequence_length', 32)
        
        model = EnhancedMusicGenerationModel(
            input_size=len(note_to_idx),
            hidden_size=512,
            output_size=len(note_to_idx),
            num_layers=3
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"模型加载成功: {model_path}")
        return model, note_to_idx, idx_to_note, sequence_length
        
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise

def generate_music_from_seed(seed_notes, style_index=0, length=100, 
                           temperature=0.8, model_path='music_generation_model_final.pth',
                           style_name='平和'):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, note_to_idx, idx_to_note, sequence_length = load_trained_model(model_path, device)
        
        # 确保种子序列有效
        valid_seed = []
        for note in seed_notes:
            if note in note_to_idx:
                valid_seed.append(note)
            else:
                # 找到最接近的有效音符
                closest_note = min(note_to_idx.keys(), key=lambda x: abs(x - note))
                valid_seed.append(closest_note)
                logger.warning(f"音符 {note} 不在词汇表中，使用最接近的音符 {closest_note}")
        
        if not valid_seed:
            logger.warning("种子序列无效，使用默认音符")
            valid_seed = [60]  # 默认使用C4
        
        # 生成音乐
        generated_notes = generate_music_with_ai(
            model=model,
            seed_sequence=valid_seed,
            note_to_idx=note_to_idx,
            idx_to_note=idx_to_note,
            style_index=style_index,
            length=length,
            temperature=temperature,
            sequence_length=sequence_length,
            device=device,
            style_name=style_name
        )
        
        return generated_notes
        
    except Exception as e:
        logger.error(f"生成音乐失败: {str(e)}")
        # 返回简单的回退序列
        return [60, 62, 64, 65, 67, 69, 71] * (length // 7 + 1)[:length]


def demo_generation():
    """演示如何使用模型生成音乐，包含和声"""
    try:
        seed_notes = [60, 62, 64, 65, 67]  # C大调音阶的一部分
        style_index = 0  # 欢快风格
        style_name = '欢快'
        
        print("开始生成音乐...")
        music_notes = generate_music_from_seed(seed_notes, style_index, length=50, style_name=style_name)
        print(f"生成完成! 共生成 {len(music_notes)} 个音符")
        print(f"前20个音符: {music_notes[:20]}")
        
        # 生成和声
        main_pitch = 60  # C4
        harmony = generate_harmony_for_ai(main_pitch, style_name)
        print(f"生成的和声进行: {harmony}")
        
        # 保存为MIDI文件（包含和声）
        midi_data = pretty_midi.PrettyMIDI()
        
        # 主旋律乐器
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        # 和声乐器
        strings_program = pretty_midi.instrument_name_to_program('String Ensemble 1')
        strings = pretty_midi.Instrument(program=strings_program)
        
        # 添加主旋律
        start_time = 0.0
        for note in music_notes:
            midi_note = pretty_midi.Note(
                velocity=100,
                pitch=note,
                start=start_time,
                end=start_time + 0.5  # 每个音符持续0.5秒
            )
            piano.notes.append(midi_note)
            start_time += 0.5
        
        # 添加和声
        harmony_start_time = 0.0
        for chord in harmony:
            for note in chord:
                harmony_note = pretty_midi.Note(
                    velocity=80,  # 和声力度稍小
                    pitch=note - 12,  # 降低一个八度，避免与主旋律冲突
                    start=harmony_start_time,
                    end=harmony_start_time + 2.0  # 每个和弦持续2秒
                )
                strings.notes.append(harmony_note)
            harmony_start_time += 2.0
        
        midi_data.instruments.append(piano)
        midi_data.instruments.append(strings)
        
        # 使用定义的输出目录
        output_path = os.path.join(OUTPUT_DIR, 'demo_music_with_harmony.mid')
        midi_data.write(output_path)
        print(f"带和声的演示音乐已保存为 '{output_path}'")
        
        return music_notes, harmony
        
    except Exception as e:
        print(f"演示失败: {e}")
        return [], []
    
if __name__ == "__main__":
    # 1. 训练模型
    train_music_model("C:\\Users\\cheny_nlk3f4d\\Desktop\\sonic heritage\\ai\\MIDI_Dataset", epochs=50)
    
    # 2. 生成音乐和和声
    demo_notes, demo_harmony = demo_generation()
    
    # 3. 保存为MIDI文件（包含和声）
    if demo_notes:
        midi_data = pretty_midi.PrettyMIDI()
        
        # 主旋律乐器
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        # 和声乐器
        strings_program = pretty_midi.instrument_name_to_program('String Ensemble 1')
        strings = pretty_midi.Instrument(program=strings_program)
        
        # 添加主旋律
        start_time = 0.0
        for note in demo_notes:
            midi_note = pretty_midi.Note(
                velocity=100,
                pitch=note,
                start=start_time,
                end=start_time + 0.5  # 每个音符持续0.5秒
            )
            piano.notes.append(midi_note)
            start_time += 0.5
        
        # 添加和声
        harmony_start_time = 0.0
        for chord in demo_harmony:
            for note in chord:
                harmony_note = pretty_midi.Note(
                    velocity=80,  # 和声力度稍小
                    pitch=note - 12,  # 降低一个八度，避免与主旋律冲突
                    start=harmony_start_time,
                    end=harmony_start_time + 2.0  # 每个和弦持续2秒
                )
                strings.notes.append(harmony_note)
            harmony_start_time += 2.0
        
        midi_data.instruments.append(piano)
        midi_data.instruments.append(strings)
        midi_data.write('demo_music_with_harmony.mid')
        print("带和声的演示音乐已保存为 'demo_music_with_harmony.mid'")