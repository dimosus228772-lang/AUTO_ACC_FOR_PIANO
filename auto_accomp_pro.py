#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Accompanist PRO (Ubuntu/ALSA, real-time)

Функционал:
- 15с анализ: тональность (маж/мин), темп (BPM), размер, стиль
- Реактивный аккомпанемент: voicing-инверсии, «развязка» (climax), консонанс-контроль
- Умная темподетекция: фильтр кратностей + адаптация «на лету» (эксп. сглаживание)
- Продвинутая гармония: 7/9/11/13 и модальные заимствования (джаз/поп)
- Нейро-генератор (опционально): LSTM/Transformer по аккордам/стилю (загрузка .pt)
- Смена стиля по «секции» (интро/куплет/припев/бридж) через фразировку/энергию
- Гладкая смена тональности: скользящая переоценка хром с гистерезисом
- HUD в терминале (curses): BPM/ключ/аккорд/стиль/секция/консонанс (опционально)
- Безопасное завершение: All Notes Off

Зависимости: mido, python-rtmidi, numpy, (опц.) torch, (опц.) curses (встроен в Linux)
"""

import argparse
import time
import math
import threading
from collections import deque, defaultdict, Counter
import numpy as np
import mido
from mido import Message

# -------- Опциональная нейросеть --------
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# -------- Опциональный HUD (curses) -----
try:
    import curses
    CURSES_OK = True
except Exception:
    CURSES_OK = False

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def pc(n): return n % 12
def midi_to_name(n): return NOTE_NAMES[pc(n)] + str(n//12 - 1)
def key_to_str(root_pc, is_minor): return f"{NOTE_NAMES[root_pc]} {'minor' if is_minor else 'major'}"

def softmax_np(x):
    x = np.array(x, dtype=float); x -= x.max()
    e = np.exp(x); return e/(e.sum()+1e-9)

# ---- Базовые паттерны ----
PATTERNS = {
    'pop':   [('note',0.0,'bass',0.45), ('note',0.0,'chord',0.48), ('note',0.5,'chord',0.45)],
    'ballad':[('note',0.0,'bass',0.95), ('arp',0.25,'arp',0.2), ('arp',0.5,'arp',0.2), ('arp',0.75,'arp',0.2)],
    'rock':  [('note',0.0,'bass',0.35), ('note',0.5,'bass',0.35), ('note',0.25,'chord',0.2), ('note',0.75,'chord',0.2)],
    'jazz':  [('note',0.0,'bass',0.3), ('note',0.33,'chord',0.2), ('note',0.66,'chord',0.2), ('note',0.9,'chord',0.1)],
    'waltz': [('note',0.0,'bass',0.3), ('note',1/3,'chord',0.25), ('note',2/3,'chord',0.25)],
}
PATTERN_VARIANTS = {
    'pop':   [PATTERNS['pop'], PATTERNS['pop']+[('arp',0.75,'arp',0.2)],
              [('note',0.0,'bass',0.3),('note',0.25,'chord',0.2),('note',0.5,'chord',0.2),('note',0.75,'chord',0.2)]],
    'ballad':[PATTERNS['ballad'], PATTERNS['ballad']+[('note',0.0,'chord',0.1)]],
    'rock':  [PATTERNS['rock'], [('note',0.0,'bass',0.25),('note',0.5,'bass',0.25),
                                 ('note',0.25,'chord',0.15),('note',0.5,'chord',0.15),('note',0.75,'chord',0.15)]],
    'jazz':  [PATTERNS['jazz'], [('note',0.0,'bass',0.25),('arp',0.25,'arp',0.2),('note',0.5,'chord',0.2),('arp',0.75,'arp',0.15)]],
    'waltz': [PATTERNS['waltz'], PATTERNS['waltz']+[('arp',2/3,'arp',0.2)]],
}

# Расширенные аккорд-формулы
CHORD_INTERVALS = {
    'maj':  [0,4,7],
    'min':  [0,3,7],
    'dim':  [0,3,6],
    'aug':  [0,4,8],
    '7':    [0,4,7,10],
    'm7':   [0,3,7,10],
    'maj7': [0,4,7,11],
    '9':    [0,4,7,10,14],     # дом. 9
    'm9':   [0,3,7,10,14],
    '11':   [0,4,7,10,17],     # дом. 11
    '13':   [0,4,7,10,21],     # дом. 13
    'm11':  [0,3,7,10,17],
    'm13':  [0,3,7,10,21],
}

DEFAULT_CHANNEL_MELODY = 0
DEFAULT_CHANNEL_ACC    = 1
VEL_CHORD=72; VEL_BASS=78; VEL_ARP=64

# --------- Небольшая нейросеть-генератор (опционально) ----------
class TinyNoteRNN(nn.Module):
    """
    Игрушечная LSTM, ожидает входные фичи (такт-фаза, корень/качество аккорда, стиль, энергия)
    и выдаёт распределение по смещениям нот относительно корня.
    Это заглушка: для серьёзности загрузите внешнюю обученную модель (--model_path).
    """
    def __init__(self, input_dim=16, hidden=64, out_dim=24):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x, h=None):
        y, h = self.lstm(x, h)
        logits = self.head(y[:, -1, :])
        return logits, h

# ---------------- Core Class ----------------
class AutoAccompPro:
    def __init__(self, in_port=None, out_port=None, analyze_secs=15.0,
                 acc_channel=DEFAULT_CHANNEL_ACC, melody_channel=DEFAULT_CHANNEL_MELODY,
                 program=None, hud=False, model_path=None, verbose=True):
        self.verbose=verbose
        self.analyze_secs=analyze_secs
        self.acc_ch=acc_channel; self.mel_ch=melody_channel
        self.program=program
        self.hud_enabled = hud and CURSES_OK

        # MIDI
        self.inp = mido.open_input(in_port) if in_port else mido.open_input()
        self.outp = mido.open_output(out_port) if out_port else mido.open_output()

        # Сбор данных
        self.events=[]
        self.ioi=[]              # межатаковые интервалы
        self.active_notes=set()
        self.note_on_times={}
        self.consonance_window=deque(maxlen=64)

        # Итоги анализа
        self.root_pc=0; self.is_minor=False
        self.bpm=90.0; self.time_sig=(4,4); self.style='pop'

        # Темпо-трекинг онлайн
        self._bpm_smooth = None
        self._tempo_alpha = 0.06  # мягкая подстройка

        # Состояние такта
        self.seconds_per_bar=4.0
        self.bar_pos=0.0
        self.bar_start=time.time()

        # Гармония
        self.current_chord_root=0
        self.current_chord_qual='maj'
        self.chord_smoothing=deque(maxlen=8)

        # Секция аранжировки (Intro/Verse/Chorus/Bridge)
        self.section='Intro'
        self.energy_hist=deque(maxlen=16)  # средняя высота и плотность
        self.section_timer=time.time()

        # Нейро-политика выбора варианта и нейро-генератор
        self.policy = None
        if TORCH_OK:
            self.policy = torch.nn.Sequential(
                torch.nn.Linear(8,32), torch.nn.ReLU(), torch.nn.Linear(32,3)
            )
        self.rnn = None
        self.rnn_h = None
        if TORCH_OK and model_path:
            try:
                self.rnn = torch.load(model_path, map_location='cpu')
            except Exception:
                self.rnn = TinyNoteRNN()
        elif TORCH_OK:
            self.rnn = TinyNoteRNN()

        # Key tracking (скользящая переоценка)
        self.key_hist=deque(maxlen=32)     # последние оценки (root,is_minor)
        self.key_stability=0.0             # для гистерезиса

        # HUD
        self.screen=None

        self.say("Нейросеть запущена, можете играть.")
        if self.program is not None:
            self.program_change(self.program)

    # --------- Util ----------
    def say(self, txt):
        if self.verbose: print(f"[INFO] {txt}")

    def program_change(self, program=0, ch=None):
        ch = self.acc_ch if ch is None else ch
        try:
            self.outp.send(Message('program_change', program=program, channel=ch))
            self.say(f"Установлен timbre (program) {program} на канале {ch}.")
        except Exception as e:
            self.say(f"Program Change fail: {e}")

    def note_on(self, note, vel, ch=None):
        ch = self.acc_ch if ch is None else ch
        self.outp.send(Message('note_on', note=note, velocity=vel, channel=ch))

    def note_off(self, note, ch=None):
        ch = self.acc_ch if ch is None else ch
        self.outp.send(Message('note_off', note=note, velocity=0, channel=ch))

    def all_notes_off(self):
        for n in range(128):
            self.note_off(n)

    # --------- Calibration (15s) ----------
    def calibrate(self):
        self.say("Идёт 15-секундный анализ: тональность, темп, размер, стиль...")
        t0=time.time(); last_on=None
        while time.time()-t0 < self.analyze_secs:
            for msg in self.inp.iter_pending():
                now=time.time()
                if msg.type=='note_on' and msg.velocity>0:
                    self.events.append((now,'on',msg.note,msg.velocity))
                    self.active_notes.add(msg.note); self.note_on_times[msg.note]=now
                    if last_on is not None: self.ioi.append(now-last_on)
                    last_on=now
                elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
                    self.events.append((now,'off',msg.note,0))
                    self.active_notes.discard(msg.note); self.note_on_times.pop(msg.note,None)
            time.sleep(0.001)

        self.root_pc, self.is_minor = self.estimate_key_from_events(self.events)
        self.bpm = self.estimate_bpm_from_ioi(self.ioi)
        self.time_sig = self.estimate_time_signature(self.ioi)
        self.style = self.estimate_style()

        beats_per_bar = self.time_sig[0]
        self.seconds_per_bar = 60.0/self.bpm * beats_per_bar
        self._bpm_smooth = self.bpm

        self.say(f"Определена тональность: {key_to_str(self.root_pc, self.is_minor)}.")
        self.say(f"Определён темп: {round(self.bpm)} BPM, размер: {self.time_sig[0]}/{self.time_sig[1]}.")
        self.say(f"Определён жанр/стиль: {self.style}.")

    # --------- Key estimation (batch) ----------
    def estimate_key_from_events(self, events):
        end_time = events[-1][0] if events else time.time()
        on_dict=defaultdict(list)
        durations=defaultdict(float); strengths=defaultdict(float)

        for t,typ,n,v in events:
            if typ=='on': on_dict[n].append((t,v))
            else:
                if on_dict[n]:
                    t_on, vv = on_dict[n].pop(0)
                    durations[n]+=max(0.01, t-t_on)
                    strengths[n]+=vv
        # hanging
        for n,arr in on_dict.items():
            for t_on, vv in arr:
                durations[n]+=max(0.01, end_time - t_on)
                strengths[n]+=vv

        chroma=np.zeros(12)
        total=max(1.0, sum(durations.values()))
        for n, dur in durations.items():
            weight = dur*(strengths[n]/(len(events)+1e-6))
            chroma[pc(n)] += weight

        if chroma.sum()<1e-6: chroma[0]=1.0

        def score_profile(profile):
            scores=[]
            for r in range(12):
                prof=np.roll(profile,r)
                c=np.corrcoef(chroma,prof)[0,1]
                if np.isnan(c): c=0.0
                scores.append(c)
            r=int(np.argmax(scores)); return r, float(np.max(scores))

        maj_root, maj_s = score_profile(MAJOR_PROFILE)
        min_root, min_s = score_profile(MINOR_PROFILE)
        return (min_root, True) if min_s>maj_s else (maj_root, False)

    # --------- BPM estimation with multiplicity filter ----------
    def estimate_bpm_from_ioi(self, ioi):
        if not ioi: return 90.0
        arr=np.array(ioi)
        arr=arr[(arr>0.06)&(arr<2.5)]
        if len(arr)==0: return 90.0

        cands=[]
        for x in arr:
            # породим семейство кратностей / субкратностей
            for k in [0.5, 2/3, 1.0, 3/2, 2.0, 3.0, 4.0]:
                bpm = 60.0/x * k
                # приведём в диапазон 40..240 с учетом октавной эквивалентности
                while bpm<40: bpm*=2
                while bpm>240: bpm/=2
                cands.append(bpm)
        # кластеризация через гистограмму
        bins=np.linspace(40,240,101)
        hist,edges=np.histogram(cands,bins=bins)
        peak=np.argmax(hist)
        bpm_est=(edges[peak]+edges[peak+1])/2
        # медианный якорь
        bpm_med=np.median(cands)
        bpm=(0.6*bpm_est+0.4*bpm_med)
        return float(bpm)

    def estimate_time_signature(self, ioi):
        if len(ioi)<6: return (4,4)
        arr=np.array(ioi); med=np.median(arr)
        # проверим сильную трёхдольность (~1.5*med шаги)
        r = np.mean((np.abs(arr - med/1.5) < 0.05) | (np.abs(arr - med*1.5) < 0.05))
        return (3,4) if r>0.25 else (4,4)

    def estimate_style(self):
        on=[(t,n,v) for (t,typ,n,v) in self.events if typ=='on']
        if not on: return 'pop'
        dt = max(1e-3, on[-1][0]-on[0][0])
        density=len(on)/dt
        vel=np.array([v for _,_,v in on]); vel_var=np.var(vel)
        beat=60.0/max(40.0,min(240.0,self.bpm))
        off=sum(1 for t,_,_ in on if 0.2<(((t-on[0][0])%beat)/beat)<0.8)
        off_ratio=off/max(1,len(on))

        if self.time_sig[0]==3: return 'waltz'
        if density>6.0 and self.bpm>=120: return 'rock'
        if off_ratio>0.45 and vel_var>300: return 'jazz'
        if self.bpm<80 and vel_var<150: return 'ballad'
        return 'pop'

    # --------- Online tempo adapt ----------
    def tempo_adapt_step(self, new_attack_time, last_attack_time):
        if last_attack_time is None: return
        dt = new_attack_time - last_attack_time
        if dt<0.06 or dt>2.5: return
        bpm_inst = 60.0/dt
        # приведём к ближайшей «октавной» кратности текущего BPM
        if self._bpm_smooth is None:
            self._bpm_smooth=bpm_inst
        base=self._bpm_smooth
        # нормализуем кандидата в окрестность базового
        while bpm_inst > base*1.9: bpm_inst/=2
        while bpm_inst < base/1.9: bpm_inst*=2
        # эксп. сглаживание
        self._bpm_smooth = (1-self._tempo_alpha)*self._bpm_smooth + self._tempo_alpha*bpm_inst
        # ограничим дрейф
        self._bpm_smooth = max(40.0, min(240.0, self._bpm_smooth))
        self.bpm = float(self._bpm_smooth)
        self.seconds_per_bar = 60.0/self.bpm * self.time_sig[0]

    # --------- Key adapt (sliding, hysteresis) ----------
    def key_adapt_step(self, recent_events):
        # каждые несколько секунд переоценивать хрому
        if not recent_events: return
        root,is_min = self.estimate_key_from_events(recent_events)
        self.key_hist.append((root,is_min))
        # голосование
        ms=Counter(self.key_hist).most_common(1)[0][0]
        # гистерезис: менять, если сохраняется > 60% последних оценок
        agree = sum(1 for x in self.key_hist if x==ms)/len(self.key_hist)
        if agree>0.6 and (ms!=(self.root_pc,self.is_minor)):
            self.root_pc, self.is_minor = ms
            self.say(f"Модуляция: новая тональность {key_to_str(self.root_pc, self.is_minor)}")

    # --------- Harmony detection with extensions & modal borrow ----------
    def detect_chord(self, active_set):
        if not active_set:
            return self.root_pc, ('min' if self.is_minor else 'maj')

        pcs=[pc(n) for n in active_set]
        counts=Counter(pcs)
        candidate_roots=[p for p,_ in counts.most_common(5)] or [self.root_pc]
        best=(self.root_pc,'min' if self.is_minor else 'maj'); best_score=-1e9

        # шкала тональности (для бонусов/заимствований)
        scale = self.major_scale(self.root_pc) if not self.is_minor else self.natural_minor_scale(self.root_pc)

        for root in candidate_roots:
            for qual, intervals in CHORD_INTERVALS.items():
                chord_set=set((root+i)%12 for i in intervals[:4])  # базис трезвучие/7
                hit=sum(counts.get(p,0) for p in chord_set)
                score=hit
                # расширения 9/11/13: смотрим, встречаются ли ступени (2/5/9 от корня)
                ext_points=0
                if (root+2)%12 in counts: ext_points+=0.4   # 9
                if (root+5)%12 in counts: ext_points+=0.4   # 11
                if (root+9)%12 in counts: ext_points+=0.4   # 13
                score+=ext_points
                # модальные заимствования: bIII, bVI, bVII для мажора — допускаем мин/маж трезвучия
                if not self.is_minor:
                    borrowed = [ (self.root_pc+3)%12, (self.root_pc+8)%12, (self.root_pc+10)%12 ]
                    if root in borrowed: score+=0.6
                # соответствие тональности
                in_key = sum(1 for p in chord_set if p in scale)/len(chord_set)
                score+=0.5*in_key
                if score>best_score:
                    best_score=score; best=(root, qual)
        return best

    def major_scale(self, r):
        return set([(r+i)%12 for i in [0,2,4,5,7,9,11]])
    def natural_minor_scale(self, r):
        return set([(r+i)%12 for i in [0,2,3,5,7,8,10]])

    # --------- Consonance & voicing ----------
    def consonance_score(self, melody_note, acc_notes):
        weights={0:+2.0,3:+0.8,4:+1.1,5:+1.5,7:+1.7,8:+0.9,9:+1.0,
                 1:-1.2,2:-1.0,6:-1.5,10:-0.9,11:-1.1}
        if not acc_notes: return 0.0
        s=0.0
        for an in acc_notes:
            d=abs((an-melody_note)%12); d=min(d,12-d)
            s+=weights.get(d,0.0)
        return s/len(acc_notes)

    def adjust_voicing(self, chord_notes, melody_note):
        best = chord_notes[:]
        best_score = self.consonance_score(melody_note, chord_notes)
        for i,n in enumerate(chord_notes):
            for shift in (-12,+12):
                cand = chord_notes[:i]+[n+shift]+chord_notes[i+1:]
                sc = self.consonance_score(melody_note, cand)
                if sc>best_score: best, best_score = cand, sc
        return best

    # --------- Section (Intro/Verse/Chorus/Bridge) ----------
    def update_section(self):
        # простая логика: растущая энергия → Chorus, снижение → Verse, скачок и редкие акценты → Bridge, старт → Intro
        energy = self.estimate_energy()
        self.energy_hist.append(energy)
        if len(self.energy_hist)<8: 
            self.section='Intro'; return
        avg = np.mean(self.energy_hist); recent=np.mean(list(self.energy_hist)[-4:])
        if recent > avg*1.15: self.section='Chorus'
        elif recent < avg*0.9: self.section='Verse'
        # редкая плотность + скачки высоты → Bridge
        if recent>avg*1.05 and self.bpm>110 and len(self.active_notes)<=2:
            self.section='Bridge'

    def estimate_energy(self):
        if not self.note_on_times: return 0.0
        recent = [n for n,_t in sorted(self.note_on_times.items(), key=lambda kv: kv[1], reverse=True)][:8]
        if not recent: return 0.0
        return np.mean(recent)/84.0 + len(self.active_notes)*0.05

    def is_climax(self):
        return self.section=='Chorus' or self.estimate_energy()>0.9

    # --------- Pattern / Neural generation ----------
    def pick_pattern(self, style, variant_idx):
        variants = PATTERN_VARIANTS.get(style,[PATTERNS['pop']])
        return variants[min(variant_idx, len(variants)-1)]

    def choose_variant(self):
        energy = 1.0 if self.is_climax() else 0.3
        conson = np.mean(self.consonance_window) if len(self.consonance_window)>=8 else 0.5
        feat=np.array([
            energy, conson, self.bpm/240.0,
            1.0 if self.time_sig[0]==3 else 0.0,
            1.0 if self.is_minor else 0.0,
            (self.current_chord_root%12)/11.0,
            1.0 if self.style in ('rock','jazz') else 0.0,
            1.0 if self.style in ('ballad','waltz') else 0.0,
        ],dtype=np.float32)
        if TORCH_OK and self.policy is not None:
            with torch.no_grad():
                logits = self.policy(torch.from_numpy(feat).unsqueeze(0)).numpy()[0]
            probs=softmax_np(logits)
            if energy>0.8 and len(probs)>=3:
                probs=probs*np.array([0.9,1.0,1.1])[:len(probs)]; probs/=probs.sum()
            return int(np.random.choice(len(probs), p=probs))
        return 2 if energy>0.8 else (1 if energy>0.5 else 0)

    def realize_role(self, role, root_pc, qual):
        base_oct_bass=36; base_oct_chord=52
        intervals = CHORD_INTERVALS.get(qual, CHORD_INTERVALS['maj'])
        if role=='bass':
            n=base_oct_bass+root_pc
            return [n, n+12] if self.is_climax() else [n]
        if role=='chord':
            notes=[base_oct_chord+root_pc+i for i in intervals[:3]]
            if self.is_climax() and len(intervals)>3: notes.append(base_oct_chord+root_pc+intervals[3])
            return notes
        if role in ('fill','arp'):
            seq=[base_oct_chord+root_pc+i for i in intervals[:3]]
            # если есть нейро-генератор — попробуем подсунуть ноту
            if TORCH_OK and self.rnn is not None:
                phase = (time.time()-self.bar_start)/self.seconds_per_bar
                x = self.build_rnn_features(phase, root_pc, qual)
                with torch.no_grad():
                    logits, self.rnn_h = self.rnn(x, self.rnn_h)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                idx = np.random.choice(len(probs), p=probs)
                # map idx->offset: [-12..+11] полутона
                offset = idx-12
                note = base_oct_chord + root_pc + (offset%12) + 12*((offset//12)+1)
                return [note]
            # fallback: простой арпеджио-одиночки
            return [seq[int(time.time()*1000)%len(seq)]]
        return []

    def build_rnn_features(self, phase, root_pc, qual):
        # формируем (batch=1, time=1, feat=16) с one-hot стиля, качества и фазой такта
        style_vec=np.zeros(5); styles=['pop','ballad','rock','jazz','waltz']
        style_vec[styles.index(self.style) if self.style in styles else 0]=1.0
        qual_vec=np.zeros(6); quals=['maj','min','7','m7','maj7','9']
        q = qual if qual in quals else 'maj'
        qual_vec[quals.index(q)]=1.0
        feat=np.array([phase, self.bpm/240.0, (root_pc%12)/11.0, 1.0 if self.is_minor else 0.0])
        vec=np.concatenate([feat, style_vec, qual_vec]).astype(np.float32)
        return torch.from_numpy(vec).view(1,1,-1)

    # --------- HUD ---------
    def hud_draw(self):
        if not self.hud_enabled or self.screen is None: return
        s=self.screen
        s.erase()
        s.addstr(0,0, "Auto Accompanist PRO (q=выход)")
        s.addstr(2,0, f"BPM: {self.bpm:6.1f}   Size: {self.time_sig[0]}/{self.time_sig[1]}   Style: {self.style}")
        s.addstr(3,0, f"Key: {key_to_str(self.root_pc,self.is_minor)}   Chord: {NOTE_NAMES[self.current_chord_root]} {self.current_chord_qual}")
        s.addstr(4,0, f"Section: {self.section}")
        cons = np.mean(self.consonance_window) if self.consonance_window else 0.0
        s.addstr(5,0, f"Consonance: {cons:5.2f}")
        s.refresh()

    # --------- Main run loop ----------
    def run(self):
        self.calibrate()
        self.say("Старт аккомпанемента.")
        if self.hud_enabled:
            self.screen = curses.initscr()
            curses.noecho(); curses.cbreak(); self.screen.nodelay(True)
        last_on=None
        bar_variant=0
        acc_active=set()
        recent_for_key=deque(maxlen=400)

        try:
            self.bar_start=time.time()
            while True:
                now=time.time()
                # input
                for msg in self.inp.iter_pending():
                    if msg.type=='note_on' and msg.velocity>0:
                        self.active_notes.add(msg.note); self.note_on_times[msg.note]=now
                        # tempo adapt
                        self.tempo_adapt_step(now, last_on)
                        last_on=now
                        # consonance probe
                        if acc_active: self.consonance_window.append(self.consonance_score(msg.note, list(acc_active)))
                        # для адаптации ключа — копим последние события
                        recent_for_key.append((now,'on',msg.note,msg.velocity))
                    elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
                        self.active_notes.discard(msg.note); self.note_on_times.pop(msg.note,None)
                        recent_for_key.append((now,'off',msg.note,0))

                # бар/позиция
                if now - self.bar_start > self.seconds_per_bar:
                    self.bar_start += self.seconds_per_bar
                    self.bar_pos=0.0
                    bar_variant=self.choose_variant()
                    self.update_section()
                self.bar_pos=(now-self.bar_start)/self.seconds_per_bar

                # ключ адаптируется мягко
                if int(now*2)%6==0:  # раз ~ каждые 0.5с
                    self.key_adapt_step(list(recent_for_key))

                # текущий аккорд
                ch_root, ch_qual = self.detect_chord(self.active_notes)
                self.chord_smoothing.append((ch_root,ch_qual))
                ch_root,ch_qual = Counter(self.chord_smoothing).most_common(1)[0][0]
                self.current_chord_root, self.current_chord_qual = ch_root, ch_qual

                # стиль корректируем по секции
                if self.section=='Chorus' and self.style in ('ballad','waltz'):
                    self.style='pop'
                if self.section=='Bridge' and self.style!='jazz':
                    self.style='rock' if self.bpm>110 else 'jazz'

                # события паттерна
                pattern=self.pick_pattern(self.style, bar_variant)
                for (typ,at,role,length) in pattern:
                    if abs(self.bar_pos - at) < (0.005 + 0.001*self.bpm/120):
                        notes=self.realize_role(role, ch_root, ch_qual)
                        if self.active_notes:
                            mel_top=max(self.active_notes)
                            notes=self.adjust_voicing(notes, mel_top)
                        for n in notes:
                            vel=VEL_CHORD
                            if role=='bass': vel=VEL_BASS
                            if typ=='arp': vel=VEL_ARP
                            self.note_on(n, vel)
                        off_t = now + length*self.seconds_per_bar
                        threading.Timer(max(0.01, off_t-time.time()), self.notes_off_batch, args=(notes,)).start()
                        for n in notes: acc_active.add(n)
                        threading.Timer(length*self.seconds_per_bar+0.03, lambda: [acc_active.discard(n) for n in notes]).start()

                # HUD
                self.hud_draw()
                # выход по 'q' в HUD
                if self.hud_enabled:
                    try:
                        ch=self.screen.getch()
                        if ch in (ord('q'), ord('Q')): break
                    except Exception:
                        pass

                time.sleep(0.001)
        finally:
            self.all_notes_off()
            if self.hud_enabled:
                curses.nocbreak(); self.screen.nodelay(False); curses.echo(); curses.endwin()
            self.say("Остановлено.")

    def notes_off_batch(self, notes):
        for n in notes: self.note_off(n)

# -------------- CLI --------------
def list_ports():
    print("Входные MIDI-порты:")
    for name in mido.get_input_names():
        print("  ", name)
    print("Выходные MIDI-порты:")
    for name in mido.get_output_names():
        print("  ", name)

def main():
    ap=argparse.ArgumentParser(description="Auto Accompanist PRO")
    ap.add_argument('--in', dest='in_port', default=None, help='Входной MIDI-порт')
    ap.add_argument('--out', dest='out_port', default=None, help='Выходной MIDI-порт')
    ap.add_argument('--program', type=int, default=None, help='MIDI Program для канала аккомпанемента')
    ap.add_argument('--analyze', type=float, default=15.0, help='Секунды калибровки (по умолчанию 15)')
    ap.add_argument('--acc_channel', type=int, default=DEFAULT_CHANNEL_ACC, help='Канал аккомпанемента (0..15)')
    ap.add_argument('--mel_channel', type=int, default=DEFAULT_CHANNEL_MELODY, help='Канал мелодии (0..15)')
    ap.add_argument('--hud', action='store_true', help='Включить HUD (curses)')
    ap.add_argument('--model_path', type=str, default=None, help='Путь к обученной нейромодели (.pt)')
    ap.add_argument('--list', action='store_true', help='Показать порты и выйти')
    args=ap.parse_args()

    if args.list: list_ports(); return

    acc=AutoAccompPro(
        in_port=args.in_port, out_port=args.out_port,
        analyze_secs=args.analyze, acc_channel=args.acc_channel,
        melody_channel=args.mel_channel, program=args.program,
        hud=args.hud, model_path=args.model_path, verbose=True
    )
    try:
        acc.run()
    except KeyboardInterrupt:
        pass

if __name__=='__main__':
    main()
