#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import tdt
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.io import wavfile as _wavfile
import pygame

from psychopy import visual, core, gui, data, logging, event


# =========================================================
# 1. EEG Trigger
# =========================================================
def send_trigger(code: int):
    syn.setParameterValue('TTL2Int1', 'IntegerValue', code)
    syn.setParameterValue('TTL2Int1', 'ManualTrigger', 1)
    core.wait(0.01)
    syn.setParameterValue('TTL2Int1', 'ManualTrigger', 0)


# =========================================================
# 2. Participant dialog
# =========================================================
exp_info = {
    'subject_number': 1,
    'experimenter_id': 'KISTv1',
    'day': 1,
}

dlg = gui.DlgFromDict(
    dictionary=exp_info,
    title='ASA Korean Experiment',
    order=['subject_number', 'experimenter_id', 'day'],
)

if not dlg.OK:
    core.quit()

subj_idx = int(exp_info['subject_number'])
exp_id = str(exp_info['experimenter_id']).strip().lower()
idx_day = int(exp_info['day'])
korean_id = f'KOREAN{subj_idx:03d}'

# =========================================================
# 2.5. TDT Synapse connection
# =========================================================
try:
    syn = tdt.SynapseAPI()
    print("TDT Synapse 연결 성공")
    print("연결 성공")
    print(f"현재 모드: {syn.getMode()}")
    print(f"현재 유저: {syn.getCurrentUser()}")
    print(f"현재 실험: {syn.getCurrentExperiment()}")
    print(f"현재 탱크: {syn.getCurrentTank()}")
    core.wait(1)  # Synapse 안정화 대기
except Exception as e:
    print(f"TDT 연결 실패: {e}")
    core.quit()

clean_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
block_name = f"SNU{subj_idx:03d}_day{idx_day}_{clean_datetime}"

try:
    # 이미 Idle이면 setMode(0) 스킵
    if syn.getMode() != 0:
        syn.setMode(0)
    syn.setCurrentUser("KIST_ducky")
    syn.setCurrentExperiment("LingcogERP")
    # syn.createSubject(korean_id, f'datetime_{clean_datetime}', 'mouse')  # 제거
    syn.setCurrentSubject(korean_id)
    syn.setCurrentBlock(block_name)
    syn.setMode(3)  # Recording 시작
    print(f"TDT 녹화 시작: {block_name}")
except Exception as e:
    print(f"TDT 설정 실패: {e}")
    core.quit()


# =========================================================
# 3. Paths / ExperimentHandler
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
STIM_DIR = os.path.join(SCRIPT_DIR, 'Stimuli_48k')
os.makedirs(DATA_DIR, exist_ok=True)

data_filename = os.path.join(
    DATA_DIR,
    f'sub-KOREAN{subj_idx:03d}_ses-{idx_day:02d}_task-selattkor'
)

thisExp = data.ExperimentHandler(
    name='asa_korean',
    version='2025.2.4',
    extraInfo=exp_info,
    runtimeInfo=None,
    originPath=__file__,
    savePickle=False,
    saveWideText=True,
    dataFileName=data_filename,
)

logging.console.setLevel(logging.WARNING)


# =========================================================
# 4. Counterbalance load
# =========================================================
cb_path = os.path.join(STIM_DIR, 'counterbalance_structure.csv')
if not os.path.exists(cb_path):
    raise FileNotFoundError(f'counterbalance_structure.csv not found: {cb_path}')

T = pd.read_csv(cb_path)

cb_subj = (subj_idx - 1) % 12 + 1
this_rows = (
    T[(T['Subject'] == cb_subj) & (T['day'] == idx_day)]
    .sort_values('block')
    .reset_index(drop=True)
)

if len(this_rows) == 0:
    raise ValueError(f'No counterbalance rows for Subject={cb_subj}, day={idx_day}')

print(f'\nSubject {subj_idx} (CB row {cb_subj}), Day {idx_day}')
print(this_rows[['day', 'block', 'Condition', 'AttendedGender']])


# =========================================================
# 5. Parameters
# =========================================================
FS_AUDIO = 48000
TRIALS_PER_HALF = 15
TRIALS_PER_BLOCK = 30
NUM_HALVES = 2
PAUSE_BEFORE_PLAY = 3.0
FEEDBACK_DUR = 1.0
INPUT_GUARD = 0.01  # 화면 뜬 직후 이 시간 동안 입력 무시

num_blocks = len(this_rows)

random.seed()
stim_order = {
    int(row['block']): random.sample(range(1, TRIALS_PER_BLOCK + 1), TRIALS_PER_BLOCK)
    for _, row in this_rows.iterrows()
}


# =========================================================
# 6. Window
# =========================================================
win = visual.Window(
    fullscr=True,
    color='black',
    colorSpace='rgb',
    units='norm',
    allowGUI=False,
    screen=1,
)
win.mouseVisible = False

globalClock = core.Clock()
logging.setDefaultClock(globalClock)


# =========================================================
# 7. pygame audio init
# =========================================================
pygame.mixer.pre_init(frequency=FS_AUDIO, size=-16, channels=2, buffer=512)
pygame.init()
pygame.mixer.init()


# =========================================================
# 8. Text stimuli
# =========================================================
FONT = 'AppleGothic'
WRAP_CENTER = 1.6
WRAP_LEFT = 1.56

text_center = visual.TextStim(
    win,
    text='',
    font=FONT,
    color='white',
    height=0.1,
    wrapWidth=WRAP_CENTER,
    alignText='center',
    anchorHoriz='center',
    anchorVert='center',
)

text_rating = visual.TextStim(
    win,
    text='',
    font=FONT,
    color='white',
    height=0.1,
    wrapWidth=WRAP_CENTER,
    alignText='center',
    anchorHoriz='center',
    anchorVert='center',
)

text_question = visual.TextStim(
    win,
    text='',
    font=FONT,
    color='white',
    height=0.1,
    wrapWidth=WRAP_LEFT,
    alignText='left',
    anchorHoriz='center',
    anchorVert='center',
)


# =========================================================
# 9. Helpers
# =========================================================
class KeyResult:
    def __init__(self, name, t_down):
        self.name = name
        self.tDown = t_down


def cleanup_audio():
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
    try:
        pygame.mixer.quit()
    except Exception:
        pass
    try:
        pygame.quit()
    except Exception:
        pass


def save_and_close():
    try:
        thisExp.saveAsWideText(data_filename + '.csv')
    except Exception:
        pass
    send_trigger(255)   # 세션 종료 트리거
    syn.setMode(0)      # TDT 녹화 종료
    thisExp.abort()
    cleanup_audio()
    win.close()
    core.quit()


def show_center(text, height=0.1):
    event.clearEvents(eventType='keyboard')
    text_center.text = text
    text_center.height = height
    text_center.draw()
    win.flip()
    return globalClock.getTime()


def show_rating(text):
    event.clearEvents(eventType='keyboard')
    text_rating.text = text
    text_rating.draw()
    win.flip()
    return globalClock.getTime()


def show_question(text):
    event.clearEvents(eventType='keyboard')
    text_question.text = text
    text_question.draw()
    win.flip()
    return globalClock.getTime()


def wait_space(onset_t, min_delay=INPUT_GUARD):
    while True:
        now = globalClock.getTime()
        if now - onset_t < min_delay:
            event.clearEvents(eventType='keyboard')
            core.wait(0.001)
            continue
        keys = event.getKeys(keyList=['space', 'escape'])
        if keys:
            if 'escape' in keys:
                save_and_close()
            if 'space' in keys:
                return globalClock.getTime()
        core.wait(0.001)


def wait_key(allowed, onset_t, min_delay=INPUT_GUARD):
    while True:
        now = globalClock.getTime()
        if now - onset_t < min_delay:
            event.clearEvents(eventType='keyboard')
            core.wait(0.001)
            continue
        keys = event.getKeys(keyList=allowed + ['escape'])
        if keys:
            if 'escape' in keys:
                save_and_close()
            for k in keys:
                if k in allowed:
                    return KeyResult(k, globalClock.getTime())
        core.wait(0.001)


# =========================================================
# 10. Session start trigger + Instructions
# =========================================================
send_trigger(254)   # 세션 시작 트리거

start_onset_t = show_center(
    '두 개의 이야기가 동시에 들립니다.\n'
    '목표 화자는 맨 처음에 지정됩니다.\n'
    '지시된 목표 화자의 목소리에 집중하여 들으세요.\n\n'
    '시작하려면 스페이스바를 누르세요',
    height=0.1,
)
wait_space(onset_t=start_onset_t)

show_center('+', height=0.15)


# =========================================================
# 11. Main loop
# =========================================================
for _, cb_row in this_rows.iterrows():
    idx_block = int(cb_row['block'])
    audio_folder = os.path.join(STIM_DIR, f'day{idx_day}block{idx_block}')

    mixinfo_path = os.path.join(audio_folder, 'mixture_info.csv')
    if not os.path.exists(mixinfo_path):
        raise FileNotFoundError(f'mixture_info.csv not found: {mixinfo_path}')

    tinfo = pd.read_csv(mixinfo_path, encoding='utf-8')

    for half in range(1, NUM_HALVES + 1):
        start_idx = 0 if half == 1 else TRIALS_PER_HALF
        end_idx = TRIALS_PER_HALF if half == 1 else TRIALS_PER_BLOCK

        for trial_pos in range(start_idx, end_idx):
            trial_in_half = trial_pos - start_idx + 1
            resp_idx = (idx_block - 1) * TRIALS_PER_BLOCK + trial_pos + 1
            real_block = (idx_block - 1) * NUM_HALVES + half
            trial_label = f'{real_block}-{trial_in_half}.'

            stim_idx = stim_order[idx_block][trial_pos]
            audio_fname = f'day{idx_day}block{idx_block}_stimulus{stim_idx:02d}.wav'
            audio_path = os.path.join(audio_folder, 'mixture', audio_fname)

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f'Audio file not found: {audio_path}')

            print(f'Block {idx_block} | half {half} | trial {trial_in_half} | stim {stim_idx}')

            # Fixation + Trial 시작 트리거
            show_center('+', height=0.3)
            trial_start_t = globalClock.getTime()
            send_trigger(200)

            static = core.StaticPeriod(screenHz=None, win=win)
            static.start(PAUSE_BEFORE_PLAY)

            _fs_wav, _audio_np = _wavfile.read(audio_path)
            if np.issubdtype(_audio_np.dtype, np.integer):
                _audio_np = _audio_np.astype(np.float32) / np.iinfo(_audio_np.dtype).max
            else:
                _audio_np = _audio_np.astype(np.float32)
            rms_val = float(np.sqrt(np.mean(_audio_np ** 2)))

            static.complete()

            # Stimulus onset 트리거
            audio_onset_t = globalClock.getTime()
            send_trigger(stim_idx)

            audio_skipped = 0

            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()

            event.clearEvents(eventType='keyboard')

            while pygame.mixer.music.get_busy():
                keys = event.getKeys()
                if 'escape' in keys:
                    save_and_close()
                if 'right' in keys:
                    pygame.mixer.music.stop()
                    audio_skipped = 1
                    break
                core.wait(0.005)

            # -----------------------------
            # Attention rating
            # Trigger 211 = onset, 221 = 응답 완료
            # -----------------------------
            att_q_onset_t = show_rating(
                f'{trial_label} 목표 화자의 목소리에 얼마나 집중이 잘 되었나요?\n\n'
                '    1    2    3    4    5    6    7\n\n'
                '전혀 안됨                     매우 잘됨'
            )
            send_trigger(211)   # Attention onset

            att_key = wait_key(['1', '2', '3', '4', '5', '6', '7'], onset_t=att_q_onset_t)
            att_resp_t = att_key.tDown
            event.clearEvents(eventType='keyboard')  # 추가
            send_trigger(221)   # Attention 응답 완료
            attention_rating = int(att_key.name)

            # -----------------------------
            # Comprehension question
            # Trigger 212 = onset, 222 = 응답 완료
            # -----------------------------
            trow = tinfo.iloc[stim_idx - 1]
            options = [str(trow[f'option{i}']) for i in range(1, 5)]

            correct_str = str(trow['answer'])
            try:
                correct_answer = options.index(correct_str) + 1
            except ValueError:
                correct_answer = -1
                logging.warning(f'Answer "{correct_str}" not found in options.')

            comp_text = (
                f'{trial_label} {trow["question"]}\n\n'
                + '\n'.join(f'{i+1}. {opt}' for i, opt in enumerate(options))
            )
            comp_q_onset_t = show_question(comp_text)
            send_trigger(212)   # Comprehension onset

            comp_key = wait_key(['1', '2', '3', '4'], onset_t=comp_q_onset_t)
            comp_resp_t = comp_key.tDown
            event.clearEvents(eventType='keyboard')  # 추가
            send_trigger(222)   # Comprehension 응답 완료 (221 → 222로 수정)

            selected_answer = int(comp_key.name)
            hit = int(selected_answer == correct_answer)

            # -----------------------------
            # Feedback
            # Trigger 213 = 피드백 onset
            # -----------------------------
            show_center('정답입니다!' if hit else '오답입니다...', height=0.1)
            send_trigger(213)   # 피드백 onset
            core.wait(FEEDBACK_DUR)

            # -----------------------------
            # Save data
            # -----------------------------
            thisExp.addData('KOREANid', korean_id)
            thisExp.addData('expId', exp_id)
            thisExp.addData('subj_idx', subj_idx)
            thisExp.addData('idx_day', idx_day)
            thisExp.addData('block_idx', idx_block)
            thisExp.addData('half', half)
            thisExp.addData('real_block', real_block)
            thisExp.addData('trial_in_half', trial_in_half)
            thisExp.addData('trial_label', trial_label)
            thisExp.addData('resp_idx', resp_idx)
            thisExp.addData('stim_idx', stim_idx)
            thisExp.addData('audio_file', audio_path)
            thisExp.addData('rms_mix', round(rms_val, 6))
            thisExp.addData('audio_skipped', audio_skipped)

            thisExp.addData('Group', cb_row.get('Group', ''))
            thisExp.addData('StorySetIdx', cb_row.get('StorySetIdx', ''))
            thisExp.addData('StoryIdx1', cb_row.get('StoryIdx1', ''))
            thisExp.addData('StoryIdx2', cb_row.get('StoryIdx2', ''))
            thisExp.addData('StoryPair', cb_row.get('StoryPair', ''))
            thisExp.addData('Condition', cb_row.get('Condition', ''))
            thisExp.addData('AttendedGender', cb_row.get('AttendedGender', ''))
            thisExp.addData('AttendedStory', cb_row.get('AttendedStory', ''))
            thisExp.addData('MaleStory', cb_row.get('MaleStory', ''))
            thisExp.addData('FemaleStory', cb_row.get('FemaleStory', ''))

            thisExp.addData('attention_rating', attention_rating)

            thisExp.addData('question', trow['question'])
            thisExp.addData('option1', options[0])
            thisExp.addData('option2', options[1])
            thisExp.addData('option3', options[2])
            thisExp.addData('option4', options[3])

            thisExp.addData('selected_answer', selected_answer)
            thisExp.addData('correct_answer', correct_answer)
            thisExp.addData('hit', hit)

            thisExp.addData('trial_start_t', round(trial_start_t, 4))
            thisExp.addData('audio_onset_t', round(audio_onset_t, 4))
            thisExp.addData('att_q_onset_t', round(att_q_onset_t, 4))
            thisExp.addData('att_rt', round(att_resp_t - att_q_onset_t, 4))
            thisExp.addData('att_resp_t', round(att_resp_t, 4))
            thisExp.addData('comp_q_onset_t', round(comp_q_onset_t, 4))
            thisExp.addData('comp_rt', round(comp_resp_t - comp_q_onset_t, 4))
            thisExp.addData('comp_resp_t', round(comp_resp_t, 4))

            thisExp.nextEntry()

            if trial_in_half != TRIALS_PER_HALF:
                show_center('+', height=0.15)

        print(f'REST SCREEN | block={idx_block}, half={half}')
        is_last = (idx_block == num_blocks and half == NUM_HALVES)
        if not is_last:
            rest_onset_t = show_center(
                '원하는 만큼 휴식을 취하세요.\n\n'
                '계속하려면 스페이스바를 누르세요.',
                height=0.1,
            )
            wait_space(onset_t=rest_onset_t)

# =========================================================
# 12. End
# =========================================================
thisExp.saveAsWideText(data_filename + '.csv')
thisExp.abort()

send_trigger(255)   # 세션 종료 트리거
core.wait(1)
syn.setMode(0)      # TDT 녹화 종료
print("TDT 녹화 종료")

end_onset_t = show_center(
    '실험이 종료되었습니다. 수고 많으셨습니다!\n\n'
    '잠시 대기하여 담당자의 안내를 기다려 주세요.',
    height=0.1
)
wait_space(onset_t=end_onset_t)

cleanup_audio()
win.close()
core.quit()