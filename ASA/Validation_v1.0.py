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
    'session': '001',
}

dlg = gui.DlgFromDict(
    dictionary=exp_info,
    title='Validation ERP Experiment',
    order=['subject_number', 'experimenter_id', 'session'],
)
if not dlg.OK:
    core.quit()

subj_idx = int(exp_info['subject_number'])
participant = f"VAL{subj_idx:03d}"
session = exp_info['session']


# =========================================================
# 2.5. TDT Synapse connection
# =========================================================
try:
    syn = tdt.SynapseAPI()
    print("TDT Synapse 연결 성공")
    print(f"현재 모드: {syn.getMode()}")
    print(f"현재 유저: {syn.getCurrentUser()}")
    print(f"현재 실험: {syn.getCurrentExperiment()}")
    print(f"현재 탱크: {syn.getCurrentTank()}")
    core.wait(1)
except Exception as e:
    print(f"TDT 연결 실패: {e}")
    core.quit()

clean_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
block_name = f"SNU{subj_idx:03d}_day{int(session)}_ERP_{clean_datetime}"

try:
    if syn.getMode() != 0:
        syn.setMode(0)
    syn.setCurrentUser("KIST_ducky")
    syn.setCurrentExperiment("LingcogERP")
    syn.setCurrentBlock(block_name)
    syn.setMode(3)  # Recording 시작

    current_tank = syn.getCurrentTank()
    block_dir = os.path.join(current_tank, block_name)
    os.makedirs(block_dir, exist_ok=True)
    print(f"TDT 녹화 시작: {block_name}")
    print(f"데이터 저장 경로: {block_dir}")
except Exception as e:
    print(f"TDT 설정 실패: {e}")
    core.quit()


# =========================================================
# 3. Paths / ExperimentHandler
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

data_filename = os.path.join(
    block_dir,
    f'sub-VAL{subj_idx:03d}_ses-{int(session):03d}_task-validation'
)

thisExp = data.ExperimentHandler(
    name='validation_erp',
    version='1.0',
    extraInfo=exp_info,
    runtimeInfo=None,
    originPath=__file__,
    savePickle=False,
    saveWideText=True,
    dataFileName=data_filename,
)

logging.console.setLevel(logging.WARNING)


# =========================================================
# 4. Stimulus load
# =========================================================
STIM_XLSX = os.path.join(SCRIPT_DIR, 'erp_stimuli', 'erp_stimuli.xlsx')
stim_df = pd.read_excel(STIM_XLSX)

N_REPS = 30  # 5행 × 30 = 150 trials
trial_list = pd.concat([stim_df] * N_REPS, ignore_index=True)
trial_list = trial_list.sample(frac=1).reset_index(drop=True)  # 랜덤 셔플

print(f"총 트라이얼: {len(trial_list)}개")


# =========================================================
# 5. Window
# =========================================================
win = visual.Window(
    fullscr=True,
    color='black',
    colorSpace='rgb',
    units='height',
    allowGUI=False,
    screen=1,
)
win.mouseVisible = False

globalClock = core.Clock()
logging.setDefaultClock(globalClock)


# =========================================================
# 6. pygame audio init
# =========================================================
FS_AUDIO = 48000
pygame.mixer.pre_init(frequency=FS_AUDIO, size=-16, channels=2, buffer=512)
pygame.init()
pygame.mixer.init()


# =========================================================
# 7. Text stimuli
# =========================================================
text_center = visual.TextStim(
    win,
    text='',
    font='Arial',
    color='white',
    height=0.05,
    anchorHoriz='center',
    anchorVert='center',
)


# =========================================================
# 8. Helpers
# =========================================================
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
    send_trigger(255)   # 세션 종료
    syn.setMode(0)
    thisExp.abort()
    cleanup_audio()
    win.close()
    core.quit()


def show_center(text, height=0.05):
    event.clearEvents(eventType='keyboard')
    text_center.text = text
    text_center.height = height
    text_center.draw()
    win.flip()
    return globalClock.getTime()


def wait_key(keylist, min_delay=0.01):
    onset_t = globalClock.getTime()
    while True:
        now = globalClock.getTime()
        if now - onset_t < min_delay:
            event.clearEvents(eventType='keyboard')
            core.wait(0.001)
            continue
        keys = event.getKeys(keyList=keylist + ['escape'])
        if 'escape' in keys:
            save_and_close()
        for k in keys:
            if k in keylist:
                return k
        core.wait(0.001)


# =========================================================
# 9. Instructions
# =========================================================
show_center(
    '지금부터 화면 중앙의 점을 응시하면서\n'
    '소리를 듣고 있으면 됩니다.\n\n'
    '긴장을 풀고 편한 자세로 있되,\n'
    '몸은 최대한 움직이지 말아주세요.\n\n'
    '준비되셨으면 스페이스바를 눌러주세요.',
    height=0.04,
)
wait_key(['space'])

# =========================================================
# 10. Session start trigger
# =========================================================
send_trigger(254)   # 세션 시작
experiment_start_time = globalClock.getTime()

show_center('+', height=0.1)
core.wait(2.0)  # ERP_start 대기 (기존 2초)


# =========================================================
# 11. Main loop
# =========================================================
for trial_num, row in trial_list.iterrows():
    fname = str(row['fname'])
    trigger_id = int(row['trigger_id'])
    isi = float(row['isi'])

    audio_path = os.path.join(SCRIPT_DIR, fname)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Audio not found: {audio_path}')

    print(f"Trial {trial_num+1}/150 | trigger={trigger_id} | isi={isi}s")

    # Fixation
    show_center('+', height=0.1)

    # 오디오 재생 + 트리거
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    send_trigger(trigger_id)

    # 오디오 끝날 때까지 대기
    while pygame.mixer.music.get_busy():
        keys = event.getKeys()
        if 'escape' in keys:
            save_and_close()
        core.wait(0.005)

    # ISI 대기
    core.wait(isi)

    # 데이터 저장
    thisExp.addData('trial_num', trial_num + 1)
    thisExp.addData('fname', fname)
    thisExp.addData('trigger_id', trigger_id)
    thisExp.addData('isi', isi)
    thisExp.nextEntry()


# =========================================================
# 12. End
# =========================================================
show_center('+', height=0.1)
core.wait(3.0)  # ERP_end 대기 (기존 3초)

thisExp.saveAsWideText(data_filename + '.csv')
thisExp.abort()

send_trigger(255)   # 세션 종료
core.wait(1)
syn.setMode(0)
print(f"TDT 녹화 종료 | 총 실험 시간: {globalClock.getTime() - experiment_start_time:.1f}초")

show_center(
    '실험이 종료되었습니다.\n\n'
    '수고 많으셨습니다!',
    height=0.05,
)
core.wait(5)

cleanup_audio()
win.close()
core.quit()