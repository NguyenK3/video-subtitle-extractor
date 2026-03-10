# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao
@Time    : 2021/4/1 6:07 下午
@FileName: gui.py
@desc: 字幕提取器图形化界面
"""
import backend.main
import os
import configparser
import PySimpleGUI as sg
import cv2
import numpy as np
from threading import Thread
import multiprocessing


class SubtitleExtractorGUI:
    def _load_config(self):
        self.config_file = os.path.join(os.path.dirname(__file__), 'settings.ini')
        self.subtitle_config_file = os.path.join(os.path.dirname(__file__), 'subtitle.ini')
        self.config = configparser.ConfigParser()
        self.interface_config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            # 如果没有配置文件，默认弹出语言选择界面
            LanguageModeGUI(self).run()
        self.INTERFACE_KEY_NAME_MAP = {
            '简体中文': 'ch',
            '繁體中文': 'chinese_cht',
            'English': 'en',
            '한국어': 'ko',
            '日本語': 'japan',
            'Tiếng Việt': 'vi',
            'Español': 'es'
        }
        self.config.read(self.config_file, encoding='utf-8')
        self.interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'interface',
                                           f"{self.INTERFACE_KEY_NAME_MAP[self.config['DEFAULT']['Interface']]}.ini")
        self.interface_config.read(self.interface_file, encoding='utf-8')

    def __init__(self):
        # 初次运行检查运行环境是否正常
        from paddle import utils
        utils.run_check()
        self.font = ('DejaVu Sans', 10)
        self.font_mono = ('DejaVu Sans Mono', 10)
        self.theme = 'DarkGrey14'
        sg.theme(self.theme)
        self.icon = os.path.join(os.path.dirname(__file__), 'design', 'vse.ico')
        self._load_config()
        self.screen_width, self.screen_height = sg.Window.get_screen_size()
        # 设置视频预览区域大小
        self.video_preview_width = min(1120, max(720, int(self.screen_width * 0.62)))
        self.video_preview_height = self.video_preview_width * 9 // 16
        self.sidebar_width = min(420, max(320, int(self.screen_width * 0.30)))
        self.progressbar_size = (42, 18)
        # 字幕提取器布局
        self.layout = None
        # 字幕提取其窗口
        self.window = None
        # 视频路径
        self.video_path = None
        # 视频cap
        self.video_cap = None
        # 视频的帧率
        self.fps = None
        # 视频的帧数
        self.frame_count = None
        # 视频的宽
        self.frame_width = None
        # 视频的高
        self.frame_height = None
        # 设置字幕区域高宽
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        # 字幕提取器
        self.se = None
        self.video_paths = []
        self.current_frame_no = 1
        self.current_frame = None
        self.is_playing = False
        self.preview_text = ''
        self.log_entries = []
        self.filter_level = 'ALL'
        self.last_logged_drawn_frame = None
        self.last_logged_decoded_frame = None

    @staticmethod
    def _normalize_selected_files(raw_selection):
        if raw_selection is None:
            return []
        if isinstance(raw_selection, str):
            candidates = raw_selection.split(';')
        elif isinstance(raw_selection, (list, tuple, set)):
            candidates = []
            for entry in raw_selection:
                if isinstance(entry, str):
                    candidates.extend(entry.split(';'))
        else:
            candidates = [str(raw_selection)]
        return [path.strip() for path in candidates if isinstance(path, str) and path.strip()]

    def _log_frame_activity(self, action, frame_no, force=False):
        if frame_no is None:
            return
        frame_no = int(frame_no)
        if action == 'Decoded':
            last_value = self.last_logged_decoded_frame
        else:
            last_value = self.last_logged_drawn_frame

        should_log = force or not self.is_playing
        if self.is_playing and self.fps:
            should_log = should_log or frame_no % max(1, int(round(self.fps))) == 0
        if self.frame_count:
            should_log = should_log or frame_no >= int(self.frame_count)

        if not should_log or last_value == frame_no:
            return

        self._log(f'{action} frame {frame_no}', 'INFO')
        if action == 'Decoded':
            self.last_logged_decoded_frame = frame_no
        else:
            self.last_logged_drawn_frame = frame_no

    def _open_video_capture(self, video_path):
        cap = None
        try:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        except Exception:
            cap = cv2.VideoCapture(video_path)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(video_path)

        hw_prop = getattr(cv2, 'CAP_PROP_HW_ACCELERATION', None)
        hw_none = getattr(cv2, 'VIDEO_ACCELERATION_NONE', None)
        if hw_prop is not None and hw_none is not None and cap is not None:
            try:
                cap.set(hw_prop, hw_none)
            except Exception:
                pass
        return cap

    @staticmethod
    def _is_probably_black_frame(frame):
        if frame is None or frame.size == 0:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        return float(np.mean(gray)) < 1.5 and float(np.std(gray)) < 1.0

    def _read_frame_safely(self, frame_no=None):
        if self.video_cap is None or not self.video_cap.isOpened() or self.video_path is None:
            return False, None
        if frame_no is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_no) - 1))
        ret, frame = self.video_cap.read()
        if ret and not self._is_probably_black_frame(frame):
            return ret, frame

        self._log('Frame decode fallback: reopening capture with software path', 'WARN')
        try:
            self.video_cap.release()
        except Exception:
            pass
        self.video_cap = self._open_video_capture(self.video_path)
        if self.video_cap is None or not self.video_cap.isOpened():
            return False, None
        if frame_no is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_no) - 1))
        ret2, frame2 = self.video_cap.read()
        return ret2, frame2

    def _log(self, message, level='INFO'):
        line = f'[{level}] {message}'
        self.log_entries.append((level, line))
        print(line)
        if self.window is None:
            return
        if self.filter_level != 'ALL' and level != self.filter_level:
            return
        self.window['-RESULTS-'].print(line)

    def run(self):
        print(self.interface_config['Main']['GPUWarning'])
        # 创建布局
        self._create_layout()
        # 创建窗口
        self.window = sg.Window(title=self.interface_config['SubtitleExtractorGUI']['Title'] + " v" + backend.main.config.VERSION, layout=self.layout,
                                icon=self.icon, finalize=True, resizable=True)
        while True:
            # 循环读取事件
            event, values = self.window.read(timeout=80)
            self.filter_level = values.get('-LOG-FILTER-', 'ALL')
            # 处理【打开】事件
            self._file_event_handler(event, values)
            # 处理【滑动】事件
            self._slide_event_handler(event, values)
            # 处理【识别语言】事件
            self._language_mode_event_handler(event)
            # 处理【运行】事件
            self._run_event_handler(event, values)
            # 处理播放控制
            self._playback_event_handler(event, values)
            self._refresh_status(values)
            # 如果关闭软件，退出
            if event == sg.WIN_CLOSED:
                break
            # 更新进度条
            if self.se is not None:
                self.window['-PROG-'].update(self.se.progress_total)
                if self.se.isFinished:
                    # 1) 打开修改字幕滑块区域按钮
                    self.window['-Y-SLIDER-'].update(disabled=False)
                    self.window['-X-SLIDER-'].update(disabled=False)
                    self.window['-Y-SLIDER-H-'].update(disabled=False)
                    self.window['-X-SLIDER-W-'].update(disabled=False)
                    # 2) 打开【运行】、【打开】和【识别语言】按钮
                    self.window['-RUN-'].update(disabled=False)
                    self.window['-FILE-'].update(disabled=False)
                    self.window['-FILE_BTN-'].update(disabled=False)
                    self.window['-LANGUAGE-MODE-'].update(disabled=False)
                    self.window['-PLAY-'].update(disabled=False)
                    self.window['-PAUSE-'].update(disabled=False)
                    self.window['-PREV-'].update(disabled=False)
                    self.window['-NEXT-'].update(disabled=False)
                    self._refresh_subtitle_preview()
                    self.se = None
                if len(self.video_paths) >= 1:
                    # 1) 关闭修改字幕滑块区域按钮
                    self.window['-Y-SLIDER-'].update(disabled=True)
                    self.window['-X-SLIDER-'].update(disabled=True)
                    self.window['-Y-SLIDER-H-'].update(disabled=True)
                    self.window['-X-SLIDER-W-'].update(disabled=True)
                    # 2) 关闭【运行】、【打开】和【识别语言】按钮
                    self.window['-RUN-'].update(disabled=True)
                    self.window['-FILE-'].update(disabled=True)
                    self.window['-FILE_BTN-'].update(disabled=True)
                    self.window['-LANGUAGE-MODE-'].update(disabled=True)
        if self.video_cap is not None:
            self.video_cap.release()

    def _refresh_status(self, values):
        if self.window is None:
            return
        fps = 0.0 if self.fps is None else self.fps
        total = 1 if self.frame_count is None else int(self.frame_count)
        self.window['-STATUS-'].update(f'Frame {self.current_frame_no}/{total} | FPS {fps:.2f}')
        if self.video_cap is not None and self.video_cap.isOpened():
            self.window['-SUB-PREVIEW-'].update(self.preview_text or 'Detected subtitle preview: --')

    def _refresh_subtitle_preview(self):
        if self.video_path is None:
            return
        srt_path = os.path.splitext(self.video_path)[0] + '.srt'
        if not os.path.exists(srt_path):
            return
        last_text = ''
        with open(srt_path, mode='r', encoding='utf-8', errors='ignore') as f:
            for raw in f.readlines():
                line = raw.strip()
                if line and '-->' not in line and not line.isdigit():
                    last_text = line
        if last_text:
            self.preview_text = f'Detected subtitle preview: {last_text}'
            self.window['-SUB-PREVIEW-'].update(self.preview_text)

    def update_interface_text(self):
        self._load_config()
        self.window.set_title(self.interface_config['SubtitleExtractorGUI']['Title'] + " v" + backend.main.config.VERSION)
        self.window['-FILE_BTN-'].Update(self.interface_config['SubtitleExtractorGUI']['Open'])
        self.window['-RUN-'].Update(self.interface_config['SubtitleExtractorGUI']['Run'])
        self.window['-LANGUAGE-MODE-'].Update(self.interface_config['SubtitleExtractorGUI']['Setting'])

    def _create_layout(self):
        """
        创建字幕提取器布局
        """
        garbage = os.path.join(os.path.dirname(__file__), 'output')
        if os.path.exists(garbage):
            import shutil
            shutil.rmtree(garbage, True)
        left_layout = [
            [sg.Frame('Preview', [[
                sg.Image(size=(self.video_preview_width, self.video_preview_height), background_color='#111111', key='-DISPLAY-')
            ]], pad=(8, 8), background_color='#252526', relief=sg.RELIEF_SOLID, border_width=1)],
            [sg.Text('Detected subtitle preview: --', key='-SUB-PREVIEW-', text_color='#e0e0e0', background_color='#1e1e1e', font=self.font)],
            [sg.Frame('Timeline', [[
                sg.Slider(size=(max(40, int(self.video_preview_width / 15)), 16), range=(1, 1), key='-SLIDER-', orientation='h',
                          enable_events=True, disable_number_display=False, text_color='#e0e0e0',
                          trough_color='#3a3a3a', background_color='#252526'),
            ], [
                sg.Button('Open Video', key='-OPEN-BTN-', button_color=('#ffffff', '#4CAF50'), size=(11, 1), border_width=0),
                sg.Button('◀ Frame', key='-PREV-', size=(8, 1), button_color=('#e0e0e0', '#3c3c3c'), border_width=0),
                sg.Button('Play', key='-PLAY-', size=(7, 1), button_color=('#ffffff', '#3c3c3c'), border_width=0),
                sg.Button('Pause', key='-PAUSE-', size=(7, 1), button_color=('#ffffff', '#3c3c3c'), border_width=0),
                sg.Button('Frame ▶', key='-NEXT-', size=(8, 1), button_color=('#e0e0e0', '#3c3c3c'), border_width=0),
                sg.Text('Frame 0/0 | FPS 0.0', key='-STATUS-', text_color='#bdbdbd', background_color='#252526')
            ]], background_color='#252526', relief=sg.RELIEF_SOLID, border_width=1, pad=(8, 4))],
            [sg.Frame('Logs / Console', [[
                sg.Output(size=(int(self.video_preview_width / 11), 11), font=self.font_mono, key='-LOG-')
            ]], pad=(8, 6), background_color='#252526', relief=sg.RELIEF_SOLID, border_width=1)],
            [sg.Frame('Subtitle Extraction Results', [[
                sg.Text('Filter', background_color='#252526', text_color='#e0e0e0'),
                sg.Combo(values=['ALL', 'INFO', 'WARN', 'ERROR'], default_value='ALL', key='-LOG-FILTER-', readonly=True, size=(8, 1)),
            ], [
                sg.Multiline('', size=(int(self.video_preview_width / 11), 8), key='-RESULTS-', autoscroll=True,
                             disabled=True, font=self.font_mono, background_color='#101010', text_color='#c8c8c8')
            ]], pad=(8, 4), background_color='#252526', relief=sg.RELIEF_SOLID, border_width=1)],
        ]

        right_layout = [
            [sg.Frame('Settings', [[
                sg.Input(key='-FILE-', visible=False, enable_events=True),
                sg.FilesBrowse(button_text=self.interface_config['SubtitleExtractorGUI']['Open'], file_types=(
                    (self.interface_config['SubtitleExtractorGUI']['AllFile'], '*.*'),
                    ('mp4', '*.mp4'), ('flv', '*.flv'), ('wmv', '*.wmv'), ('avi', '*.avi')
                ), key='-FILE_BTN-', size=(18, 1), font=self.font),
                sg.Button(button_text=self.interface_config['SubtitleExtractorGUI']['Run'], key='-RUN-',
                          font=self.font, size=(10, 1), button_color=('#ffffff', '#4CAF50'), border_width=0),
            ], [
                sg.Button(button_text=self.interface_config['SubtitleExtractorGUI']['Setting'], key='-LANGUAGE-MODE-',
                          font=self.font, size=(30, 1), button_color=('#ffffff', '#3c3c3c'), border_width=0),
            ], [
                sg.Text('OCR Language', text_color='#e0e0e0', background_color='#252526'),
                sg.Text(self.config['DEFAULT']['Language'], key='-LANG-LABEL-', text_color='#a6d189', background_color='#252526')
            ], [
                sg.Text('Performance Mode', text_color='#e0e0e0', background_color='#252526'),
                sg.Text(self.config['DEFAULT']['Mode'], key='-MODE-LABEL-', text_color='#a6d189', background_color='#252526')
            ], [
                sg.Text('Sampling Rate', text_color='#e0e0e0', background_color='#252526'),
                sg.Slider(range=(1, 10), default_value=3, orientation='h', size=(24, 16),
                          key='-SAMPLING-SLIDER-', disable_number_display=False,
                          trough_color='#3a3a3a', background_color='#252526')
            ], [
                sg.Text('Detection Sensitivity', text_color='#e0e0e0', background_color='#252526'),
                sg.Slider(range=(60, 95), default_value=80, orientation='h', size=(24, 16),
                          key='-SENS-SLIDER-', disable_number_display=False,
                          trough_color='#3a3a3a', background_color='#252526')
            ]], background_color='#252526', relief=sg.RELIEF_SOLID, border_width=1, pad=(6, 8))],
            [sg.Frame('Subtitle Bounding Box', [[
                sg.Text(self.interface_config['SubtitleExtractorGUI']['Vertical'], background_color='#252526', text_color='#e0e0e0'),
                sg.Slider(range=(0, 0), orientation='h', size=(22, 16), disable_number_display=False,
                          enable_events=True, default_value=0, key='-Y-SLIDER-', trough_color='#3a3a3a', background_color='#252526')
            ], [
                sg.Text('Height', background_color='#252526', text_color='#e0e0e0'),
                sg.Slider(range=(0, 0), orientation='h', size=(22, 16), disable_number_display=False,
                          enable_events=True, default_value=0, key='-Y-SLIDER-H-', trough_color='#3a3a3a', background_color='#252526')
            ], [
                sg.Text(self.interface_config['SubtitleExtractorGUI']['Horizontal'], background_color='#252526', text_color='#e0e0e0'),
                sg.Slider(range=(0, 0), orientation='h', size=(22, 16), disable_number_display=False,
                          enable_events=True, default_value=0, key='-X-SLIDER-', trough_color='#3a3a3a', background_color='#252526')
            ], [
                sg.Text('Width', background_color='#252526', text_color='#e0e0e0'),
                sg.Slider(range=(0, 0), orientation='h', size=(22, 16), disable_number_display=False,
                          enable_events=True, default_value=0, key='-X-SLIDER-W-', trough_color='#3a3a3a', background_color='#252526')
            ]], background_color='#252526', relief=sg.RELIEF_SOLID, border_width=1, pad=(6, 6))],
            [sg.Frame('Progress', [[
                sg.ProgressBar(100, orientation='h', size=self.progressbar_size, key='-PROG-', bar_color=('#4CAF50', '#3a3a3a'))
            ]], background_color='#252526', relief=sg.RELIEF_SOLID, border_width=1, pad=(6, 6))],
        ]

        self.layout = [[
            sg.Column(left_layout, background_color='#1e1e1e', expand_x=True, expand_y=True, scrollable=False),
            sg.VSeparator(color='#2f2f2f'),
            sg.Column(right_layout, size=(self.sidebar_width, self.screen_height - 120), background_color='#1e1e1e', scrollable=True, vertical_scroll_only=True)
        ]]

    def _file_event_handler(self, event, values):
        """
        当点击打开按钮时：
        1）打开视频文件，将画布显示视频帧
        2）获取视频信息，初始化进度条滑块范围
        """
        if event == '-FILE-' or event == '-OPEN-BTN-':
            if event == '-OPEN-BTN-':
                selected = sg.popup_get_file(
                    self.interface_config['SubtitleExtractorGUI']['Open'],
                    file_types=((self.interface_config['SubtitleExtractorGUI']['AllFile'], '*.*'),
                                ('mp4', '*.mp4'), ('flv', '*.flv'), ('wmv', '*.wmv'), ('avi', '*.avi')),
                    multiple_files=True,
                    no_window=True
                )
                if not selected:
                    return
                values['-FILE-'] = selected
            self.video_paths = self._normalize_selected_files(values.get('-FILE-'))
            if not self.video_paths:
                self._log('No video file was selected.', 'WARN')
                return
            self.video_path = self.video_paths[0]
            self._log(f'Opening video: {self.video_path}')
            if self.video_path != '':
                self.video_cap = self._open_video_capture(self.video_path)
            if self.video_cap is None:
                return
            if self.video_cap.isOpened():
                ret, frame = self._read_frame_safely(frame_no=1)
                if ret:
                    for video in self.video_paths:
                        self._log(f"{self.interface_config['SubtitleExtractorGUI']['OpenVideoSuccess']}：{video}")
                    # 获取视频的帧数
                    self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # 获取视频的高度
                    self.frame_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    # 获取视频的宽度
                    self.frame_width = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    # 获取视频的帧率
                    self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                    self.current_frame_no = 1
                    self._log_frame_activity('Decoded', self.current_frame_no, force=True)
                    self._log(f'Video metadata: {int(self.frame_width)}x{int(self.frame_height)}, {self.frame_count:.0f} frames, {self.fps:.2f} FPS')
                    # 调整视频帧大小，使播放器能够显示
                    resized_frame = self._img_resize(frame)
                    # resized_frame = cv2.resize(src=frame, dsize=(self.video_preview_width, self.video_preview_height))
                    # 显示视频帧
                    self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())
                    self._log_frame_activity('Drawn', self.current_frame_no, force=True)
                    # 更新视频进度条滑块range
                    self.window['-SLIDER-'].update(range=(1, self.frame_count))
                    self.window['-SLIDER-'].update(1)
                    # 预设字幕区域位置
                    y_p, h_p, x_p, w_p = self.parse_subtitle_config()
                    y = self.frame_height * y_p
                    h = self.frame_height * h_p
                    x = self.frame_width * x_p
                    w = self.frame_width * w_p
                    # 更新视频字幕位置滑块range
                    # 更新Y-SLIDER范围
                    self.window['-Y-SLIDER-'].update(range=(0, self.frame_height), disabled=False)
                    # 更新Y-SLIDER默认值
                    self.window['-Y-SLIDER-'].update(y)
                    # 更新X-SLIDER范围
                    self.window['-X-SLIDER-'].update(range=(0, self.frame_width), disabled=False)
                    # 更新X-SLIDER默认值
                    self.window['-X-SLIDER-'].update(x)
                    # 更新Y-SLIDER-H范围
                    self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height - y))
                    # 更新Y-SLIDER-H默认值
                    self.window['-Y-SLIDER-H-'].update(h)
                    # 更新X-SLIDER-W范围
                    self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width - x))
                    # 更新X-SLIDER-W默认值
                    self.window['-X-SLIDER-W-'].update(w)
                    self.preview_text = 'Detected subtitle preview: --'
                    self._update_preview(frame, (y, h, x, w))
                else:
                    self._log('Failed to decode first frame. Check codec/hardware decoding settings.', 'ERROR')

    def _language_mode_event_handler(self, event):
        if event != '-LANGUAGE-MODE-':
            return
        if 'OK' == LanguageModeGUI(self).run():
            # 重新加载config
            pass

    def _run_event_handler(self, event, values):
        """
        当点击运行按钮时：
        1) 禁止修改字幕滑块区域
        2) 禁止再次点击【运行】和【打开】按钮
        3) 设定字幕区域位置
        """
        if event == '-RUN-':
            if self.video_cap is None:
                print(self.interface_config['SubtitleExtractorGUI']['OpenVideoFirst'])
            else:
                # 1) 禁止修改字幕滑块区域
                self.window['-Y-SLIDER-'].update(disabled=True)
                self.window['-X-SLIDER-'].update(disabled=True)
                self.window['-Y-SLIDER-H-'].update(disabled=True)
                self.window['-X-SLIDER-W-'].update(disabled=True)
                # 2) 禁止再次点击【运行】、【打开】和【识别语言】按钮
                self.window['-RUN-'].update(disabled=True)
                self.window['-FILE-'].update(disabled=True)
                self.window['-FILE_BTN-'].update(disabled=True)
                self.window['-LANGUAGE-MODE-'].update(disabled=True)
                # 3) 设定字幕区域位置
                self.xmin = int(values['-X-SLIDER-'])
                self.xmax = int(values['-X-SLIDER-'] + values['-X-SLIDER-W-'])
                self.ymin = int(values['-Y-SLIDER-'])
                self.ymax = int(values['-Y-SLIDER-'] + values['-Y-SLIDER-H-'])
                if self.ymax > self.frame_height:
                    self.ymax = self.frame_height
                if self.xmax > self.frame_width:
                    self.xmax = self.frame_width
                self._log(f"{self.interface_config['SubtitleExtractorGUI']['SubtitleArea']}：({self.ymin},{self.ymax},{self.xmin},{self.xmax})")
                subtitle_area = (self.ymin, self.ymax, self.xmin, self.xmax)
                y_p = self.ymin / self.frame_height
                h_p = (self.ymax - self.ymin) / self.frame_height
                x_p = self.xmin / self.frame_width
                w_p = (self.xmax - self.xmin) / self.frame_width
                self.set_subtitle_config(y_p, h_p, x_p, w_p)

                def task():
                    while self.video_paths:
                        video_path = self.video_paths.pop()
                        self._log(f'Processing: {video_path}')
                        self.se = backend.main.SubtitleExtractor(video_path, subtitle_area)
                        self.se.run()
                Thread(target=task, daemon=True).start()
                self.video_cap.release()
                self.video_cap = None
                self.is_playing = False

    def _slide_event_handler(self, event, values):
        """
        当滑动视频进度条/滑动字幕选择区域滑块时：
        1) 判断视频是否存在，如果存在则显示对应的视频帧
        2) 绘制rectangle
        """
        if event == '-SLIDER-' or event == '-Y-SLIDER-' or event == '-Y-SLIDER-H-' or event == '-X-SLIDER-' or event \
                == '-X-SLIDER-W-':
            if self.video_cap is not None and self.video_cap.isOpened():
                frame_no = int(values['-SLIDER-'])
                ret, frame = self._read_frame_safely(frame_no=frame_no)
                if ret:
                    self.current_frame_no = frame_no
                    self._log_frame_activity('Decoded', frame_no)
                    self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height-values['-Y-SLIDER-']))
                    self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width-values['-X-SLIDER-']))
                    # 画字幕框
                    y = int(values['-Y-SLIDER-'])
                    h = int(values['-Y-SLIDER-H-'])
                    x = int(values['-X-SLIDER-'])
                    w = int(values['-X-SLIDER-W-'])
                    self._update_preview(frame, (y, h, x, w))
                else:
                    self._log(f'Frame decode failed at frame {frame_no}', 'ERROR')

    def _playback_event_handler(self, event, values):
        if self.video_cap is None or not self.video_cap.isOpened():
            return
        if event == '-PLAY-':
            self.is_playing = True
            self._log(f'Video playback started at frame {self.current_frame_no}', 'INFO')
        elif event == '-PAUSE-':
            self.is_playing = False
            self._log(f'Video playback paused at frame {self.current_frame_no}', 'INFO')
        elif event == '-PREV-':
            self.is_playing = False
            new_frame = max(1, int(values['-SLIDER-']) - 1)
            self.window['-SLIDER-'].update(new_frame)
            self._slide_event_handler('-SLIDER-', {**values, '-SLIDER-': new_frame})
        elif event == '-NEXT-':
            self.is_playing = False
            new_frame = min(int(self.frame_count), int(values['-SLIDER-']) + 1)
            self.window['-SLIDER-'].update(new_frame)
            self._slide_event_handler('-SLIDER-', {**values, '-SLIDER-': new_frame})

        if self.is_playing:
            step = max(1, int(round((self.fps or 25) / 12)))
            frame_no = min(int(self.frame_count), int(values['-SLIDER-']) + step)
            self.window['-SLIDER-'].update(frame_no)
            self._slide_event_handler('-SLIDER-', {**values, '-SLIDER-': frame_no})
            if frame_no >= int(self.frame_count):
                self.is_playing = False

    def _update_preview(self, frame, y_h_x_w):
        y, h, x, w = y_h_x_w
        # 画字幕框
        overlay = frame.copy()
        cv2.rectangle(img=overlay, pt1=(int(x), int(y)), pt2=(int(x) + int(w), int(y) + int(h)),
                      color=(76, 175, 80), thickness=2)
        cv2.putText(overlay, 'Subtitle Detection Box', (int(x) + 6, max(20, int(y) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (224, 224, 224), 2, cv2.LINE_AA)
        draw = cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)
        # 调整视频帧大小，使播放器能够显示
        resized_frame = self._img_resize(draw)
        # 显示视频帧
        self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())
        self._log_frame_activity('Drawn', self.current_frame_no)


    def _img_resize(self, image):
        top, bottom, left, right = (0, 0, 0, 0)
        height, width = image.shape[0], image.shape[1]
        # 对长短不想等的图片，找到最长的一边
        longest_edge = height
        # 计算短边需要增加多少像素宽度使其与长边等长
        if width < longest_edge:
            dw = longest_edge - width
            left = dw // 2
            right = dw - left
        else:
            pass
        # 给图像增加边界
        constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return cv2.resize(constant, (self.video_preview_width, self.video_preview_height))

    def set_subtitle_config(self, y, h, x, w):
        # 写入配置文件
        with open(self.subtitle_config_file, mode='w', encoding='utf-8') as f:
            f.write('[AREA]\n')
            f.write(f'Y = {y}\n')
            f.write(f'H = {h}\n')
            f.write(f'X = {x}\n')
            f.write(f'W = {w}\n')

    def parse_subtitle_config(self):
        y_p, h_p, x_p, w_p = .78, .21, .05, .9
        # 如果配置文件不存在，则写入配置文件
        if not os.path.exists(self.subtitle_config_file):
            self.set_subtitle_config(y_p, h_p, x_p, w_p)
            return y_p, h_p, x_p, w_p
        else:
            try:
                config = configparser.ConfigParser()
                config.read(self.subtitle_config_file, encoding='utf-8')
                conf_y_p, conf_h_p, conf_x_p, conf_w_p = float(config['AREA']['Y']), float(config['AREA']['H']), float(config['AREA']['X']), float(config['AREA']['W'])
                return conf_y_p, conf_h_p, conf_x_p, conf_w_p
            except Exception:
                self.set_subtitle_config(y_p, h_p, x_p, w_p)
                return y_p, h_p, x_p, w_p


class LanguageModeGUI:
    def __init__(self, subtitle_extractor_gui):
        self.subtitle_extractor_gui = subtitle_extractor_gui
        self.icon = os.path.join(os.path.dirname(__file__), 'design', 'vse.ico')
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.ini')
        # 设置界面
        self.INTERFACE_DEF = '简体中文'
        if not os.path.exists(self.config_file):
            self.interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'interface',
                                               "ch.ini")
        self.interface_config = configparser.ConfigParser()
        # 设置语言
        self.INTERFACE_KEY_NAME_MAP = {
            '简体中文': 'ch',
            '繁體中文': 'chinese_cht',
            'English': 'en',
            '한국어': 'ko',
            '日本語': 'japan',
            'Tiếng Việt': 'vi',
            'Español': 'es'
        }
        # 设置语言
        self.LANGUAGE_DEF = 'ch'
        self.LANGUAGE_NAME_KEY_MAP = None
        self.LANGUAGE_KEY_NAME_MAP = None
        self.MODE_DEF = 'fast'
        self.MODE_NAME_KEY_MAP = None
        self.MODE_KEY_NAME_MAP = None
        # 语言选择布局
        self.layout = None
        # 语言选择窗口
        self.window = None

    def run(self):
        # 创建布局
        title = self._create_layout()
        # 创建窗口
        self.window = sg.Window(title=title, layout=self.layout, icon=self.icon)
        while True:
            # 循环读取事件
            event, values = self.window.read(timeout=10)
            # 处理【OK】事件
            self._ok_event_handler(event, values)
            # 处理【切换界面语言】事件
            self._interface_event_handler(event, values)
            # 如果关闭软件，退出
            if event == sg.WIN_CLOSED:
                if os.path.exists(self.config_file):
                    break
                else:
                    exit(0)
            if event == 'Cancel':
                if os.path.exists(self.config_file):
                    self.window.close()
                    break
                else:
                    exit(0)

    def _load_interface_text(self):
        self.interface_config.read(self.interface_file, encoding='utf-8')
        config_language_mode_gui = self.interface_config["LanguageModeGUI"]
        # 设置界面
        self.INTERFACE_DEF = config_language_mode_gui["InterfaceDefault"]

        self.LANGUAGE_DEF = config_language_mode_gui["LanguageCH"]
        self.LANGUAGE_NAME_KEY_MAP = {}
        for lang in backend.main.config.MULTI_LANG:
            self.LANGUAGE_NAME_KEY_MAP[config_language_mode_gui[f"Language{lang.upper()}"]] = lang
        self.LANGUAGE_NAME_KEY_MAP = dict(sorted(self.LANGUAGE_NAME_KEY_MAP.items(), key=lambda item: item[1]))
        self.LANGUAGE_KEY_NAME_MAP = {v: k for k, v in self.LANGUAGE_NAME_KEY_MAP.items()}
        self.MODE_DEF = config_language_mode_gui['ModeFast']
        self.MODE_NAME_KEY_MAP = {
            config_language_mode_gui['ModeAuto']: 'auto',
            config_language_mode_gui['ModeFast']: 'fast',
            config_language_mode_gui['ModeAccurate']: 'accurate',
        }
        self.MODE_KEY_NAME_MAP = {v: k for k, v in self.MODE_NAME_KEY_MAP.items()}

    def _create_layout(self):
        interface_def, language_def, mode_def = self.parse_config(self.config_file)
        # 加载界面文本
        self._load_interface_text()
        choose_language_text = self.interface_config["LanguageModeGUI"]["InterfaceLanguage"]
        choose_sub_lang_text = self.interface_config["LanguageModeGUI"]["SubtitleLanguage"]
        choose_mode_text = self.interface_config["LanguageModeGUI"]["Mode"]
        self.layout = [
            # 显示选择界面语言
            [sg.Text(choose_language_text),
             sg.DropDown(values=list(self.INTERFACE_KEY_NAME_MAP.keys()), size=(30, 20),
                         pad=(0, 20),
                         key='-INTERFACE-', readonly=True,
                         default_value=interface_def),
             sg.OK(key='-INTERFACE-OK-')],
            # 显示选择字幕语言
            [sg.Text(choose_sub_lang_text),
             sg.DropDown(values=list(self.LANGUAGE_NAME_KEY_MAP.keys()), size=(30, 20),
                         pad=(0, 20),
                         key='-LANGUAGE-', readonly=True, default_value=language_def)],
            # 显示识别模式
            [sg.Text(choose_mode_text),
             sg.DropDown(values=list(self.MODE_NAME_KEY_MAP.keys()), size=(30, 20), pad=(0, 20),
                         key='-MODE-', readonly=True, default_value=mode_def)],
            # 显示确认关闭按钮
            [sg.OK(), sg.Cancel()]
        ]
        return self.interface_config["LanguageModeGUI"]["Title"]

    def _ok_event_handler(self, event, values):
        if event == 'OK':
            # 设置模型语言配置
            interface = None
            language = None
            mode = None
            # 设置界面语言
            interface_str = values['-INTERFACE-']
            if interface_str in self.INTERFACE_KEY_NAME_MAP:
                interface = interface_str
            language_str = values['-LANGUAGE-']
            # 设置字幕语言
            print(self.interface_config["LanguageModeGUI"]["SubtitleLanguage"], language_str)
            if language_str in self.LANGUAGE_NAME_KEY_MAP:
                language = self.LANGUAGE_NAME_KEY_MAP[language_str]
            # 设置模型语言配置
            mode_str = values['-MODE-']
            print(self.interface_config["LanguageModeGUI"]["Mode"], mode_str)
            if mode_str in self.MODE_NAME_KEY_MAP:
                mode = self.MODE_NAME_KEY_MAP[mode_str]
            self.set_config(self.config_file, interface, language, mode)
            if self.subtitle_extractor_gui is not None:
                self.subtitle_extractor_gui.update_interface_text()
            self.window.close()

    def _interface_event_handler(self, event, values):
        if event == '-INTERFACE-OK-':
            self.interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'interface',
                                               f"{self.INTERFACE_KEY_NAME_MAP[values['-INTERFACE-']]}.ini")
            self.interface_config.read(self.interface_file, encoding='utf-8')
            config = configparser.ConfigParser()
            if os.path.exists(self.config_file):
                config.read(self.config_file, encoding='utf-8')
                self.set_config(self.config_file, values['-INTERFACE-'], config['DEFAULT']['Language'],
                                config['DEFAULT']['Mode'])
            self.window.close()
            title = self._create_layout()
            self.window = sg.Window(title=title, layout=self.layout, icon=self.icon)

    @staticmethod
    def set_config(config_file, interface, language_code, mode):
        # 写入配置文件
        with open(config_file, mode='w', encoding='utf-8') as f:
            f.write('[DEFAULT]\n')
            f.write(f'Interface = {interface}\n')
            f.write(f'Language = {language_code}\n')
            f.write(f'Mode = {mode}\n')

    def parse_config(self, config_file):
        if not os.path.exists(config_file):
            self.interface_config.read(self.interface_file, encoding='utf-8')
            interface_def = self.interface_config['LanguageModeGUI']['InterfaceDefault']
            language_def = self.interface_config['LanguageModeGUI']['InterfaceDefault']
            mode_def = self.interface_config['LanguageModeGUI']['ModeFast']
            return interface_def, language_def, mode_def
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        interface = config['DEFAULT']['Interface']
        language = config['DEFAULT']['Language']
        mode = config['DEFAULT']['Mode']
        self.interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'interface',
                                           f"{self.INTERFACE_KEY_NAME_MAP[interface]}.ini")
        self._load_interface_text()
        interface_def = interface if interface in self.INTERFACE_KEY_NAME_MAP else \
            self.INTERFACE_DEF
        language_def = self.LANGUAGE_KEY_NAME_MAP[language] if language in self.LANGUAGE_KEY_NAME_MAP else \
            self.LANGUAGE_DEF
        mode_def = self.MODE_KEY_NAME_MAP[mode] if mode in self.MODE_KEY_NAME_MAP else self.MODE_DEF
        return interface_def, language_def, mode_def


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method("spawn")
        # 运行图形化界面
        subtitleExtractorGUI = SubtitleExtractorGUI()
        subtitleExtractorGUI.run()
    except Exception as e:
        err_msg = str(e).lower()
        if "couldn't connect to display" in err_msg or "no display name and no $display" in err_msg:
            print("GUI cannot start: no DISPLAY is available in the current environment.")
            print("Use CLI mode instead: python backend/main.py")
            print("Or run gui.py on a local desktop with X11/Wayland display support.")
            raise SystemExit(1)
        print(f'[{type(e)}] {e}')
        import traceback
        traceback.print_exc()
        msg = traceback.format_exc()
        err_log_path = os.path.join(os.path.expanduser('~'), 'VSE-Error-Message.log')
        with open(err_log_path, 'w', encoding='utf-8') as f:
            f.writelines(msg)
        import platform
        if platform.system() == 'Windows':
            os.system('pause')
        else:
            if os.isatty(0):
                input()
