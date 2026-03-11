# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao
@Time    : 2021/3/24 9:28 AM
@FileName: main.py
@desc: Main program entry file
"""
import os
import random
import shutil
from collections import Counter, namedtuple
import unicodedata
from threading import Thread
from pathlib import Path
import cv2
from Levenshtein import ratio
from PIL import Image
from numpy import average, dot, linalg
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(__file__))
import importlib
import config
from tools import reformat

from backend.tools.ocr import OcrRecogniser, get_coordinates
from backend.tools import subtitle_ocr
from backend.tools.fast_pipeline import FastSubtitlePipeline
import threading
import platform
import multiprocessing
import time
import pysrt


class SubtitleDetect:
    """
    Text box detection class for detecting text boxes in video frames
    """

    def __init__(self):
        from paddleocr.tools.infer import utility
        from paddleocr.tools.infer.predict_det import TextDetector
        # Get argument object
        importlib.reload(config)
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = config.DET_MODEL_PATH
        self.text_detector = TextDetector(args)

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse


class SubtitleExtractor:
    """
    Video subtitle extraction class
    """

    def __init__(self, vd_path, sub_area=None):
        importlib.reload(config)
        # Thread lock
        self.lock = threading.RLock()
        # User-specified subtitle area position
        self.sub_area = sub_area
        # Create subtitle detection object
        self.sub_detector = SubtitleDetect()
        # Video path
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # Get video name from video path
        self.vd_name = Path(self.video_path).stem
        # Temporary storage directory
        self.temp_output_dir = os.path.join(os.path.dirname(config.BASE_DIR), 'output', str(self.vd_name))
        # Total number of video frames
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # Video frame rate
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # Video dimensions
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # Default subtitle area when user has not specified one
        self.default_subtitle_area = config.DEFAULT_SUBTITLE_AREA
        # Directory for storing extracted video frames
        self.frame_output_dir = os.path.join(self.temp_output_dir, 'frames')
        # Directory for storing extracted subtitle files
        self.subtitle_output_dir = os.path.join(self.temp_output_dir, 'subtitle')
        # Create directories if they don't exist
        if not os.path.exists(self.frame_output_dir):
            os.makedirs(self.frame_output_dir)
        if not os.path.exists(self.subtitle_output_dir):
            os.makedirs(self.subtitle_output_dir)
        # Whether to use VSF to extract subtitle frames
        self.use_vsf = False
        # VSF subtitle output path
        self.vsf_subtitle = os.path.join(self.subtitle_output_dir, 'raw_vsf.srt')
        # Path for storing raw extracted subtitle text
        self.raw_subtitle_path = os.path.join(self.subtitle_output_dir, 'raw.txt')
        # Custom OCR object
        self.ocr = None
        # Print recognition language and mode
        print(f"{config.interface_config['Main']['RecSubLang']}：{config.REC_CHAR_TYPE}")
        print(f"{config.interface_config['Main']['RecMode']}：{config.MODE_TYPE}")
        # Print GPU acceleration info if GPU is enabled
        if config.USE_GPU:
            print(config.interface_config['Main']['GPUSpeedUp'])
        # Total processing progress
        self.progress_total = 0
        # Video frame extraction progress
        self.progress_frame_extract = 0
        # OCR recognition progress
        self.progress_ocr = 0
        # Whether finished
        self.isFinished = False
        # Subtitle OCR task queue
        self.subtitle_ocr_task_queue = None
        # Subtitle OCR progress queue
        self.subtitle_ocr_progress_queue = None
        # VSF running status
        self.vsf_running = False
        # Number of queued frame tasks (for debugging frame extraction)
        self.frame_task_count = 0

    def run_fast(self):
        """
        Run the HIGH-PERFORMANCE subtitle extraction pipeline.
        Uses scene change detection, frame differencing, text caching,
        and smart sampling to achieve 50-100x speedup.
        """
        start_time = time.time()
        self.lock.acquire()
        self.update_progress(ocr=0, frame_extract=0)
        print(f"{config.interface_config['Main']['FrameCount']}：{self.frame_count}"
              f"，{config.interface_config['Main']['FrameRate']}：{self.fps}")
        print(f'{os.path.basename(os.path.dirname(config.DET_MODEL_PATH))}-{os.path.basename(config.DET_MODEL_PATH)}')
        print(f'{os.path.basename(os.path.dirname(config.REC_MODEL_PATH))}-{os.path.basename(config.REC_MODEL_PATH)}')
        print(config.interface_config['Main']['StartProcessFrame'])

        srt_path = os.path.splitext(self.video_path)[0] + '.srt'
        txt_path = os.path.splitext(self.video_path)[0] + '.txt' if config.GENERATE_TXT else None

        def progress_cb(pct):
            self.update_progress(ocr=pct, frame_extract=pct)

        pipeline = FastSubtitlePipeline(
            video_path=self.video_path,
            sub_area=self.sub_area,
            sample_fps=config.EXTRACT_FREQUENCY,
            ocr_batch_size=max(config.REC_BATCH_NUM, 8),
            drop_score=config.DROP_SCORE,
            similarity_threshold=config.THRESHOLD_TEXT_SIMILARITY,
            progress_callback=progress_cb,
        )

        stats = pipeline.run(srt_path, txt_path)

        if config.WORD_SEGMENTATION:
            reformat.execute(srt_path, config.REC_CHAR_TYPE)

        print(config.interface_config['Main']['FinishGenerateSub'], f"{round(time.time() - start_time, 2)}s")
        self.update_progress(ocr=100, frame_extract=100)
        self.isFinished = True
        self.lock.release()

    def run(self):
        """
        Run the complete video subtitle extraction pipeline (original method).
        Consider using run_fast() for 50-100x better performance.
        """
        # Record start time
        start_time = time.time()
        self.lock.acquire()
        # Reset progress bar
        self.update_progress(ocr=0, frame_extract=0)
        # Print frame count and frame rate
        print(f"{config.interface_config['Main']['FrameCount']}：{self.frame_count}"
              f"，{config.interface_config['Main']['FrameRate']}：{self.fps}")
        # Print loaded model info
        print(f'{os.path.basename(os.path.dirname(config.DET_MODEL_PATH))}-{os.path.basename(config.DET_MODEL_PATH)}')
        print(f'{os.path.basename(os.path.dirname(config.REC_MODEL_PATH))}-{os.path.basename(config.REC_MODEL_PATH)}')
        # Print frame extraction start message
        print(config.interface_config['Main']['StartProcessFrame'])
        # Create a subtitle OCR recognition process
        subtitle_ocr_process = self.start_subtitle_ocr_async()
        if self.sub_area is not None:
            if platform.system() in ['Windows', 'Linux']:
                # Only use this method when GPU is enabled and mode is 'accurate':
                if config.USE_GPU and config.MODE_TYPE == 'accurate':
                    self.extract_frame_by_det()
                else:
                    self.extract_frame_by_vsf()
            else:
                self.extract_frame_by_fps()
        else:
            self.extract_frame_by_fps()

        # Add OCR task end marker to the subtitle OCR task queue
        # Task format: (total_frame_count, current_frame_no, dt_box, rec_res, current_frame_time, subtitle_area)
        self.subtitle_ocr_task_queue.put((self.frame_count, -1, None, None, None, None))
        # Wait for subprocess to finish
        subtitle_ocr_process.join()
        # Print completion message
        print(config.interface_config['Main']['FinishProcessFrame'])
        print(config.interface_config['Main']['FinishFindSub'])

        if self.sub_area is None:
            print(config.interface_config['Main']['StartDetectWaterMark'])
            # Ask user if video has watermark areas
            user_input = input(config.interface_config['Main']['checkWaterMark']).strip()
            if user_input == 'y':
                self.filter_watermark()
                print(config.interface_config['Main']['FinishDetectWaterMark'])
            else:
                print('-----------------------------')

        if self.sub_area is None:
            print(config.interface_config['Main']['StartDeleteNonSub'])
            self.filter_scene_text()
            print(config.interface_config['Main']['FinishDeleteNonSub'])

        # Print subtitle generation start message
        print(config.interface_config['Main']['StartGenerateSub'])
        # Check whether VSF was used for subtitle extraction
        if self.use_vsf:
            # If VSF was used, use VSF subtitle generation method
            self.generate_subtitle_file_vsf()
        else:
            # If VSF was not used, use standard subtitle generation method
            self.generate_subtitle_file()
        if config.WORD_SEGMENTATION:
            reformat.execute(os.path.join(os.path.splitext(self.video_path)[0] + '.srt'), config.REC_CHAR_TYPE)
        print(config.interface_config['Main']['FinishGenerateSub'], f"{round(time.time() - start_time, 2)}s")
        self.update_progress(ocr=100, frame_extract=100)
        self.isFinished = True
        # Delete cache files
        self.empty_cache()
        self.lock.release()
        if config.GENERATE_TXT:
            self.srt2txt(os.path.join(os.path.splitext(self.video_path)[0] + '.srt'))

    def extract_frame_by_fps(self):
        """
        Extract video frames at fixed intervals based on frame rate. May miss subtitles but is faster.
        Adds extracted frames to the OCR task queue.
        """
        # Delete cache
        self.__delete_frame_cache()
        # Current video frame number
        current_frame_no = 0
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # If reading video frame fails (reached last frame)
            if not ret:
                break
            # Successfully read video frame
            else:
                current_frame_no += 1
                # subtitle_ocr_task_queue: (total_frame_count, current_frame_no, dt_box, rec_res, current_frame_time, subtitle_area)
                task = (self.frame_count, current_frame_no, None, None, None, self.default_subtitle_area)
                self.subtitle_ocr_task_queue.put(task)
                self.frame_task_count += 1
                # Skip remaining frames
                for i in range(int(self.fps // config.EXTRACT_FREQUENCY) - 1):
                    ret, _ = self.video_cap.read()
                    if ret:
                        current_frame_no += 1
                        # Update progress bar
                        self.update_progress(frame_extract=(current_frame_no / self.frame_count) * 100)

        self.video_cap.release()
        print(f"[FrameExtract][FPS] queued tasks: {self.frame_task_count}")

    def extract_frame_by_det(self):
        """
        Extract subtitle frames by detecting subtitle area positions
        """
        # Delete cache
        self.__delete_frame_cache()

        # Current video frame number
        current_frame_no = 0
        frame_lru_list = []
        frame_lru_list_max_size = 2
        ocr_args_list = []
        compare_ocr_result_cache = {}
        tbar = tqdm(total=int(self.frame_count), unit='f', position=0, file=sys.__stdout__)
        first_flag = True
        is_finding_start_frame_no = False
        is_finding_end_frame_no = False
        start_frame_no = 0
        start_end_frame_no = []
        start_frame = None
        if self.ocr is None:
            self.ocr = OcrRecogniser()
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # If reading video frame fails (reached last frame)
            if not ret:
                break
            # Successfully read video frame
            current_frame_no += 1
            tbar.update(1)
            dt_boxes, elapse = self.sub_detector.detect_subtitle(frame)
            has_subtitle = False
            if self.sub_area is not None:
                s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                coordinate_list = get_coordinates(dt_boxes.tolist())
                if coordinate_list:
                    for coordinate in coordinate_list:
                        xmin, xmax, ymin, ymax = coordinate
                        if (s_xmin <= xmin and xmax <= s_xmax
                                and s_ymin <= ymin
                                and ymax <= s_ymax):
                            has_subtitle = True
                            # When subtitle detected, if list is empty, this is the subtitle start
                            if first_flag:
                                is_finding_start_frame_no = True
                                first_flag = False
                            break
            else:
                has_subtitle = len(dt_boxes) > 0
            # Detect start and end frame numbers of subtitle frames
            if has_subtitle:
                # Determine if this is the start or end of subtitle
                if is_finding_start_frame_no:
                    start_frame_no = current_frame_no
                    dt_box, rec_res = self.ocr.predict(frame)
                    area_text1 = "".join(self.__get_area_text((dt_box, rec_res)))
                    if start_frame_no not in compare_ocr_result_cache.keys():
                        compare_ocr_result_cache[current_frame_no] = {'text': area_text1, 'dt_box': dt_box, 'rec_res': rec_res}
                        frame_lru_list.append((frame, current_frame_no))
                        ocr_args_list.append((self.frame_count, current_frame_no))
                        # Cache start frame
                        start_frame = frame
                    # Start finding the end
                    is_finding_start_frame_no = False
                    is_finding_end_frame_no = True
                # Check if this is the last frame
                if is_finding_end_frame_no and current_frame_no == self.frame_count:
                    is_finding_end_frame_no = False
                    is_finding_start_frame_no = False
                    end_frame_no = current_frame_no
                    frame_lru_list.append((frame, current_frame_no))
                    ocr_args_list.append((self.frame_count, current_frame_no))
                    start_end_frame_no.append((start_frame_no, end_frame_no))
                # While searching for the end frame
                if is_finding_end_frame_no:
                    # Check if OCR content matches the start frame; if not, the end is the previous frame
                    if not self._compare_ocr_result(compare_ocr_result_cache, None, start_frame_no, frame, current_frame_no):
                        is_finding_end_frame_no = False
                        is_finding_start_frame_no = True
                        end_frame_no = current_frame_no - 1
                        frame_lru_list.append((start_frame, end_frame_no))
                        ocr_args_list.append((self.frame_count, end_frame_no))
                        start_end_frame_no.append((start_frame_no, end_frame_no))

            else:
                # If no subtitle after start, the end is the previous frame
                if is_finding_end_frame_no:
                    end_frame_no = current_frame_no - 1
                    is_finding_end_frame_no = False
                    is_finding_start_frame_no = True
                    frame_lru_list.append((start_frame, end_frame_no))
                    ocr_args_list.append((self.frame_count, end_frame_no))
                    start_end_frame_no.append((start_frame_no, end_frame_no))

            while len(frame_lru_list) > frame_lru_list_max_size:
                frame_lru_list.pop(0)

            # if len(start_end_frame_no) > 0:
                # print(start_end_frame_no)

            while len(ocr_args_list) > 1:
                total_frame_count, ocr_info_frame_no = ocr_args_list.pop(0)
                if current_frame_no in compare_ocr_result_cache:
                    predict_result = compare_ocr_result_cache[current_frame_no]
                    dt_box, rec_res = predict_result['dt_box'], predict_result['rec_res']
                else:
                    dt_box, rec_res = None, None
                # subtitle_ocr_task_queue: (total_frame_count, current_frame_no, dt_box, rec_res, current_frame_time, subtitle_area)
                task = (total_frame_count, ocr_info_frame_no, dt_box, rec_res, None, self.default_subtitle_area)
                # Add task
                self.subtitle_ocr_task_queue.put(task)
                self.frame_task_count += 1
                self.update_progress(frame_extract=(current_frame_no / self.frame_count) * 100)

        while len(ocr_args_list) > 0:
            total_frame_count, ocr_info_frame_no = ocr_args_list.pop(0)
            if current_frame_no in compare_ocr_result_cache:
                predict_result = compare_ocr_result_cache[current_frame_no]
                dt_box, rec_res = predict_result['dt_box'], predict_result['rec_res']
            else:
                dt_box, rec_res = None, None
            task = (total_frame_count, ocr_info_frame_no, dt_box, rec_res, None, self.default_subtitle_area)
            # Add task
            self.subtitle_ocr_task_queue.put(task)
            self.frame_task_count += 1
        self.video_cap.release()
        print(f"[FrameExtract][DET] queued tasks: {self.frame_task_count}")

    def extract_frame_by_vsf(self):
        """
       Extract subtitle frames by calling VideoSubFinder
       """
        self.use_vsf = True

        def ms_to_frame_no(total_ms):
            # total_ms is in milliseconds, convert to frame number by dividing by 1000 then multiplying by fps
            return int(round((total_ms / 1000.0) * self.fps))

        def count_process():
            duration_ms = (self.frame_count / self.fps) * 1000
            last_total_ms = 0
            processed_image = set()
            rgb_images_path = os.path.join(self.temp_output_dir, 'RGBImages')
            while self.vsf_running and not self.isFinished:
                # If rgb_images_path doesn't exist yet, VSF is still processing
                if not os.path.exists(rgb_images_path):
                    # Keep waiting
                    continue
                try:
                    # Sort list by filename
                    rgb_images = sorted(os.listdir(rgb_images_path))
                    for rgb_image in rgb_images:
                        # Skip if current image has already been processed
                        if rgb_image in processed_image:
                            continue
                        processed_image.add(rgb_image)
                        # Read timestamp from VSF-generated filename
                        h, m, s, ms = rgb_image.split('__')[0].split('_')
                        total_ms = int(ms) + int(s) * 1000 + int(m) * 60 * 1000 + int(h) * 60 * 60 * 1000
                        if total_ms > last_total_ms:
                            frame_no = ms_to_frame_no(total_ms)
                            task = (self.frame_count, frame_no, None, None, total_ms, self.default_subtitle_area)
                            self.subtitle_ocr_task_queue.put(task)
                            self.frame_task_count += 1
                        last_total_ms = total_ms
                        if total_ms / duration_ms >= 1:
                            self.update_progress(frame_extract=100)
                            return
                        else:
                            self.update_progress(frame_extract=(total_ms / duration_ms) * 100)
                # Files were cleaned up
                except FileNotFoundError:
                    return

        def vsf_output(out, ):
            duration_ms = (self.frame_count / self.fps) * 1000
            last_total_ms = 0
            for line in iter(out.readline, b''):
                line = line.decode("utf-8")
                # print('line', line, type(line), line.startswith('Frame: '))
                if line.startswith('Frame: '):
                    line = line.replace("\n", "")
                    line = line.replace("Frame: ", "")
                    h, m, s, ms = line.split('__')[0].split('_')
                    total_ms = int(ms) + int(s) * 1000 + int(m) * 60 * 1000 + int(h) * 60 * 60 * 1000
                    if total_ms > last_total_ms:
                        frame_no = ms_to_frame_no(total_ms)
                        task = (self.frame_count, frame_no, None, None, total_ms, self.default_subtitle_area)
                        self.subtitle_ocr_task_queue.put(task)
                        self.frame_task_count += 1
                    last_total_ms = total_ms
                    if total_ms / duration_ms >= 1:
                        self.update_progress(frame_extract=100)
                        return
                    else:
                        self.update_progress(frame_extract=(total_ms / duration_ms) * 100)
                else:
                    print(line.strip())
            out.close()

        # Delete cache
        self.__delete_frame_cache()
        # Define VideoSubFinder path
        if platform.system() == 'Windows':
            path_vsf = os.path.join(config.BASE_DIR, 'subfinder', 'windows', 'VideoSubFinderWXW.exe')
        else:
            path_vsf = os.path.join(config.BASE_DIR, 'subfinder', 'linux', 'VideoSubFinderCli.run')
            os.chmod(path_vsf, 0o775)
        # Percentage of upper part of image, range [0-1]
        top_end = 1 - self.sub_area[0] / self.frame_height
        # bottom_end: percentage of lower part of image, range [0-1]
        bottom_end = 1 - self.sub_area[1] / self.frame_height
        # left_end: percentage of left part of image, range [0-1]
        left_end = self.sub_area[2] / self.frame_width
        # right_end: percentage of right part of image, range [0-1]
        right_end = self.sub_area[3] / self.frame_width
        if config.USE_GPU and len(config.ONNX_PROVIDERS) > 0:
            cpu_count = multiprocessing.cpu_count()
        else:
            cpu_count = max(int(multiprocessing.cpu_count() * 2 / 3), 1)
            if cpu_count < 4:
                cpu_count = max(multiprocessing.cpu_count() - 1, 1)
        if platform.system() == 'Windows':
            # Define execution command
            cmd = f"{path_vsf} --use_cuda -c -r -i \"{self.video_path}\" -o \"{self.temp_output_dir}\" -ces \"{self.vsf_subtitle}\" "
            cmd += f"-te {top_end} -be {bottom_end} -le {left_end} -re {right_end} -nthr {cpu_count} -nocrthr {cpu_count}"
            self.vsf_running = True
            # Calculate progress
            Thread(target=count_process, daemon=True).start()
            import subprocess
            subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.vsf_running = False
        else:
            # Define execution command
            cmd = f"{path_vsf} -c -r -i \"{self.video_path}\" -o \"{self.temp_output_dir}\" -ces \"{self.vsf_subtitle}\" "
            if config.USE_GPU:
                cmd += "--use_cuda "
            cmd += f"-te {top_end} -be {bottom_end} -le {left_end} -re {right_end} -nthr {cpu_count} -dsi"
            self.vsf_running = True
            import subprocess
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1,
                                 close_fds='posix' in sys.builtin_module_names, shell=True)
            Thread(target=vsf_output, daemon=True, args=(p.stderr,)).start()
            p.wait()
            if p.returncode != 0:
                print(f"[VSF] failed with exit code {p.returncode}")
                if p.returncode == 139:
                    print("[VSF] segmentation fault detected, fallback to FPS extraction")
                if self.frame_task_count == 0:
                    self.use_vsf = False
                    self.extract_frame_by_fps()
                    return
            if self.frame_task_count == 0:
                print("[VSF] no frame tasks produced, fallback to FPS extraction")
                self.use_vsf = False
                self.extract_frame_by_fps()
                return
            print(f"[FrameExtract][VSF] queued tasks: {self.frame_task_count}")
            self.vsf_running = False

    def filter_watermark(self):
        """
        Remove text from watermark areas in raw subtitle text
        """
        # Get potential watermark areas
        watermark_areas = self._detect_watermark_area()

        # Randomly select a frame, mark all watermark areas for user to visually confirm
        cap = cv2.VideoCapture(self.video_path)
        ret, sample_frame = False, None
        for i in range(10):
            frame_no = random.randint(int(self.frame_count * 0.1), int(self.frame_count * 0.9))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, sample_frame = cap.read()
            if ret:
                break
        cap.release()

        if not ret:
            print("Error in filter_watermark: reading frame from video")
            return

        # Number the potential watermark areas
        area_num = ['E', 'D', 'C', 'B', 'A']

        for watermark_area in watermark_areas:
            ymin = min(watermark_area[0][2], watermark_area[0][3])
            ymax = max(watermark_area[0][3], watermark_area[0][2])
            xmin = min(watermark_area[0][0], watermark_area[0][1])
            xmax = max(watermark_area[0][1], watermark_area[0][0])
            cover = sample_frame[ymin:ymax, xmin:xmax]
            cover = cv2.blur(cover, (10, 10))
            cv2.rectangle(cover, pt1=(0, cover.shape[0]), pt2=(cover.shape[1], 0), color=(0, 0, 255), thickness=3)
            sample_frame[ymin:ymax, xmin:xmax] = cover
            position = ((xmin + xmax) // 2, ymax)

            cv2.putText(sample_frame, text=area_num.pop(), org=position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        sample_frame_file_path = os.path.join(os.path.dirname(self.frame_output_dir), 'watermark_area.jpg')
        cv2.imwrite(sample_frame_file_path, sample_frame)
        print(f"{config.interface_config['Main']['WatchPicture']}: {sample_frame_file_path}")

        area_num = ['E', 'D', 'C', 'B', 'A']
        for watermark_area in watermark_areas:
            user_input = input(f"{area_num.pop()}{str(watermark_area)} "
                               f"{config.interface_config['Main']['QuestionDelete']}").strip()
            if user_input == 'y' or user_input == '\n':
                with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
                    content = f.readlines()
                    f.seek(0)
                    for i in content:
                        if i.find(str(watermark_area[0])) == -1:
                            f.write(i)
                    f.truncate()
                print(config.interface_config['Main']['FinishDelete'])
        print(config.interface_config['Main']['FinishWaterMarkFilter'])
        # Delete cache
        if os.path.exists(sample_frame_file_path):
            os.remove(sample_frame_file_path)

    def filter_scene_text(self):
        """
        Filter out scene text, keeping only the subtitle area
        """
        # Get potential subtitle area
        subtitle_area = self._detect_subtitle_area()[0][0]

        # Randomly select a frame, mark the subtitle area for user to visually confirm
        cap = cv2.VideoCapture(self.video_path)
        ret, sample_frame = False, None
        for i in range(10):
            frame_no = random.randint(int(self.frame_count * 0.1), int(self.frame_count * 0.9))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, sample_frame = cap.read()
            if ret:
                break
        cap.release()

        if not ret:
            print("Error in filter_scene_text: reading frame from video")
            return

        # Expand y-range of subtitle area to accommodate double-line subtitles based on tolerance
        ymin = abs(subtitle_area[0] - config.SUBTITLE_AREA_DEVIATION_PIXEL)
        ymax = subtitle_area[1] + config.SUBTITLE_AREA_DEVIATION_PIXEL
        # Draw the subtitle area rectangle
        cv2.rectangle(sample_frame, pt1=(0, ymin), pt2=(sample_frame.shape[1], ymax), color=(0, 0, 255), thickness=3)
        sample_frame_file_path = os.path.join(os.path.dirname(self.frame_output_dir), 'subtitle_area.jpg')
        cv2.imwrite(sample_frame_file_path, sample_frame)
        print(f"{config.interface_config['Main']['CheckSubArea']} {sample_frame_file_path}")

        user_input = input(f"{(ymin, ymax)} {config.interface_config['Main']['DeleteNoSubArea']}").strip()
        if user_input == 'y' or user_input == '\n':
            with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
                content = f.readlines()
                f.seek(0)
                for i in content:
                    i_ymin = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[2])
                    i_ymax = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[3])
                    if ymin <= i_ymin and i_ymax <= ymax:
                        f.write(i)
                f.truncate()
            print(config.interface_config['Main']['FinishDeleteNoSubArea'])
        # Delete cache
        if os.path.exists(sample_frame_file_path):
            os.remove(sample_frame_file_path)

    def generate_subtitle_file(self):
        """
        Generate SRT format subtitle file
        """
        if not self.use_vsf:
            subtitle_content = self._remove_duplicate_subtitle()
            srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
            # Save subtitle lines with duration less than 1 second for post-processing
            post_process_subtitle = []
            with open(srt_filename, mode='w', encoding='utf-8') as f:
                for index, content in enumerate(subtitle_content):
                    line_code = index + 1
                    frame_start = self._frame_to_timecode(int(content[0]))
                    # Compare start and end frame numbers; if subtitle duration is less than 1 second, set display time to 1s
                    if abs(int(content[1]) - int(content[0])) < self.fps:
                        frame_end = self._frame_to_timecode(int(int(content[0]) + self.fps))
                        post_process_subtitle.append(line_code)
                    else:
                        frame_end = self._frame_to_timecode(int(content[1]))
                    frame_content = content[2]
                    subtitle_line = f'{line_code}\n{frame_start} --> {frame_end}\n{frame_content}\n'
                    f.write(subtitle_line)
            print(f"[NO-VSF]{config.interface_config['Main']['SubLocation']} {srt_filename}")
            # Return subtitle lines with duration less than 1s
            return post_process_subtitle

    def generate_subtitle_file_vsf(self):
        if not self.use_vsf:
            return
        subs = pysrt.open(self.vsf_subtitle)
        sub_no_map = {}
        for sub in subs:
            sub.start.no = self._timestamp_to_frameno(sub.start.ordinal)
            sub_no_map[sub.start.no] = sub

        subtitle_content = self._remove_duplicate_subtitle()
        subtitle_content_start_map = {int(a[0]): a for a in subtitle_content}
        final_subtitles = []
        for sub in subs:
            found = sub.start.no in subtitle_content_start_map
            if found:
                subtitle_content_line = subtitle_content_start_map[sub.start.no]
                sub.text = subtitle_content_line[2]
                end_no = int(subtitle_content_line[1])
                sub.end = sub_no_map[end_no].end if end_no in sub_no_map else sub.end
                sub.index = len(final_subtitles) + 1
                final_subtitles.append(sub)

            if not found and not config.DELETE_EMPTY_TIMESTAMP:
                # Keep timeline
                sub.text = ""
                sub.index = len(final_subtitles) + 1
                final_subtitles.append(sub)
                continue

        srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
        pysrt.SubRipFile(final_subtitles).save(srt_filename, encoding='utf-8')
        print(f"[VSF]{config.interface_config['Main']['SubLocation']} {srt_filename}")

    def _detect_watermark_area(self):
        """
        Detect watermark areas based on coordinate information in the raw txt file.
        Assumption: Watermark areas (e.g. station logos) have fixed coordinates in both
        horizontal and vertical directions, i.e. (xmin, xmax, ymin, ymax) are relatively constant.
        Statistically identifies text areas with consistently fixed coordinates.
        :return: Most likely watermark areas
        """
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # Open txt file with utf-8 encoding
        line = f.readline()  # Read file line by line
        # Coordinate list
        coordinates_list = []
        # Frame number list
        frame_no_list = []
        # Content list
        content_list = []
        while line:
            frame_no = line.split('\t')[0]
            text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
            content = line.split('\t')[2]
            frame_no_list.append(frame_no)
            coordinates_list.append((int(text_position[0]),
                                     int(text_position[1]),
                                     int(text_position[2]),
                                     int(text_position[3])))
            content_list.append(content)
            line = f.readline()
        f.close()
        # Unify similar values in coordinate list
        coordinates_list = self._unite_coordinates(coordinates_list)

        # Update coordinates in original txt file with unified values
        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, coordinate, content in zip(frame_no_list, coordinates_list, content_list):
                f.write(f'{frame_no}\t{coordinate}\t{content}')

        if len(Counter(coordinates_list).most_common()) > config.WATERMARK_AREA_NUM:
            # Read config, return list of coordinates likely to be watermark areas
            return Counter(coordinates_list).most_common(config.WATERMARK_AREA_NUM)
        else:
            # Return as many as available
            return Counter(coordinates_list).most_common()

    def _detect_subtitle_area(self):
        """
        Read the raw txt file after watermark filtering and detect subtitle area based on coordinates.
        Assumption: Subtitle area has a relatively fixed y-axis coordinate range that appears more
        frequently than scene text.
        :return: Subtitle area position
        """
        # Open raw txt after watermark removal
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # Open txt file with utf-8 encoding
        line = f.readline()  # Read file line by line
        # Y-coordinate list
        y_coordinates_list = []
        while line:
            text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
            y_coordinates_list.append((int(text_position[2]), int(text_position[3])))
            line = f.readline()
        f.close()
        return Counter(y_coordinates_list).most_common(1)

    def _frame_to_timecode(self, frame_no):
        """
        Convert video frame number to timecode using arithmetic (no VideoCapture).
        :param frame_no: Video frame number, i.e. which frame
        :returns: SMPTE format timestamp as string, e.g. '01:02:12,032'
        """
        # Compute timestamp directly from frame number and fps
        total_ms = (frame_no / self.fps) * 1000.0
        hours = int(total_ms // 3600000)
        total_ms %= 3600000
        minutes = int(total_ms // 60000)
        total_ms %= 60000
        seconds = int(total_ms // 1000)
        millis = int(total_ms % 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    def _timestamp_to_frameno(self, time_ms):
        return int(round((time_ms / 1000.0) * self.fps))

    def _frameno_to_milliseconds(self, frame_no):
        return float(int(frame_no / self.fps * 1000))

    def _remove_duplicate_subtitle(self):
        """
        Read raw txt, remove duplicate lines, and return deduplicated subtitle list
        """
        self._concat_content_with_same_frameno()
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()
        RawInfo = namedtuple('RawInfo', 'no content')
        content_list = []
        for line in lines:
            frame_no = line.split('\t')[0]
            content = line.split('\t')[2]
            content_list.append(RawInfo(frame_no, content))
        # Deduplicated subtitle list
        unique_subtitle_list = []
        idx_i = 0
        content_list_len = len(content_list)
        # Iterate through each subtitle line, record start and end times
        while idx_i < content_list_len:
            i = content_list[idx_i]
            start_frame = i.no
            idx_j = idx_i
            while idx_j < content_list_len:
                # Calculate Levenshtein distance between current and next line
                # Check if next frame of idx_j differs from idx_i; if so (or if last frame), end frame is found
                if idx_j + 1 == content_list_len or ratio(i.content.replace(' ', ''), content_list[idx_j + 1].content.replace(' ', '')) < config.THRESHOLD_TEXT_SIMILARITY:
                    # If end frame found, define subtitle end frame number
                    end_frame = content_list[idx_j].no
                    if not self.use_vsf:
                        if end_frame == start_frame and idx_j + 1 < content_list_len:
                            # For single-frame case, use next frame's start time (unless it's the last frame)
                            end_frame = content_list[idx_j + 1][0]
                    # Find the longest subtitle
                    similar_list = content_list[idx_i:idx_j + 1]
                    similar_content_strip_list = [item.content.replace(' ', '') for item in similar_list]
                    index, _ = max(enumerate(similar_content_strip_list), key=lambda x: len(x[1]))

                    # Add to list
                    unique_subtitle_list.append((start_frame, end_frame, similar_list[index].content))
                    idx_i = idx_j + 1
                    break
                else:
                    idx_j += 1
                    continue
        return unique_subtitle_list

    def _concat_content_with_same_frameno(self):
        """
        Merge subtitle lines with the same frame number in raw txt
        """
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()
        content_list = []
        frame_no_list = []
        for line in lines:
            frame_no = line.split('\t')[0]
            frame_no_list.append(frame_no)
            coordinate = line.split('\t')[1]
            content = line.split('\t')[2]
            content_list.append([frame_no, coordinate, content])

        # Find frame numbers that appear more than once
        frame_no_list = [i[0] for i in Counter(frame_no_list).most_common() if i[1] > 1]

        # Find positions of these frame numbers
        concatenation_list = []
        for frame_no in frame_no_list:
            position = [i for i, x in enumerate(content_list) if x[0] == frame_no]
            concatenation_list.append((frame_no, position))

        for i in concatenation_list:
            content = []
            for j in i[1]:
                content.append(content_list[j][2])
            content = ' '.join(content).replace('\n', ' ') + '\n'
            for k in i[1]:
                content_list[k][2] = content

        # Delete redundant subtitle lines
        to_delete = []
        for i in concatenation_list:
            for j in i[1][1:]:
                to_delete.append(content_list[j])
        for i in to_delete:
            if i in content_list:
                content_list.remove(i)

        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, coordinate, content in content_list:
                content = unicodedata.normalize('NFKC', content)
                f.write(f'{frame_no}\t{coordinate}\t{content}')

    def _unite_coordinates(self, coordinates_list):
        """
        Given a coordinate list, unify similar coordinates to a single value.
        e.g. Detection results for the same text position may vary slightly between frames,
        such as (255,123,456,789) vs (253,122,456,799), so similar coordinates need to be unified.
        :param coordinates_list: List containing coordinate points
        :return: Coordinate list with unified values
        """
        # Unify similar coordinates into one
        index = 0
        for coordinate in coordinates_list:  # TODO: O(n^2) time complexity, optimization needed
            for i in coordinates_list:
                if self.__is_coordinate_similar(coordinate, i):
                    coordinates_list[index] = i
            index += 1
        return coordinates_list

    def _compute_image_similarity(self, image1, image2):
        """
        Compute cosine similarity between two images
        """
        image1 = self.__get_thum(image1)
        image2 = self.__get_thum(image2)
        images = [image1, image2]
        vectors = []
        norms = []
        for image in images:
            vector = []
            for pixel_tuple in image.getdata():
                vector.append(average(pixel_tuple))
            vectors.append(vector)
            # linalg = linear algebra, norm represents the norm
            # Compute image norm
            norms.append(linalg.norm(vector, 2))
        a, b = vectors
        a_norm, b_norm = norms
        # dot returns dot product, computed on 2D arrays (matrices)
        res = dot(a / a_norm, b / b_norm)
        return res

    def __get_area_text(self, ocr_result):
        """
        Get text content within subtitle area
        """
        box, text = ocr_result
        coordinates = get_coordinates(box)
        area_text = []
        for content, coordinate in zip(text, coordinates):
            if self.sub_area is not None:
                s_ymin = self.sub_area[0]
                s_ymax = self.sub_area[1]
                s_xmin = self.sub_area[2]
                s_xmax = self.sub_area[3]
                xmin = coordinate[0]
                xmax = coordinate[1]
                ymin = coordinate[2]
                ymax = coordinate[3]
                if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                    area_text.append(content[0])
        return area_text

    def _compare_ocr_result(self, result_cache, img1, img1_no, img2, img2_no):
        """
        Compare whether subtitle area text predicted from two images is the same
        """
        if self.ocr is None:
            self.ocr = OcrRecogniser()
        if img1_no in result_cache:
            area_text1 = result_cache[img1_no]['text']
        else:
            dt_box, rec_res = self.ocr.predict(img1)
            area_text1 = "".join(self.__get_area_text((dt_box, rec_res)))
            result_cache[img1_no] = {'text': area_text1, 'dt_box': dt_box, 'rec_res': rec_res}

        if img2_no in result_cache:
            area_text2 = result_cache[img2_no]['text']
        else:
            dt_box, rec_res = self.ocr.predict(img2)
            area_text2 = "".join(self.__get_area_text((dt_box, rec_res)))
            result_cache[img2_no] = {'text': area_text2, 'dt_box': dt_box, 'rec_res': rec_res}
        delete_no_list = []
        for no in result_cache:
            if no < min(img1_no, img2_no) - 10:
                delete_no_list.append(no)
        for no in delete_no_list:
            del result_cache[no]
        if ratio(area_text1, area_text2) > config.THRESHOLD_TEXT_SIMILARITY:
            return True
        else:
            return False

    @staticmethod
    def __is_coordinate_similar(coordinate1, coordinate2):
        """
        Check if two coordinates are similar. If the differences in xmin, xmax, ymin, ymax
        are all within pixel tolerance, the two coordinates are considered similar.
        """
        return abs(coordinate1[0] - coordinate2[0]) < config.PIXEL_TOLERANCE_X and \
            abs(coordinate1[1] - coordinate2[1]) < config.PIXEL_TOLERANCE_X and \
            abs(coordinate1[2] - coordinate2[2]) < config.PIXEL_TOLERANCE_Y and \
            abs(coordinate1[3] - coordinate2[3]) < config.PIXEL_TOLERANCE_Y

    @staticmethod
    def __get_thum(image, size=(64, 64), greyscale=False):
        """
        Normalize image for processing
        """
        # Resize image, Image.ANTIALIAS for high quality
        image = image.resize(size, Image.ANTIALIAS)
        if greyscale:
            # Convert image to L mode (grayscale, 8 bits per pixel)
            image = image.convert('L')
        return image

    def __delete_frame_cache(self):
        if not config.DEBUG_NO_DELETE_CACHE:
            if len(os.listdir(self.frame_output_dir)) > 0:
                for i in os.listdir(self.frame_output_dir):
                    os.remove(os.path.join(self.frame_output_dir, i))

    def empty_cache(self):
        """
        Delete all cache files generated during subtitle extraction
        """
        if not config.DEBUG_NO_DELETE_CACHE:
            if os.path.exists(self.temp_output_dir):
                shutil.rmtree(self.temp_output_dir, True)

    def update_progress(self, ocr=None, frame_extract=None):
        """
        Update progress bar
        :param ocr: OCR progress
        :param frame_extract: Video frame extraction progress
        """
        if ocr is not None:
            self.progress_ocr = ocr
        if frame_extract is not None:
            self.progress_frame_extract = frame_extract
        self.progress_total = (self.progress_frame_extract + self.progress_ocr) / 2

    def start_subtitle_ocr_async(self):
        def get_ocr_progress():
            """
            Get OCR recognition progress
            """
            # Get total video frame count
            total_frame_count = self.frame_count
            # Whether to print subtitle search start message
            notify = True
            while True:
                current_frame_no = self.subtitle_ocr_progress_queue.get(block=True)
                if notify:
                    print(config.interface_config['Main']['StartFindSub'])
                    notify = False
                self.update_progress(
                    ocr=100 if current_frame_no == -1 else (current_frame_no / total_frame_count * 100))
                # print(f'recv total_ms:{total_ms}')
                if current_frame_no == -1:
                    return

        process, task_queue, progress_queue = subtitle_ocr.async_start(self.video_path,
                                                                       self.raw_subtitle_path,
                                                                       self.sub_area,
                                                                       options={'REC_CHAR_TYPE': config.REC_CHAR_TYPE,
                                                                                'DROP_SCORE': config.DROP_SCORE,
                                                                                'SUB_AREA_DEVIATION_RATE': config.SUB_AREA_DEVIATION_RATE,
                                                                                'DEBUG_OCR_LOSS': config.DEBUG_OCR_LOSS,
                                                                                }
                                                                       )
        self.subtitle_ocr_task_queue = task_queue
        self.subtitle_ocr_progress_queue = progress_queue
        # Start thread for updating OCR progress
        Thread(target=get_ocr_progress, daemon=True).start()
        return process

    @staticmethod
    def srt2txt(srt_file):
        subs = pysrt.open(srt_file, encoding='utf-8')
        output_path = os.path.join(os.path.dirname(srt_file), Path(srt_file).stem + '.txt')
        print(output_path)
        with open(output_path, 'w') as f:
            for sub in subs:
                f.write(f'{sub.text}\n')


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    # Prompt user to input video path
    video_path = input(f"{config.interface_config['Main']['InputVideo']}").strip()
    # Prompt user to input subtitle area
    try:
        y_min, y_max, x_min, x_max = map(int, input(
            f"{config.interface_config['Main']['ChooseSubArea']} (ymin ymax xmin xmax)：").split())
        subtitle_area = (y_min, y_max, x_min, x_max)
    except ValueError as e:
        subtitle_area = None
    # Create subtitle extractor object
    se = SubtitleExtractor(video_path, subtitle_area)
    # Use fast pipeline by default; pass --legacy for original behavior
    if '--legacy' in sys.argv:
        se.run()
    else:
        se.run_fast()
