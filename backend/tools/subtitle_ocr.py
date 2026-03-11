import os
import re
from multiprocessing import Queue, Process
import cv2
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
from backend.tools.ocr import OcrRecogniser, get_coordinates
from backend.tools.constant import SubtitleArea
from backend.tools import constant
from threading import Thread
import queue
from shapely.geometry import Polygon
from types import SimpleNamespace
import shutil
import numpy as np
from collections import namedtuple


def preprocess_for_ocr(frame):
    """
    Preprocess OCR input: grayscale, denoise, upscale, threshold.
    Returns the processed BGR image and the scale factor.
    """
    if frame is None or frame.size == 0:
        return frame, 1.0

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    scale = 2.0
    resized = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    binary = cv2.adaptiveThreshold(
        resized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return processed, scale


def remap_coordinate(coordinate, offset_x=0, offset_y=0, scale=1.0):
    """Remap cropped/scaled coordinates back to the original video coordinate system."""
    xmin, xmax, ymin, ymax = coordinate
    if scale != 1.0:
        xmin = int(round(xmin / scale))
        xmax = int(round(xmax / scale))
        ymin = int(round(ymin / scale))
        ymax = int(round(ymax / scale))
    return xmin + offset_x, xmax + offset_x, ymin + offset_y, ymax + offset_y


def extract_subtitles(data, text_recogniser, img, raw_subtitle_file,
                      sub_area, options, dt_box_arg, rec_res_arg, ocr_loss_debug_path, preprocess_meta=None):
    """
    Extract subtitle information from video frames
    """
    # Get detection boxes and recognition results from arguments
    dt_box = dt_box_arg
    rec_res = rec_res_arg
    # If no detection results available, perform detection
    if dt_box is None or rec_res is None:
        dt_box, rec_res = text_recogniser.predict(img)
        # rec_res format: ("hello", 0.997)
    preprocess_meta = preprocess_meta or {}
    offset_x = int(preprocess_meta.get('offset_x', 0))
    offset_y = int(preprocess_meta.get('offset_y', 0))
    scale = float(preprocess_meta.get('scale', 1.0))

    # Get text coordinates and remap to original frame coordinate system
    coordinates = [remap_coordinate(c, offset_x, offset_y, scale) for c in get_coordinates(dt_box)]
    # Write results to txt file
    if options.REC_CHAR_TYPE == 'en':
        # If recognition language is English, remove Chinese characters
        text_res = [(re.sub('[\u4e00-\u9fa5]', '', res[0]), res[1]) for res in rec_res]
    else:
        text_res = [(res[0], res[1]) for res in rec_res]
    line = ''
    loss_list = []
    selected_texts = []
    for content, coordinate in zip(text_res, coordinates):
        text = content[0]
        prob = content[1]
        if sub_area is not None:
            selected = False
            # Initialize overflow deviation to 0
            overflow_area_rate = 0
            # User-specified subtitle area
            sub_area_polygon = sub_area_to_polygon(sub_area)
            # Detected subtitle area
            coordinate_polygon = coordinate_to_polygon(coordinate)
            # Calculate intersection between the two areas
            intersection = sub_area_polygon.intersection(coordinate_polygon)
            # If there is an intersection
            if not intersection.is_empty:
                # Calculate overflow tolerance
                overflow_area_rate = ((sub_area_polygon.area + coordinate_polygon.area - intersection.area) / sub_area_polygon.area) - 1
                # If overflow ratio is below threshold and text recognition confidence is above threshold
                if overflow_area_rate <= options.SUB_AREA_DEVIATION_RATE and prob > options.DROP_SCORE:
                    # Keep this frame
                    selected = True
                    selected_texts.append(text)
                    line += f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n'
                    raw_subtitle_file.write(f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n')
            # Save dropped recognition results
            loss_info = namedtuple('loss_info', 'text prob overflow_area_rate coordinate selected')
            loss_list.append(loss_info(text, prob, overflow_area_rate, coordinate, selected))
        else:
            selected_texts.append(text)
            raw_subtitle_file.write(f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n')
    if selected_texts and (int(data['i']) % 10 == 1):
        print(f"[OCR][{str(data['i']).zfill(8)}] {' '.join(selected_texts)}")
    # Output debug information
    dump_debug_info(options, line, img, loss_list, ocr_loss_debug_path, sub_area, data)


def dump_debug_info(options, line, img, loss_list, ocr_loss_debug_path, sub_area, data):
    loss = False
    if options.DEBUG_OCR_LOSS and options.REC_CHAR_TYPE in ('ch', 'japan ', 'korea', 'ch_tra'):
        loss = len(line) > 0 and re.search(r'[\u4e00-\u9fa5\u3400-\u4db5\u3130-\u318F\uAC00-\uD7A3\u0800-\u4e00]', line) is None
    if loss:
        if not os.path.exists(ocr_loss_debug_path):
            os.makedirs(ocr_loss_debug_path, mode=0o777, exist_ok=True)
        img = cv2.rectangle(img, (sub_area[2], sub_area[0]), (sub_area[3], sub_area[1]), constant.BGR_COLOR_BLUE, 2)
        for loss_info in loss_list:
            coordinate = loss_info.coordinate
            color = constant.BGR_COLOR_GREEN if loss_info.selected else constant.BGR_COLOR_RED
            text = f"[{loss_info.text}] prob:{loss_info.prob:.4f} or:{loss_info.overflow_area_rate:.2f}"
            img = paint_chinese_opencv(img, text, pos=(coordinate[0], coordinate[2] - 30), color=color)
            img = cv2.rectangle(img, (coordinate[0], coordinate[2]), (coordinate[1], coordinate[3]), color, 2)
        cv2.imwrite(os.path.join(os.path.abspath(ocr_loss_debug_path), f'{str(data["i"]).zfill(8)}.png'), img)


def sub_area_to_polygon(sub_area):
    s_ymin = sub_area[0]
    s_ymax = sub_area[1]
    s_xmin = sub_area[2]
    s_xmax = sub_area[3]
    return Polygon([[s_xmin, s_ymin], [s_xmax, s_ymin], [s_xmax, s_ymax], [s_xmin, s_ymax]])


def coordinate_to_polygon(coordinate):
    xmin = coordinate[0]
    xmax = coordinate[1]
    ymin = coordinate[2]
    ymax = coordinate[3]
    return Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NotoSansCJK-Bold.otf')
FONT = ImageFont.truetype(FONT_PATH, 20)


def paint_chinese_opencv(im, chinese, pos, color):
    img_pil = Image.fromarray(im)
    fill_color = color  # (color[2], color[1], color[0])
    position = pos
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, chinese, font=FONT, fill=fill_color)
    img = np.asarray(img_pil)
    return img


def ocr_task_consumer(ocr_queue, raw_subtitle_path, sub_area, video_path, options):
    """
    Consumer: consumes ocr_queue, extracts data from OCR queue, performs OCR recognition, and writes to subtitle file
    :param ocr_queue (current_frame_no, frame, dt_box detection box, rec_res recognition result)
    :param raw_subtitle_path
    :param sub_area
    :param video_path
    :param options
    """
    data = {'i': 1}
    # Initialize text recogniser object
    text_recogniser = OcrRecogniser()
    # Storage path for lost subtitles
    ocr_loss_debug_path = os.path.join(os.path.abspath(os.path.splitext(video_path)[0]), 'loss')
    # Remove previous cache
    if os.path.exists(ocr_loss_debug_path):
        shutil.rmtree(ocr_loss_debug_path, True)

    with open(raw_subtitle_path, mode='w+', encoding='utf-8') as raw_subtitle_file:
        while True:
            try:
                frame_no, frame, dt_box, rec_res, preprocess_meta = ocr_queue.get(block=True)
                if frame_no == -1:
                    return
                data['i'] = frame_no
                extract_subtitles(data, text_recogniser, frame, raw_subtitle_file, sub_area, options, dt_box,
                                  rec_res, ocr_loss_debug_path, preprocess_meta)
            except Exception as e:
                print(e)
                break


def ocr_task_producer(ocr_queue, task_queue, progress_queue, video_path, raw_subtitle_path, sub_area):
    """
    Producer: produces data for OCR recognition and adds it to ocr_queue
    :param ocr_queue (current_frame_no, frame, dt_box detection box, rec_res recognition result)
    :param task_queue (total_frame_count, current_frame_no, dt_box detection box, rec_res recognition result, subtitle_area)
    :param progress_queue
    :param video_path
    :param raw_subtitle_path
    """
    cap = cv2.VideoCapture(video_path)
    tbar = None
    while True:
        try:
            # Extract task information from the task queue
            total_frame_count, current_frame_no, dt_box, rec_res, total_ms, default_subtitle_area = task_queue.get(block=True)
            progress_queue.put(current_frame_no)
            if tbar is None:
                tbar = tqdm(total=round(total_frame_count), position=1)
            # current_frame equals -1 means all video frames have been read
            if current_frame_no == -1:
                # Add end marker to OCR recognition queue
                ocr_queue.put((-1, None, None, None, None))
                # Update progress bar
                tbar.update(tbar.total - tbar.n)
                break
            tbar.update(round(current_frame_no - tbar.n))
            # Set current video frame
            # If total_ms is not empty, VSF was used to extract subtitles
            if total_ms is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, total_ms)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_no - 1)
            # Read video frame
            ret, frame = cap.read()
            # If read successful
            if ret:
                offset_x, offset_y = 0, 0
                # When subtitle area is specified, crop by user area first to reduce background noise
                if isinstance(sub_area, (tuple, list)) and len(sub_area) == 4:
                    y_min, y_max, x_min, x_max = map(int, sub_area)
                    y_min = max(0, min(y_min, frame.shape[0] - 1))
                    y_max = max(0, min(y_max, frame.shape[0]))
                    x_min = max(0, min(x_min, frame.shape[1] - 1))
                    x_max = max(0, min(x_max, frame.shape[1]))
                    if y_max > y_min and x_max > x_min:
                        frame = frame[y_min:y_max, x_min:x_max]
                        offset_x, offset_y = x_min, y_min

                # Crop the video frame based on default subtitle position, then process
                if default_subtitle_area is not None:
                    frame = frame_preprocess(default_subtitle_area, frame)

                frame, scale = preprocess_for_ocr(frame)
                preprocess_meta = {'offset_x': offset_x, 'offset_y': offset_y, 'scale': scale}

                # After cropping/scaling, cached detection box coordinates are no longer reusable
                if offset_x != 0 or offset_y != 0 or scale != 1.0:
                    dt_box, rec_res = None, None

                ocr_queue.put((current_frame_no, frame, dt_box, rec_res, preprocess_meta))
        except Exception as e:
            print(e)
            break
    cap.release()


def subtitle_extract_handler(task_queue, progress_queue, video_path, raw_subtitle_path, sub_area, options):
    """
    Create and start a video frame extraction thread and an OCR recognition thread
    :param task_queue Task queue, (total_frame_count, current_frame_no, dt_box detection box, rec_res recognition result, subtitle_area)
    :param progress_queue Progress queue
    :param video_path Video path
    :param raw_subtitle_path Raw subtitle file path
    :param sub_area Subtitle area
    :param options Options
    """
    # Delete cache
    if os.path.exists(raw_subtitle_path):
        os.remove(raw_subtitle_path)
    # Create an OCR queue, recommended size 8-20
    ocr_queue = queue.Queue(20)
    # Create an OCR event producer thread
    ocr_event_producer_thread = Thread(target=ocr_task_producer,
                                       args=(ocr_queue, task_queue, progress_queue, video_path, raw_subtitle_path, sub_area,),
                                       daemon=True)
    # Create an OCR event consumer thread
    ocr_event_consumer_thread = Thread(target=ocr_task_consumer,
                                       args=(ocr_queue, raw_subtitle_path, sub_area, video_path, options,),
                                       daemon=True)
    # Start consumer thread
    ocr_event_producer_thread.start()
    # Start producer thread
    ocr_event_consumer_thread.start()
    # join() blocks the main thread until all child threads have finished execution
    ocr_event_producer_thread.join()
    ocr_event_consumer_thread.join()


def async_start(video_path, raw_subtitle_path, sub_area, options):
    """
    Start a process to handle async tasks
    options.REC_CHAR_TYPE
    options.DROP_SCORE
    options.SUB_AREA_DEVIATION_RATE
    options.DEBUG_OCR_LOSS
    """
    assert 'REC_CHAR_TYPE' in options, "options missing parameter: REC_CHAR_TYPE"
    assert 'DROP_SCORE' in options, "options missing parameter: DROP_SCORE"
    assert 'SUB_AREA_DEVIATION_RATE' in options, "options missing parameter: SUB_AREA_DEVIATION_RATE"
    assert 'DEBUG_OCR_LOSS' in options, "options missing parameter: DEBUG_OCR_LOSS"
    # Create a task queue
    # Task format: (total_frame_count, current_frame_no, dt_box detection box, rec_res recognition result, subtitle_area)
    task_queue = Queue()
    # Create a progress update queue
    progress_queue = Queue()
    # Create a new process
    p = Process(target=subtitle_extract_handler,
                args=(task_queue, progress_queue, video_path, raw_subtitle_path, sub_area, SimpleNamespace(**options),))
    # Start the process
    p.start()
    return p, task_queue, progress_queue


def frame_preprocess(subtitle_area, frame):
    """
    Crop the video frame
    """
    # For videos with resolution greater than 1920*1080, scale frames proportionally to 1280*720 for recognition
    # paddlepaddle will compress the image to 640*640
    # if self.frame_width > 1280:
    #     scale_rate = round(float(1280 / self.frame_width), 2)
    #     frames = cv2.resize(frames, None, fx=scale_rate, fy=scale_rate, interpolation=cv2.INTER_AREA)
    # If subtitle area is in the lower part
    if subtitle_area == SubtitleArea.LOWER_PART:
        cropped = int(frame.shape[0] // 2)
        # Crop the video frame to the lower half
        frame = frame[cropped:]
    # If subtitle area is in the upper part
    elif subtitle_area == SubtitleArea.UPPER_PART:
        cropped = int(frame.shape[0] // 2)
        # Crop the video frame to the upper half
        frame = frame[:cropped]
    return frame


if __name__ == "__main__":
    pass
