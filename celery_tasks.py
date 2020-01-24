import os
import json
import logging
import urllib.request

import arrow
from celery import Celery
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image
from eyewitness.result_handler.db_writer import BboxPeeweeDbWriter
from eyewitness.config import RAW_IMAGE_PATH
from peewee import SqliteDatabase
from bistiming import Stopwatch
from eyewitness.config import BBOX
from eyewitness.detection_result_filter import FeedbackBboxDeNoiseFilter

from naive_detector import TensorRTYoloV3DetectorWrapper
from line_detection_result_handler import LineAnnotationSender
from facebook_detection_result_handler import FaceBookAnnoationSender


# leave interface for inference image shape
INFERENCE_SHAPE = os.environ.get('inference_shape', '608,608')
INFERENCE_SHAPE = tuple(int(i) for i in INFERENCE_SHAPE.split(','))
assert len(INFERENCE_SHAPE) == 2

# leave interface for detector threshold
DETECTION_THRESHOLD = float(os.environ.get('threshold', 0.14))

# valid_labels
VALID_LABELS = os.environ.get('valid_labels')
if VALID_LABELS is not None:
    VALID_LABELS = set(VALID_LABELS.split(','))

GLOBAL_OBJECT_DETECTOR = TensorRTYoloV3DetectorWrapper(
    engine_file=os.environ.get('engine_file', 'yolov3.engine'),
    image_shape=INFERENCE_SHAPE,
    threshold=DETECTION_THRESHOLD,
    valid_labels=VALID_LABELS
)

RAW_IMAGE_FOLDER = os.environ.get('raw_image_fodler', 'raw_image')
DETECTED_IMAGE_FOLDER = os.environ.get('detected_image_folder', 'detected_image')
BROKER_URL = os.environ.get('broker_url', 'amqp://guest:guest@rabbitmq:5672')

DETECTION_RESULT_HANDLERS = []


def image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image"""
    site_domain = os.environ.get('site_domain')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, drawn_image_path)


def raw_image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image"""
    site_domain = os.environ.get('site_domain')
    raw_image_path = drawn_image_path.replace('detected_image/', 'raw_image/')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, raw_image_path)


def line_detection_result_filter(detection_result):
    """
    used to check if sent notification or not
    """
    return any(i.label == 'person' for i in detection_result.detected_objects)


SQLITE_DB_PATH = os.environ.get('db_path')
RESULT_HANDLERS = []
DENOISE_FILTERS = []
if SQLITE_DB_PATH is not None:
    DATABASE = SqliteDatabase(SQLITE_DB_PATH)
    DB_RESULT_HANDLER = BboxPeeweeDbWriter(DATABASE)
    IMAGE_REGISTER = DB_RESULT_HANDLER
    DETECTION_RESULT_HANDLERS.append(DB_RESULT_HANDLER)

    # setup your line channel token and audience
    channel_access_token = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
    if channel_access_token:
        line_annotation_sender = LineAnnotationSender(
            channel_access_token=channel_access_token,
            image_url_handler=image_url_handler,
            raw_image_url_handler=raw_image_url_handler,
            detection_result_filter=line_detection_result_filter,
            detection_method=BBOX,
            update_audience_period=10,
            database=DATABASE)
        RESULT_HANDLERS.append(line_annotation_sender)

    fb_user_email = os.environ.get('FACEBOOK_USER_EMAIL')
    if fb_user_email:
        fb_user_password = os.environ.get('FACEBOOK_USER_PASSWORD')
        fb_session_cookie_path = os.environ.get('FACEBOOK_SESSION_COOKIES_PATH')
        audience_id_str = os.environ.get('YOUR_USER_ID')
        audience_ids = set([i for i in audience_id_str.split(',') if i])
        with open(fb_session_cookie_path, 'r') as f:
            session_dict = json.load(f)

        facebook_annotation_sender = FaceBookAnnoationSender(
            audience_ids=audience_ids,
            user_email=fb_user_email,
            user_password=fb_user_password,
            session_dict=session_dict,
            image_url_handler=image_url_handler,
            detection_result_filter=line_detection_result_filter,
            detection_method=BBOX)
        RESULT_HANDLERS.append(facebook_annotation_sender)

    # denoise filter
    denoise_filter = FeedbackBboxDeNoiseFilter(
        DATABASE, detection_threshold=DETECTION_THRESHOLD)
    DENOISE_FILTERS.append(denoise_filter)

celery = Celery('tasks', broker=BROKER_URL)


def generate_image_url(channel):
    return "https://upload.wikimedia.org/wikipedia/commons/2/25/5566_and_Daily_Air_B-55507_20050820.jpg"  # noqa


def generate_image(channel, timestamp, raw_image_path=None):
    image_id = ImageId(channel=channel, timestamp=timestamp, file_format='jpg')
    if not raw_image_path:
        raw_image_path = "%s/%s.jpg" % ('raw_image', str(image_id))  # used for db
        # generate raw image
        urllib.request.urlretrieve(generate_image_url(channel), raw_image_path)
    return Image(image_id, raw_image_path=raw_image_path)


@celery.task(name='detect_image')
def detect_image(params):
    channel = params.get('channel', 'demo')
    timestamp = params.get('timestamp', arrow.now().timestamp)
    is_store_detected_image = params.get('is_store_detected_image', True)
    raw_image_path = params.get('raw_image_path')

    image_obj = generate_image(channel, timestamp, raw_image_path)
    IMAGE_REGISTER.register_image(image_obj.image_id, {RAW_IMAGE_PATH: raw_image_path})

    with Stopwatch('Running inference on image {}...'.format(image_obj.raw_image_path)):
        detection_result = GLOBAL_OBJECT_DETECTOR.detect(image_obj)

    if is_store_detected_image and len(detection_result.detected_objects) > 0:
        ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
        drawn_image_path = "%s/%s.jpg" % (DETECTED_IMAGE_FOLDER, str(image_obj.image_id))
        ImageHandler.save(image_obj.pil_image_obj, drawn_image_path)
        # used for visualization
        drawn_image_path_for_db = "%s/%s.jpg" % ('detected_image', str(image_obj.image_id))
        detection_result.image_dict['drawn_image_path'] = drawn_image_path_for_db

    for detection_result_filter in DENOISE_FILTERS:
        try:
            detection_result = detection_result_filter.apply(detection_result)
        except Exception as e:
            logging.info(e)

    for detection_result_handler in DETECTION_RESULT_HANDLERS:
        try:
            detection_result_handler.handle(detection_result)
        except Exception as e:
            logging.info(e)
