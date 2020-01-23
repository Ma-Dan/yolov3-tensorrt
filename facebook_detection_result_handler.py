import os
import json

import arrow
from eyewitness.config import (
    BBOX,
    BoundedBoxObject,
    DRAWN_IMAGE_PATH,
    DETECTED_OBJECTS,
    IMAGE_ID,
    DETECTION_METHOD
)
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.detection_utils import DetectionResultHandler
from eyewitness.models.feedback_models import RegisteredAudience
from eyewitness.models.db_proxy import DATABASE_PROXY
from fbchat import Client
from fbchat.models import ThreadType, Message

FACEBOOK_FALSE_ALERT_MSG_TEMPLATE = "false_alert_{image_id}_{meta}"
FACEBOOK_PLATFROM = 'facebook'


class CustomClient56(Client):
    def onMessage(self, message_object, author_id, thread_id, thread_type, **kwargs):
        self.markAsDelivered(thread_id, message_object.uid)
        self.markAsRead(thread_id)
        text_msg = Message(
            text="What makes me sad, is that I give up you and love, and my dream is also broken, "
            "but I need to fight back tears")

        if '5566' in message_object.text:
            self.send(text_msg, thread_id=author_id, thread_type=thread_type)


class FaceBookAnnoationSender(DetectionResultHandler):
    def __init__(self, user_email, user_password, image_url_handler, raw_image_url_handler=None,
                 audience_ids=None, update_audience_period=0, detection_result_filter=None,
                 detection_method=BBOX, database=None, session_dict=None):
        """ Facebook Annotation sender which requires python library: `fbchat`

        Parameters
        ----------
        audience_ids: set
            line audience ids
        channel_access_token: str
            channel_access_token
        image_url_handler: Callable
            compose drawn image to image_url
        update_audience_period: int
            the period(seconds) that update the audience_ids from audience model, 0 will not update
        detection_result_filter: Callable
            a function check if the detection result to sent or not
        detection_method: str
            detection method
        database: peewee.Database
            peewee database obj, used to query registered audiences
        """
        self.water_mark_time = arrow.now()
        self.update_audience_period = update_audience_period
        if session_dict:
            self.client = Client(
                email=user_email, password=user_password, session_cookies=session_dict)
        else:
            # without seesion information might failure here
            self.client = Client(email=user_email, password=user_password)

        self.detection_result_filter = detection_result_filter
        self._detection_method = detection_method
        self.database = database
        if database:
            self.create_db_table()
        if audience_ids is None:
            self.audience_ids = self.get_registered_audiences()
        else:
            self.audience_ids = audience_ids

        # setup image_url handler and raw_image_url_handler
        self.image_url_handler = image_url_handler
        if raw_image_url_handler is not None:
            self.raw_image_url_handler = raw_image_url_handler
        else:
            self.raw_image_url_handler = self.image_url_handler

    @property
    def detection_method(self):
        """str: detection_method"""
        return self._detection_method

    def create_db_table(self):
        """create the RegisteredAudience if not exist
        """
        self.check_proxy_db()
        RegisteredAudience.create_table()

    def get_registered_audiences(self):
        """ get the RegisteredAudience id list
        """
        if self.database is None:
            raise("the database is not set")
        self.check_proxy_db()
        query = RegisteredAudience.select().where(
            RegisteredAudience.platform_id == FACEBOOK_PLATFROM)
        audiences = set(i.user_id for i in query)
        return audiences

    def check_proxy_db(self):
        """check if the db proxy is correct one, if not initialize again.
        """
        if not (self.database is DATABASE_PROXY.obj):
            DATABASE_PROXY.initialize(self.database)

    def audience_update(self):
        """
        update the audiences from RegisteredAudience model
        """
        if self.update_audience_period and self.database is not None:
            diff_seconds = (arrow.now() - self.water_mark_time).total_seconds()
            if diff_seconds > self.update_audience_period:
                audience_ids = self.get_registered_audiences()
                self.audience_ids = audience_ids

    def _handle(self, detection_result):
        # update audience
        self.audience_update()

        # check if detection result need to sent_out
        if self.detection_result_filter(detection_result):
            image_url = self.image_url_handler(detection_result.drawn_image_path)
            # TODO: consider a better way to generate raw_image_url
            raw_image_url = self.raw_image_url_handler(detection_result.drawn_image_path)
            false_alert_feedback_text = FACEBOOK_FALSE_ALERT_MSG_TEMPLATE.format(
                image_id=str(detection_result.image_id), meta='')
            self.send_annotation_button_msg(image_url, raw_image_url, false_alert_feedback_text)

    def send_annotation_button_msg(self, image_url, raw_image_url, false_alert_feedback_text):
        """
        sent line botton msg to audience_ids

        Parameters
        ----------
        image_url: str
            the url of image

        raw_image_url: str
            the url of raw_image, which might be another bigger/clear image
            not used for FB

        false_alert_feedback_text: str
            false_alert msg used to sent to feedback_handler
        """
        image_url = image_url
        text_msg = Message(text=false_alert_feedback_text)
        if self.audience_ids:  # check if audiences
            for audience_id in self.audience_ids:
                self.client.sendRemoteFiles(
                    message=text_msg, file_urls=image_url,
                    thread_id=audience_id, thread_type=ThreadType.USER)


if __name__ == '__main__':
    user_email = os.environ.get('FACEBOOK_USER_EMAIL')
    user_password = os.environ.get('FACEBOOK_USER_PASSWORD')
    session_cookie_path = os.environ.get('FACEBOOK_SESSION_COOKIES_PATH')
    with open(session_cookie_path, 'r') as f:
        session_dict = json.load(f)

    audience_ids = set([os.environ.get('YOUR_USER_ID')])

    def image_url_handler(drawn_image_path):
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'

    def detection_result_filter(detection_result):
        return any(i.label == 'pikachu' for i in detection_result.detected_objects)

    facebook_annotation_sender = FaceBookAnnoationSender(
        audience_ids=audience_ids,
        user_email=user_email,
        user_password=user_password,
        session_dict=session_dict,
        image_url_handler=image_url_handler,
        detection_result_filter=detection_result_filter,
        detection_method=BBOX)

    image_dict = {
        IMAGE_ID: ImageId('pikachu', 1541860141, 'jpg'),
        DETECTED_OBJECTS: [
            BoundedBoxObject(*(250, 100, 800, 900, 'pikachu', 0.5, ''))
        ],
        DRAWN_IMAGE_PATH: 'pikachu_test.png',
        DETECTION_METHOD: BBOX
    }
    detection_result = DetectionResult(image_dict)

    # sent the button msg out
    facebook_annotation_sender.handle(detection_result)

    facebook_annotation_sender.client.listen()
