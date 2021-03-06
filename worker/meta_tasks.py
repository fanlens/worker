"""`Celery` tasks related to extracting meta data of stored `Data`"""
from datetime import datetime
from typing import Any, Iterable, Optional, cast

import dateutil.parser
from celery.utils.log import get_task_logger

from brain.feature.fingerprint import get_fingerprints
from brain.feature.language_detect import language_detect
from brain.feature.translate import translate
from common.db import Session, get_session
from common.db.models.activities import Data, Fingerprint, Lang, Language, SourceFeature, SourceUser, TagSetUser, Text, \
    Time, Translation, Type
from common.db.models.brain import Model, ModelSources, ModelUser, Prediction
from common.db.models.users import User
from common.job import Space
from common.utils.buffered import Buffered, HandlerBase
from common.utils.simple_utc import SimpleUTC
from . import exclusive_task
from .app import app
from .brain_tasks import predict_stored_all

_LOGGER = get_task_logger(__name__)

_TEXT_EXTRACTORS = {
    Type.facebook: lambda data: data.data['message'],
    Type.twitter: lambda data: data.data['text'],
    Type.twitter_dm: lambda data: data.data['message_create']['message_data']['text'],
    Type.generic: lambda data: data.data['text'],
}

_TIME_EXTRACTORS = {
    Type.facebook: lambda data: dateutil.parser.parse(data.data.get('created_time')),
    Type.twitter: lambda data: datetime.strptime(data.data['created_at'],
                                                 '%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=_SIMPLE_UTC),
    Type.twitter_dm: lambda data: datetime.utcfromtimestamp(int(data['created_timestamp'])),
    Type.generic: lambda data: dateutil.parser.parse(data.data.get('created_time')),
}

_SIMPLE_UTC = SimpleUTC()


def _extract_text(data: Data) -> Data:
    # text = (text_extractors[data.source.type](data)
    #     .replace('\x7f', '')
    #     .replace('\b', '')
    #     .encode('ascii', 'ignore')
    #     .decode('utf-8', 'ignore'))
    text = _TEXT_EXTRACTORS[Type(data.source.type)](data)  # type: ignore
    data.text = Text(text=text)
    return data


@app.task(trail=True, ignore_result=True)
def extract_text(*_: Any) -> None:
    """Task that extracts text from stored data"""
    _LOGGER.info('Extracting text ...')
    num = 0
    session: Session
    with get_session() as session:
        for datum in session.query(Data).outerjoin(Text).filter(Text.id.is_(None)):
            _extract_text(datum)
            num += 1
            if not num % 500:
                session.commit()
        session.commit()
    _LOGGER.info('... Done, added %d texts', num)


def _extract_time(data: Data) -> Data:
    time = _TIME_EXTRACTORS[Type(data.source.type)](data)  # type: ignore
    data.time = Time(time=time)
    return data


@app.task(trail=True, ignore_result=True)
def extract_time(*_: Any) -> None:
    """Task that extracts creation time from stored data"""
    _LOGGER.info('Extracting time ...')
    num = 0
    session: Session
    with get_session() as session:
        for datum in session.query(Data).outerjoin(Time).filter(Time.id.is_(None)):
            _extract_time(datum)
            num += 1
            if not num % 500:
                session.commit()
        session.commit()
    _LOGGER.info('... Done, added %d times', num)


def _add_language(data: Data) -> Data:
    try:
        lang = language_detect(data.text.text)
        data.language = Language(language=Lang[lang])  # false positive, pylint: disable=unsubscriptable-object
    except KeyError as err:
        _LOGGER.error('couldn\'t lang detect: %s', data.text.text)
        _LOGGER.exception(err)
    return data


@app.task(trail=True, ignore_result=True)
def add_language(*_: Any) -> None:
    """Task that detects language of the text"""
    _LOGGER.info('Adding language ...')
    num = 0
    session: Session
    with get_session() as session:
        for datum in session.query(Data).join(Text).outerjoin(Language).filter(Language.id.is_(None)):
            _add_language(datum)
            num += 1
            if not num % 500:
                session.commit()
        session.commit()
    _LOGGER.info('... Done, added %d languages', num)


class _TranslationsHandler(HandlerBase[Data]):
    # pylint: disable=too-few-public-methods
    """Handler used for buffered translation"""

    def __init__(self, session: Session) -> None:
        """ :param session: the database session to bind to """
        self._session = session

    def __call__(self, batch: Iterable[Data]) -> None:
        """ :param batch: source of `Data` objects to create translations for """
        texts = [entry.text.text for entry in batch]
        if not texts:
            return
        _LOGGER.info('Creating translations for %d texts', len(texts))
        translations = translate(texts)
        for store_entry, translation in zip(batch, translations):
            if translation:
                store_entry.text.translations.append(Translation(translation=translation, target_language='en'))
        _LOGGER.info('Flushing translations data, %d translations', len(translations))
        self._session.commit()


@app.task(trail=True, ignore_result=True)
def add_translation(*_: Any) -> None:
    """
    Task that translates non-english text of stored `Data`.
    Attention: currently limited to german and spanish.
    Attention: Needs to be enabled as a source feature
    """
    _LOGGER.info('Adding translations ...')
    session: Session
    with get_session() as session:
        entries = (session.query(Data)
                   .join(SourceFeature, SourceFeature.source_id == Data.source_id)
                   .join(Text, Text.data_id == Data.id)
                   .join(Language, Language.data_id == Data.id)
                   .outerjoin(Translation, Translation.text_id == Text.id)
                   .filter(Translation.id.is_(None) &
                           (SourceFeature.feature == 'translate') &
                           Language.language.in_((Lang.de.name, Lang.es.name))))
        buffered = Buffered(entries, _TranslationsHandler(session), 10)
        buffered()
        _LOGGER.info('... Done translations')


class _FingerprintHandler(HandlerBase[Data]):
    # pylint: disable=too-few-public-methods
    """Handler used for buffered translation"""

    def __init__(self, session: Session) -> None:
        """ :param session: the database session to bind to """
        self._session = session

    @staticmethod
    def _get_text(entry: Data) -> str:
        """
        Get the english text of a data entry
        :param entry: the data entry
        :raises ValueError, if no english text can be provided
        """
        entry_text: str = entry.text.text
        if entry_text is None:
            raise ValueError('Data entry %s has no text' % entry.id)

        language = entry.language.language
        if language == Lang.en:
            return entry_text

        english_translation: Optional[Translation] = entry.text.translations.filter(
            Translation.target_language == Lang.en.name).one_or_none()
        if english_translation:
            return cast(str, english_translation.translation)
        else:
            raise ValueError(
                'Data entry %d is neither english nor posses an English translation:'
                '\n\t%s\n\tLanguage: %s' % (entry.id, entry_text, language))

    def __call__(self, batch: Iterable[Data]) -> None:
        """ :param batch: source of `Data` objects to create fingerprints for """
        texts = [self._get_text(entry) or "unknown" for entry in batch]
        if not texts:
            return
        _LOGGER.info('Creating fingerprints for %d texts', len(texts))
        fingerprints = list(get_fingerprints(texts))
        for store_entry, fingerprint in zip(batch, fingerprints):
            store_entry.fingerprint = Fingerprint(fingerprint=fingerprint)
        _LOGGER.info('Flushing fingerprint data, %d fingerprints', len(fingerprints))
        self._session.commit()


@app.task(trail=True, ignore_result=True)
def add_fingerprint(*_: Any) -> None:
    """Task that adds fingerprints to stored `Data`."""
    _LOGGER.info('Adding fingerprints ...')
    session: Session
    with get_session() as session:
        entries = (session.query(Data)
                   .join(Text, Text.data_id == Data.id)
                   .outerjoin(Translation, (Translation.text_id == Text.id))
                   .outerjoin(Fingerprint, Fingerprint.data_id == Data.id)
                   .filter((Data.language.has(language=Lang.en.name) | (Translation.target_language == Lang.en.name)) &
                           Fingerprint.id.is_(None)))
        buffered = Buffered(entries, _FingerprintHandler(session), 500)
        buffered()
    _LOGGER.info('... Done fingerprints')


@app.task(trail=True, ignore_result=True)
def add_prediction(*_: Any) -> None:
    """Task that adds `Predcition` to stored `Data`."""
    _LOGGER.info('Adding predictions ...')
    session: Session
    with get_session() as session:
        predict_stored_all(session.query(Data.id)
                           .join(SourceUser, SourceUser.source_id == Data.source_id)
                           .join(User, (SourceUser.user_id == User.id))
                           .join(TagSetUser, (TagSetUser.user_id == User.id))
                           .join(Text, Text.data_id == Data.id)
                           .outerjoin(Translation,
                                      (Translation.text_id == Text.id) & (Translation.target_language == Lang.en.name))
                           .join(Time, Time.data_id == Data.id)
                           .join(Fingerprint, Fingerprint.data_id == Data.id)
                           .join(Model, Model.tagset_id == TagSetUser.tagset_id)
                           .join(ModelUser, (ModelUser.model_id == Model.id) & (ModelUser.user_id == User.id))
                           .join(ModelSources,
                                 (ModelSources.model_id == Model.id) & (ModelSources.source_id == Data.source_id))
                           .outerjoin(Prediction,
                                      (Prediction.data_id == Data.id) & (Prediction.model_id == Model.id))
                           .filter(Prediction.id.is_(None))
                           .add_columns(Model.id, Text.text, Translation.translation,
                                        Fingerprint.fingerprint, Time.time)
                           .order_by(Model.id)  # ordered for better caching
                           .yield_per(1000),
                           session)
    _LOGGER.info('... Done')


@exclusive_task(app, Space.WORKER, trail=True, ignore_result=True)
def meta_pipeline(*_: Any) -> None:
    """
    Run all meta tasks in correct sequence
    Meta tasks must be enabled via FANLENS_META_MODULES.
    """
    _LOGGER.info('Starting full meta pipeline workers...')
    modules = [
        ('text', extract_text),
        ('time', extract_time),
        ('language', add_language),
        ('translation', add_translation),
        ('fingerprint', add_fingerprint),
        ('prediction', add_prediction),
    ]
    for module_key, module_function in modules:
        # assure correct order
        if module_key in app.conf['FANLENS_META_MODULES']:
            module_function()

    _LOGGER.info('Done meta pipeline workers...')


if __name__ == "__main__":
    meta_pipeline()
