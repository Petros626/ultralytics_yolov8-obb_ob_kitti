# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    #if ray.train._internal.session._get_session():  # replacement for deprecated ray.tune.is_session_enabled() <- use when ray v2.3.1
    if ray.train._internal.session.get_session():   # v2.41.0 replacement for depreacted _get_session() <- use when ray v2.41.0
        metrics = trainer.metrics
        session.report({**metrics, **{"epoch": trainer.epoch + 1}})


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
