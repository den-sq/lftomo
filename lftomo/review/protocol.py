"""The canonical versioned experiment-review vocabulary."""

from .models import ReviewProtocol


REVIEW_PROTOCOL = ReviewProtocol()


def current_review_protocol() -> ReviewProtocol:
	"""Return an isolated copy of the current review protocol."""

	return REVIEW_PROTOCOL.model_copy(deep=True)
