"""Owner-lifetime handling shared by multiprocessing helpers."""

from multiprocessing import parent_process
import threading
from typing import Callable


def watch_parent_exit(on_exit: Callable[[], None]) -> None:
	"""Run cleanup in a daemon thread when this child's creator exits.

	Multiprocessing's parent sentinel follows the creating process even when
	the OS parent is a forkserver. Call only inside a multiprocessing child.
	"""

	parent = parent_process()

	def watch() -> None:
		parent.join()
		on_exit()

	threading.Thread(target=watch, name="lftomo-owner-watch", daemon=True).start()
